"""
scripts/compile_hailo_hef.py

Documents and partially automates the pipeline from trained PyTorch weights
to a Hailo .hef file ready for deployment on the Pi.

This script handles the parts that run on the laptop (ONNX export and
calibration data export). The DFC compilation steps must run inside the
Hailo Docker container — this script prints the exact commands.

Full pipeline:
    Step 1 (laptop)  — Export frozen_extractor.pt → ONNX
    Step 2 (laptop)  — Export 64 real calibration images → calib_images.npy
    Step 3 (Docker)  — Parse ONNX → HAR
    Step 4 (Docker)  — Quantize HAR with calibration data → quantized HAR
    Step 5 (Docker)  — Compile quantized HAR → .hef
    Step 6 (laptop)  — Copy .hef from Docker to Pi via SCP

Compilation environment:
    Docker image: hailo8_ai_sw_suite_2025-07:1
    DFC version:  3.32.0  (targets HailoRT 4.22.0)
    Pi runtime:   HailoRT 4.23.0  (forward-compatible — confirmed)

Usage:
    # Run Steps 1 + 2 on laptop (requires venv with torch, timm, onnx, pandas, Pillow)
    cd avis-birdfeeder
    python scripts/compile_hailo_hef.py --export

    # Print Docker commands for Steps 3–5
    python scripts/compile_hailo_hef.py --print-docker-commands

    # Full run (Steps 1+2 + prints Docker commands)
    python scripts/compile_hailo_hef.py

Key technical decisions recorded:
    - Backbone only: num_classes=0, global_pool='avg' — no classification head
    - ONNX end node: /global_pool/pool/GlobalAveragePool (Flatten is unsupported)
    - SE block fix: pre_quantization_optimization with global_avgpool_reduction
      division_factors=[7,7] — required because EfficientNet SE avgpool shift
      delta (3.53) exceeds the HAILO8L hardware limit of 2.0
    - Calibration: 64 real SD bird images from visual_train.csv, uint8 224x224
      normalized to float32 [0,1] for the quantizer
    - Output file: models/visual/efficientnet_b0_avis_v2.hef (13 MB)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT      = Path(__file__).resolve().parent.parent
EXTRACTOR_PATH = REPO_ROOT / "models" / "visual" / "frozen_extractor.pt"
SPLITS_DIR     = REPO_ROOT / "data" / "splits"
ONNX_PATH      = REPO_ROOT / "efficientnet_b0_avis_features.onnx"
CALIB_PATH     = REPO_ROOT / "calib_images.npy"
HEF_OUT_PATH   = REPO_ROOT / "models" / "visual" / "efficientnet_b0_avis_v2.hef"

# Model script content for SE block avgpool fix
SE_FIX_SCRIPT = (
    "pre_quantization_optimization(global_avgpool_reduction, "
    "layers=efficientnet_b0_avis/avgpool1, division_factors=[7, 7])\n"
)


def step1_export_onnx() -> None:
    """Export frozen_extractor.pt → ONNX (run on laptop)."""
    try:
        import timm
        import torch
    except ImportError:
        print("ERROR: torch and timm required for ONNX export.")
        print("       Run: pip install torch timm onnx")
        return

    if not EXTRACTOR_PATH.exists():
        print(f"ERROR: {EXTRACTOR_PATH} not found.")
        print("       Run notebooks/visual_efficientnet.ipynb cell 28 first.")
        return

    print(f"[Step 1] Exporting {EXTRACTOR_PATH.name} → ONNX...")

    ckpt  = torch.load(str(EXTRACTOR_PATH), map_location="cpu")
    model = timm.create_model(
        ckpt["model_name"],
        pretrained=False,
        num_classes=0,
        global_pool=ckpt["global_pool"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        str(ONNX_PATH),
        input_names=["input"],
        output_names=["features"],
        opset_version=11,
    )

    # Verify output shape
    with torch.no_grad():
        out = model(dummy)

    size_mb = ONNX_PATH.stat().st_size / 1024 / 1024
    print(f"  Exported: {ONNX_PATH}  ({size_mb:.1f} MB)")
    print(f"  Output shape: {out.shape}  (expected torch.Size([1, 1280]))")

    if out.shape != torch.Size([1, 1280]):
        print(f"  WARNING: unexpected output shape {out.shape}")
    else:
        print("  Shape verified. ONNX export complete.")


def step2_export_calibration(n_images: int = 64) -> None:
    """Export calibration images from training split → calib_images.npy (run on laptop)."""
    try:
        import numpy as np
        import pandas as pd
        from PIL import Image
    except ImportError:
        print("ERROR: numpy, pandas, Pillow required.")
        return

    csv_path = SPLITS_DIR / "visual_train.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run data splits first.")
        return

    print(f"[Step 2] Exporting {n_images} calibration images from visual_train.csv...")

    df = pd.read_csv(csv_path)
    sample = df.sample(n=min(n_images * 2, len(df)), random_state=42)

    images = []
    skipped = 0
    for path_str in sample["file_path"]:
        if len(images) >= n_images:
            break
        try:
            img = Image.open(path_str).convert("RGB").resize((224, 224))
            images.append(np.array(img, dtype=np.uint8))
        except Exception:
            skipped += 1

    if len(images) < n_images:
        print(f"  WARNING: only {len(images)} images loaded ({skipped} skipped)")

    calib = np.stack(images[:n_images])
    np.save(str(CALIB_PATH), calib)
    print(f"  Saved: {CALIB_PATH}  shape={calib.shape} dtype={calib.dtype}")
    print(f"  Range: {calib.min()}–{calib.max()}  (raw uint8 — normalized to [0,1] in Docker)")


def print_docker_commands() -> None:
    """Print the exact commands to run inside the Hailo Docker container."""
    print("\n" + "=" * 70)
    print("Steps 3–5: Run inside Hailo Docker container")
    print("=" * 70)
    print("""
# Load and start the Docker container (run on WSL2 Ubuntu or native Linux)
cd ~/hailo_suite
sudo docker load -i hailo8_ai_sw_suite_2025-07.tar.gz
sudo docker run -it --rm \\
  -v ~/hailo_suite/shared_with_docker:/workspace \\
  hailo8_ai_sw_suite_2025-07:1 bash

# Inside container — copy assets from Windows/shared volume
cp /workspace/efficientnet_b0_avis_features.onnx /tmp/
cp /workspace/calib_images.npy /tmp/

# Step 3: Parse ONNX to HAR
# --end-node-names stops before Flatten (unsupported Hailo op)
hailo parser onnx /tmp/efficientnet_b0_avis_features.onnx \\
  --hw-arch hailo8l \\
  --net-name efficientnet_b0_avis \\
  --har-path /tmp/efficientnet_b0_avis.har \\
  --end-node-names /global_pool/pool/GlobalAveragePool \\
  -y

# Step 4: Quantize with SE block fix and real calibration data
# The model script fixes the SE avgpool shift delta (3.53 > 2.0 hardware limit)
cat > /tmp/avis_se_fix.alls << 'EOF'
pre_quantization_optimization(global_avgpool_reduction, layers=efficientnet_b0_avis/avgpool1, division_factors=[7, 7])
EOF

python3 << 'PYEOF'
import numpy as np
from hailo_sdk_client import ClientRunner

runner = ClientRunner(har="/tmp/efficientnet_b0_avis.har")
runner.load_model_script("/tmp/avis_se_fix.alls")

# Normalize uint8 calibration images to float32 [0, 1] (matches training preprocessing)
calib = np.load("/tmp/calib_images.npy").astype(np.float32) / 255.0
print(f"Calibration: shape={calib.shape} range=[{calib.min():.3f}, {calib.max():.3f}]")
runner.optimize(calib)
runner.save_har("/tmp/efficientnet_b0_avis_quantized.har")
print("Quantization complete")
PYEOF

# Step 5: Compile to HEF (~15-20 minutes, CPU-only in Docker)
hailo compiler /tmp/efficientnet_b0_avis_quantized.har \\
  --hw-arch hailo8l \\
  --output-dir /tmp/
# Output: /tmp/efficientnet_b0_avis.hef  (~13 MB)

# Copy HEF out of container
cp /tmp/efficientnet_b0_avis.hef /workspace/efficientnet_b0_avis_v2.hef
exit
""")

    print("=" * 70)
    print("Step 6: Transfer HEF to Pi (run on WSL2 Ubuntu after exiting Docker)")
    print("=" * 70)
    print("""
# From WSL2 Ubuntu:
scp ~/hailo_suite/shared_with_docker/efficientnet_b0_avis_v2.hef \\
    birdfeeder01@192.168.4.76:/mnt/data/avis-birdfeeder/models/visual/

# Verify on Pi:
# python3 -c "
# from hailo_platform import HEF
# hef = HEF('models/visual/efficientnet_b0_avis_v2.hef')
# for ng in hef.get_network_group_names():
#     print(ng, [(v.name, v.shape) for v in hef.get_input_vstream_infos(ng)])
# "
# Expected: efficientnet_b0_avis [('efficientnet_b0_avis/input_layer1', (224, 224, 3))]
""")

    print("=" * 70)
    print("SE block fix explained")
    print("=" * 70)
    print("""
EfficientNet-B0 Squeeze-and-Excitation blocks contain a GlobalAveragePool
that compresses a 7x7 spatial feature map to 1x1. The spatial reduction
ratio produces a shift delta of 3.53 in the quantizer's fixed-point math,
exceeding the HAILO8L hardware limit of 2.0.

The fix uses the pre_quantization_optimization command to pre-reduce the
spatial dimensions before quantization, keeping the shift delta within limits.
The command name (pre_quantization_optimization) and feature name
(global_avgpool_reduction) are specific to DFC 3.32.0 — the correct command
string was found by introspecting:
    hailo_sdk_client.sdk_backend.script_parser.commands.SupportedCommands
""")


def main() -> None:
    parser = argparse.ArgumentParser(description="Avis Hailo HEF compilation pipeline")
    parser.add_argument(
        "--export",
        action="store_true",
        help="Run Steps 1+2 (ONNX export + calibration data export)",
    )
    parser.add_argument(
        "--print-docker-commands",
        action="store_true",
        help="Print Docker commands for Steps 3–5 only",
    )
    parser.add_argument(
        "--n-calib",
        type=int,
        default=64,
        help="Number of calibration images to export (default 64)",
    )
    args = parser.parse_args()

    if args.print_docker_commands:
        print_docker_commands()
        return

    # Default: run both laptop steps and print Docker commands
    step1_export_onnx()
    step2_export_calibration(args.n_calib)
    print_docker_commands()

    print("\nNext: copy efficientnet_b0_avis_features.onnx and calib_images.npy")
    print(f"      to ~/hailo_suite/shared_with_docker/ and follow the Docker steps above.")
    print(f"\n      HEF will be saved to: {HEF_OUT_PATH}")


if __name__ == "__main__":
    main()
