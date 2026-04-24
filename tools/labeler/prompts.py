"""Prompt templates and species reference for the labeling assistant.

Keeping prompts in a dedicated module lets us:
- Version them cleanly (PROMPT_VERSION constant goes in each PreLabel record)
- Iterate on prompt wording without touching agent code
- Write unit tests that validate prompt construction
- Review prompt changes in isolation during code review

When you edit this file, bump PROMPT_VERSION so prior pre-label runs can be
distinguished from new ones in pre_labels.jsonl.
"""
from __future__ import annotations

# ── Prompt version ────────────────────────────────────────────────────────────

# Bump this whenever the prompt text changes. Persisted on every PreLabel
# record so we can identify which prompt generated any given label.
# Format: vMAJOR.MINOR — MAJOR for semantic changes (e.g. new species list,
# different classification task), MINOR for wording tweaks.
PROMPT_VERSION = "v1.0"


# ── Species reference ─────────────────────────────────────────────────────────

# Full descriptions of each SD region species, written for zero-shot
# recognition by a capable vision LLM. Each entry describes distinguishing
# VISUAL features only — size, colour, markings, shape — because the LLM
# cannot hear the bird. Behavioural cues that would help a human birder
# (call type, flight pattern) are omitted.
#
# Sources: Merlin Bird ID descriptions, Cornell All About Birds, cross-
# referenced against the images in NABirds training data for each species.
#
# Kept in sync manually with configs/species.yaml. If species are added or
# removed from that file, update this dict AND bump PROMPT_VERSION.
SPECIES_REFERENCE: dict[str, str] = {
    "HOFI": (
        "House Finch — small songbird, ~14cm. Males have bright red on head, "
        "chest, and rump; brown-streaked body. Females are all brown with "
        "fine streaks on the chest. Often in groups at feeders."
    ),
    "MODO": (
        "Mourning Dove — medium-large, ~30cm, slim profile with a long "
        "pointed tail. Soft grey-brown overall with a small round head and "
        "black spots on the wings. Relatively large compared to other "
        "feeder visitors."
    ),
    "ANHU": (
        "Anna's Hummingbird — tiny, ~10cm, stocky for a hummingbird. Males "
        "have iridescent rose-pink head and throat (gorget); females are "
        "greenish-grey with a small pink throat patch. Usually hovering, "
        "not perched."
    ),
    "CAVI": (
        "California Scrub-Jay — medium-large, ~29cm. Bright blue above with "
        "a grey-brown back, white throat bordered by a partial blue "
        "'necklace', white belly. Bold and conspicuous, often dominating "
        "the feeder."
    ),
    "MOCH": (
        "Northern Mockingbird — medium, ~25cm, slim with a long tail. "
        "Grey above, paler below, with two white wing bars and large white "
        "patches in the wings visible in flight. Long, narrow bill."
    ),
    "AMRO": (
        "American Robin — medium-large, ~25cm. Warm orange-red breast, "
        "dark grey back and head, white eye-arcs, yellow bill. Distinctive "
        "upright posture."
    ),
    "SOSP": (
        "Song Sparrow — small, ~15cm. Brown streaky back, heavily streaked "
        "breast with a dark central spot, grey face with a brown 'moustache' "
        "stripe. Long rounded tail often pumped while flying."
    ),
    "LEGO": (
        "Lesser Goldfinch — very small, ~11cm. Males have black cap, bright "
        "yellow underparts, and black-and-white wings. Females and immatures "
        "are olive-green above and pale yellow below. Small pointed bill."
    ),
    "DOWO": (
        "Downy Woodpecker — small, ~17cm, smallest North American woodpecker. "
        "Black-and-white checkered wings, white back, white underparts. "
        "Males have a small red spot on the back of the head. Short stubby bill."
    ),
    "WREN": (
        "House Wren — small, ~12cm. Plain brown overall with fine barring "
        "on wings and tail, no strong facial pattern. Long slightly curved "
        "bill. Often holds tail cocked up over the back."
    ),
    "AMCR": (
        "American Crow — large, ~45cm. All glossy black including legs and "
        "bill, heavy straight bill, fan-shaped tail. Much larger than any "
        "other common feeder bird."
    ),
    "SPTO": (
        "Spotted Towhee — medium, ~21cm. Males have jet-black head and "
        "upperparts with white spots on the wings and back, rich rufous "
        "sides, white belly. Females are similar but grey-brown where "
        "males are black. Red eye."
    ),
    "BLPH": (
        "Black Phoebe — small, ~17cm. Entirely sooty black head, back, "
        "chest, and wings with a clean white belly. Slim with a long tail "
        "that is often wagged."
    ),
    "HOSP": (
        "House Sparrow — small, ~16cm. Males have grey crown, chestnut "
        "nape, black bib, and white cheeks. Females are plain dusty brown "
        "with a pale eyebrow stripe. Stocky build, thick conical bill."
    ),
    "EUST": (
        "European Starling — medium, ~22cm. In breeding plumage: glossy "
        "black with purple and green iridescence, yellow bill. In "
        "non-breeding: heavily speckled with white spots on black, dark bill. "
        "Short tail, pointed wings."
    ),
    "WCSP": (
        "White-crowned Sparrow — small-medium, ~17cm. Plain grey underparts, "
        "brown streaky back, and a striking black-and-white striped crown "
        "(adults) or brown-and-tan striped crown (immatures). Pink or "
        "yellowish bill."
    ),
    "HOORI": (
        "Hooded Oriole — medium, ~20cm, slim with a long tail. Males are "
        "bright orange-yellow overall with black throat, back, wings, and "
        "tail. Females and immatures are dull yellow with greyish back and "
        "dark wings. Attracted to oranges and nectar."
    ),
    "WBNU": (
        "White-breasted Nuthatch — small, ~14cm, stocky with a short tail. "
        "Blue-grey back, black cap (males) or grey cap (females), clean "
        "white face and underparts, rusty wash on lower belly. Often seen "
        "climbing head-down on trees."
    ),
    "OCWA": (
        "Orange-crowned Warbler — small, ~12cm. Plain olive-yellow overall "
        "with faint darker streaks on the breast and a thin broken eye-ring. "
        "The 'orange crown' is rarely visible. Thin pointed bill."
    ),
    "YRUM": (
        "Yellow-rumped Warbler — small, ~13cm. Distinguishing feature is a "
        "bright yellow rump patch. Grey-brown back with streaks, white "
        "throat (Myrtle form) or yellow throat (Audubon's form), yellow "
        "patches on sides. Active, often flitting."
    ),
}


def format_species_reference() -> str:
    """Render the species reference dict as a prompt-friendly text block."""
    lines = []
    for code, desc in SPECIES_REFERENCE.items():
        lines.append(f"- {code}: {desc}")
    return "\n".join(lines)


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT_TEMPLATE = """You are an expert ornithologist helping label bird images from an
automated birdfeeder monitoring system.

## Camera setup

The images come from a Raspberry Pi camera mounted approximately 30cm directly
above a wooden birdfeeder tray. The tray contains mixed birdseed and often has
orange slices clamped to a metal bar at one edge (bait for orioles). The
background is typically green ivy. The camera is in continuous autofocus.
You are viewing birds from ABOVE — you will see the back and top of the bird,
not the side profile typical in field guides.

## Your task

For each image, identify whether a bird is present, and if so, which species.
Respond with structured output only.

## Species reference

These are the only species that appear in this deployment. Use these 4-letter
codes or a sentinel:

{species_reference}

Sentinels:
- NONE: no bird visible (empty feeder, just seed and orange)
- UNKNOWN: a bird is visible but the species cannot be identified with
  reasonable confidence (blurry, partial view, unfamiliar species)

## Rules

1. Look at the IMAGE. Your classification should be driven by what you see.

2. You may be provided with an AUDIO HINT — a species code that BirdNET
   detected in audio around the capture time. Treat this as weak context only.
   Do NOT copy the audio hint if visual evidence contradicts it. A bird
   calling in a nearby tree may not be the bird in the image, or there may
   be no bird in the image at all despite an audio detection.

3. If no bird is visible, species_code is NONE with high confidence (0.9+).
   An empty feeder is an important and valid label.

4. If a bird is visible but you are not sure which species:
   - If you can reasonably narrow it to 2-3 candidates, pick your best guess
     and list the alternatives in uncertain_between with confidence 0.4-0.7.
   - If you truly cannot guess, use UNKNOWN with confidence reflecting how
     confident you are that it's a bird at all (not a leaf shadow or artifact).

5. Confidence should reflect your actual visual certainty:
   - 0.9+: clearly that species based on visible features
   - 0.7-0.9: likely but some ambiguity
   - 0.4-0.7: best guess among a few plausible species
   - Below 0.4: really uncertain — consider UNKNOWN

6. Remember: the view is from above. Adjust your mental species key for
   top-down silhouettes. Back colour, head shape, tail length and wing
   pattern are your most reliable features from this angle. Breast markings
   will usually NOT be visible.

7. In your reasoning field, briefly describe what you see — this is valuable
   for the human reviewer even if your species guess turns out to be wrong.
"""


def build_system_prompt() -> str:
    """Construct the final system prompt with species reference interpolated.

    Returns the full system prompt string ready to pass to the model.
    """
    return SYSTEM_PROMPT_TEMPLATE.format(
        species_reference=format_species_reference(),
    )


# ── User message template ─────────────────────────────────────────────────────


def build_user_message_text(audio_hint: str | None, audio_confidence: float | None) -> str:
    """Construct the per-image user message text.

    The image itself is attached separately as an image_url content block.
    This function returns the TEXT that accompanies the image in the
    HumanMessage.
    """
    if audio_hint is None:
        return (
            "Classify the bird (or empty feeder) visible in this image.\n"
            "No audio hint available — base your answer on the image only."
        )

    conf_str = f" (confidence {audio_confidence:.2f})" if audio_confidence is not None else ""
    return (
        "Classify the bird (or empty feeder) visible in this image.\n"
        f"Audio hint: BirdNET detected {audio_hint}{conf_str} "
        "around the capture time. Treat this as weak context — if the image "
        "shows a different bird or no bird, follow the visual evidence."
    )