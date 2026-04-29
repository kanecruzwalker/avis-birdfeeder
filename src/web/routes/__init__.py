"""Route modules for the Avis web dashboard.

Each module here defines an ``APIRouter`` plus any request /
response Pydantic models tightly coupled to its endpoints.
The factory in ``src.web.app`` mounts them.
"""

from . import images, observations, status, stream

__all__ = ["images", "observations", "status", "stream"]
