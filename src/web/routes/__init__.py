"""Route modules for the Avis web dashboard.

Each module here defines an ``APIRouter`` plus any request /
response Pydantic models tightly coupled to its endpoints.
The factory in ``src.web.app`` mounts them.
"""

from . import chat, images, observations, pages, status, stream

__all__ = ["chat", "images", "observations", "pages", "status", "stream"]
