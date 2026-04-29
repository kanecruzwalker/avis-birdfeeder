"""Route modules for the Avis web dashboard.

Each module here defines an ``APIRouter`` plus any request /
response Pydantic models that are tightly coupled to its endpoints.
The factory in ``src.web.app`` mounts them.

Modules
-------
``status``       — ``/api/status`` and ``/health``
``observations`` — ``/api/observations``, ``/api/observations/{id}``
``stream``       — ``/api/stream``, ``/api/frame``  (PR 3)
``images``       — ``/api/observations/{id}/image/{variant}``  (PR 4)
``chat``         — ``/api/ask``  (PR 8)
"""

from . import images, observations, status, stream

__all__ = ["images", "observations", "status", "stream"]
