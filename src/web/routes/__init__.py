"""Route modules for the Avis web dashboard.

Each module here defines an ``APIRouter`` plus any request /
response Pydantic models that are tightly coupled to its endpoints.
The factory in ``src.web.app`` mounts them.

Modules
-------
``status``       — ``/api/status`` and ``/health``
``observations`` — ``/api/observations``, ``/api/observations/{id}``
``stream``       — ``/api/stream``, ``/api/frame``  (PR 3)
``chat``         — ``/api/ask``  (PR 8)
"""

from . import observations, status, stream

__all__ = ["observations", "status", "stream"]
