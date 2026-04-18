import sys
from pathlib import Path

# Ensure backend/ is on sys.path so bare imports (from database import ...)
# work whether the app is started with `uvicorn backend.app:app` from the
# project root or `uvicorn app:app` from within backend/.
_backend_dir = Path(__file__).resolve().parent
_backend_str = str(_backend_dir)
if _backend_str not in sys.path:
    sys.path.insert(0, _backend_str)
