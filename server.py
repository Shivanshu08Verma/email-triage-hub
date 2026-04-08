"""
server.py – Compatibility shim for direct `python server.py` execution.
The real application lives in server/app.py.
"""
from server.app import app, main  # noqa: F401

if __name__ == "__main__":
    main()
