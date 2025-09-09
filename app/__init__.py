# app/__init__.py
try:
    import pysqlite3 as sqlite3  # noqa: F401
    import sys
    sys.modules["sqlite3"] = sqlite3
except Exception:
    pass
