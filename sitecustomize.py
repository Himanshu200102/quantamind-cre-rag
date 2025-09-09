
import sys
try:
    import pysqlite3 as sqlite3  # noqa: F401
    sys.modules["sqlite3"] = sqlite3
except Exception:
    pass
