# app/utils/timing.py
from __future__ import annotations

import time
import logging
from contextlib import contextmanager
from typing import Optional, Dict

log = logging.getLogger(__name__)

def fmt_ms(ms: float) -> str:
    if ms >= 1000:
        return f"{ms/1000:.2f}s"
    return f"{ms:.0f}ms"

@contextmanager
def timed(label: str, bucket: Optional[Dict[str, float]] = None):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = (time.perf_counter() - t0) * 1000.0
        if bucket is not None:
            bucket[label] = bucket.get(label, 0.0) + dt
        log.info(f"[TIME] {label:>28} ... {fmt_ms(dt)}")

class Stopwatch:
    """Group multiple timed stages and print a compact summary."""
    def __init__(self, title: str):
        self.title = title
        self.stages: Dict[str, float] = {}
        self.t0 = time.perf_counter()

    @contextmanager
    def stage(self, label: str):
        with timed(label, self.stages):
            yield

    def done(self):
        total_ms = (time.perf_counter() - self.t0) * 1000.0
        lines = [f"[TIME] {self.title} (total) ... {fmt_ms(total_ms)}"]
        for k, v in self.stages.items():
            lines.append(f"       └─ {k:>24} ... {fmt_ms(v)}")
        logging.getLogger(__name__).info("\n".join(lines))
        return total_ms
