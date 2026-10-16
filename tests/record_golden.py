"""Re-record golden traces. Only do this when a behaviour change is intended."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from golden_scenarios import GOLDEN_PATH, run_all  # noqa: E402

if __name__ == "__main__":
    results = run_all()
    with open(GOLDEN_PATH, "w") as fh:
        json.dump(results, fh, indent=1, sort_keys=True)
    for name, r in results.items():
        print(f"{name:42s} frames={r['frames']:5d} result={r['result']} events={r['events']}")
    print(f"wrote {GOLDEN_PATH}")
