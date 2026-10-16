"""Golden-trace regression test: the simulation must stay bit-for-bit identical."""
import pytest

from golden_scenarios import load_golden, run_all

_results = None


def _run():
    global _results
    if _results is None:
        _results = run_all()
    return _results


GOLDEN = load_golden()


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_scenario_matches_golden(name):
    expected = GOLDEN[name]
    actual = _run()[name]
    if actual["final"] != expected["final"]:
        first_bad = next(
            (i for i, (a, e) in enumerate(zip(actual["checkpoints"], expected["checkpoints"]))
             if a != e), None)
        frame = None if first_bad is None else first_bad * 30
        pytest.fail(
            f"{name}: simulation diverged (first differing checkpoint near frame {frame}); "
            f"expected result {expected['result']}, got {actual['result']}")
