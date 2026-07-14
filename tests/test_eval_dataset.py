from src.eval.golden import load_golden_cases


def test_golden_cases_loaded():
    cases = load_golden_cases()
    assert len(cases) >= 5
    assert any(c.get("expect_greeting") for c in cases)
