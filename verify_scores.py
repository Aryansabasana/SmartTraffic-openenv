import math
from src.tasks import to_open_unit_interval, EPS

def test_normalization():
    test_cases = [
        (0.0, 0.01),
        (1.0, 0.99),
        (-0.5, 0.01),
        (1.5, 0.99),
        (0.5, 0.5),
        (0.0000005, 0.01),
        (0.9999995, 0.99),
        (float('nan'), 0.5),
        (float('inf'), 0.99),
        (float('-inf'), 0.01),
        (None, 0.5)
    ]

    print(f"Testing with EPS = {EPS}")
    print(f"{'Input':>10} | {'Output':>10} | {'Expected':>10} | {'Result':>8}")
    print("-" * 50)

    all_passed = True
    for val, expected in test_cases:
        res = to_open_unit_interval(val)
        passed = math.isclose(res, expected, rel_tol=1e-9)
        print(f"{str(val):>10} | {res:>10.6f} | {expected:>10.6f} | {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n[PASS] All normalization tests passed!")
    else:
        print("\n[FAIL] Some normalization tests failed.")

if __name__ == "__main__":
    test_normalization()
