from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repository unit tests")
    parser.add_argument("--path", type=str, default="tests", help="Test root directory")
    parser.add_argument("--pattern", type=str, default="test_*.py", help="Test file pattern")
    parser.add_argument("--verbosity", type=int, default=2, help="unittest verbosity level")
    parser.add_argument("--failfast", action="store_true", help="Stop on first failure")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    test_path = Path(args.path)
    if not test_path.exists():
        print(f"Test path does not exist: {test_path}", file=sys.stderr)
        return 2

    suite = unittest.defaultTestLoader.discover(
        start_dir=str(test_path),
        pattern=args.pattern,
    )
    if suite.countTestCases() == 0:
        print("No tests discovered.", file=sys.stderr)
        return 1

    runner = unittest.TextTestRunner(verbosity=args.verbosity, failfast=args.failfast)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
