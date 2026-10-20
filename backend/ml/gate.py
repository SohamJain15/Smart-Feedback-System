"""Quality gate: promote a candidate only if its aggregate F1 is strictly higher.

Exit codes: 0 = promote, 10 = reject, 1 = error.
"""
import argparse
import json
import sys
from pathlib import Path


def decide(candidate_f1: float, current_f1: float) -> tuple[bool, str]:
    if candidate_f1 > current_f1:
        return True, f"PROMOTE: candidate F1 {candidate_f1:.4f} > live F1 {current_f1:.4f}"
    return False, f"REJECT: candidate F1 {candidate_f1:.4f} is not higher than live F1 {current_f1:.4f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare candidate and live F1")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--current-f1", type=float, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        candidate = float(json.loads(args.metrics.read_text())["aggregate_f1"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"ERROR: cannot read candidate metrics: {error}")
        return 1
    promote, reason = decide(candidate, args.current_f1)
    print(reason)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(reason + "\n")
    return 0 if promote else 10


if __name__ == "__main__":
    sys.exit(main())
