import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from api_client import APIError, CandidateAPIClient  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Restore all candidates in your private workspace to the original test data."
    )
    parser.parse_args()
    try:
        with CandidateAPIClient() as api:
            api.reset_workspace()
    except APIError as exc:
        print(f"Workspace reset not confirmed: {exc}", file=sys.stderr)
        return 1
    print("Test workspace reset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
