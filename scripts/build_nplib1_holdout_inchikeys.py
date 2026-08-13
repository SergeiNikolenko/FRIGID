"""Write the connectivity blocks a corpus-scale run must never train on.

The list that shipped, ``data/nplib1_test_inchikeys.csv``, holds the 701 blocks of
the locked 803 test split and nothing else. Every panel we iterate on --- the
clean 321 --- is drawn from the val fold, so that file leaves all 320 of its
blocks eligible. Measured against 10 row groups of the fp2mol corpus, 23 of those
320 structures occur in the first 10,000,000 corpus molecules, which is the size
of one paired stage-1 experiment. A headline read on a panel whose structures the
run trained on is not a headline.

This writes the union of the test and val connectivity blocks, which is what a
run has to exclude for both the locked 803 and the clean 321 to stay honest.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
# The list is read from inside the repository, so it is written there: a copy
# that lives only beside the repository is a copy a `git clone` does not carry.
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def blocks(metadata: Path) -> set[str]:
    table = pd.read_csv(metadata)
    return {str(value) for value in table["inchikey_first_block"].dropna()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "nplib1_holdout_inchikeys_v2.csv",
    )
    parser.add_argument("--split", action="append", default=None)
    arguments = parser.parse_args()

    splits = arguments.split or ["test", "val"]
    held: set[str] = set()
    for split in splits:
        found = blocks(arguments.processed / split / "metadata.csv")
        print(f"{split}: {len(found)} connectivity blocks")
        held |= found

    train = blocks(arguments.processed / "train" / "metadata.csv")
    print(f"train: {len(train)} connectivity blocks, {len(train & held)} shared with the holdout")

    ordered = sorted(held)
    frame = pd.DataFrame({"inchikey": ordered})
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(arguments.output, index=False)
    digest = hashlib.sha256(arguments.output.read_bytes()).hexdigest()
    print(f"wrote {len(ordered)} blocks to {arguments.output}")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
