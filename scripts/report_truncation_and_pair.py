"""Price a decode honestly: truncation against the cap, and the paired contrast.

Two questions this answers that a metrics file does not.

1. **What does the cap cost?** ``truncated`` is a per-row boolean, so a run
   reports the truncation rate at the cap it happened to use and nothing else.
   A row that finished in 40 s would also have finished under any cap above
   40 s, so one uncapped-enough run gives the whole truncation-versus-cap
   curve, and the curve is what tells you whether the number you are holding is
   censored.

2. **Is the pair still the pair?** Two arms are comparable only over the spec
   names they share, so the McNemar table is built on the intersection and the
   discordance is printed with it.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            rows[row["spec_name"]] = row
    return rows


def exact_top1(row: dict) -> bool:
    candidates = row.get("candidates") or []
    return bool(candidates) and bool(candidates[0].get("exact_connectivity"))


def tanimoto_top1(row: dict) -> float:
    candidates = row.get("candidates") or []
    if not candidates:
        return 0.0
    return float(candidates[0].get("target_fingerprint_tanimoto") or 0.0)


def exact_mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail / (2.0**n))


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half, centre + half)


def describe(name: str, rows: dict[str, dict], caps: list[float]) -> dict:
    runtimes = sorted(row["runtime_seconds"] for row in rows.values())
    n = len(rows)
    hits = sum(exact_top1(row) for row in rows.values())
    returned = sum(1 for row in rows.values() if row.get("candidates"))
    truncated = sum(1 for row in rows.values() if row.get("truncated"))
    lo, hi = wilson(hits, n)
    summary = {
        "run": name,
        "spectra": n,
        "exact_top1": hits / n,
        "exact_top1_count": hits,
        "exact_top1_wilson95": [lo, hi],
        "tanimoto_top1_all_spectra": sum(tanimoto_top1(r) for r in rows.values()) / n,
        "candidate_return_rate": returned / n,
        "truncated_at_run_cap": truncated,
        "runtime_seconds_total": sum(runtimes),
        "runtime_seconds_mean": sum(runtimes) / n,
        "runtime_seconds_median": runtimes[n // 2],
        "runtime_seconds_p90": runtimes[min(n - 1, int(0.9 * n))],
        "runtime_seconds_max": runtimes[-1],
        # A row that finished in t seconds would have been truncated by any cap
        # below t. Rows already truncated at the run's own cap are censored and
        # counted as truncated at every cap.
        "truncation_rate_by_cap": {
            str(int(cap)): sum(
                1
                for row in rows.values()
                if row.get("truncated") or row["runtime_seconds"] > cap
            )
            / n
            for cap in caps
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--dreams", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    parser.add_argument(
        "--caps",
        type=float,
        nargs="+",
        default=[60, 120, 300, 600, 900, 1200, 1800, 3600],
    )
    args = parser.parse_args()

    oracle = load(args.oracle)
    dreams = load(args.dreams)
    shared = sorted(set(oracle) & set(dreams))

    report = {
        "arms": [
            describe("oracle", oracle, args.caps),
            describe("dreams", dreams, args.caps),
        ],
        "paired_spectra": len(shared),
        "oracle_only_spectra": len(set(oracle) - set(dreams)),
        "dreams_only_spectra": len(set(dreams) - set(oracle)),
    }

    both = sum(1 for s in shared if exact_top1(oracle[s]) and exact_top1(dreams[s]))
    oracle_only = sum(
        1 for s in shared if exact_top1(oracle[s]) and not exact_top1(dreams[s])
    )
    dreams_only = sum(
        1 for s in shared if not exact_top1(oracle[s]) and exact_top1(dreams[s])
    )
    neither = len(shared) - both - oracle_only - dreams_only
    delta = (both + oracle_only) / len(shared) - (both + dreams_only) / len(shared)
    report["paired"] = {
        "both_correct": both,
        "oracle_only": oracle_only,
        "dreams_only": dreams_only,
        "neither": neither,
        "discordance": f"{oracle_only}-{dreams_only}",
        "exact_mcnemar_p": exact_mcnemar(oracle_only, dreams_only),
        "paired_exact_top1_delta_pp": 100.0 * delta,
        "paired_tanimoto_top1_delta": sum(
            tanimoto_top1(oracle[s]) - tanimoto_top1(dreams[s]) for s in shared
        )
        / len(shared),
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json:
        args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
