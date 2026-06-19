"""FRIGID command line interface.

The CLI intentionally reuses the repository benchmark scripts for model loading
and metrics so that `frigid predict` and `frigid benchmark` stay comparable with
paper reproduction runs.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

from .artifacts import utc_stamp, write_manifest


def _repo_root() -> Path:
    env_root = os.environ.get("FRIGID_REPO_ROOT")
    if env_root:
        return Path(env_root).resolve()
    cwd = Path.cwd().resolve()
    if (cwd / "scripts" / "benchmark_spec2mol.py").exists():
        return cwd
    for parent in [cwd, *cwd.parents]:
        if (parent / "scripts" / "benchmark_spec2mol.py").exists():
            return parent
    raise SystemExit(
        "Could not find FRIGID repository root. Run from the repo or set FRIGID_REPO_ROOT."
    )


def _default_output(mode: str, scaler: str) -> str:
    return str(Path("runs") / f"{utc_stamp()}_{mode}_{scaler}")


def _run_script(repo_root: Path, script_name: str, argv: list[str]) -> int:
    script_path = repo_root / "scripts" / script_name
    if not script_path.exists():
        raise SystemExit(f"Missing script: {script_path}")
    old_argv = sys.argv[:]
    sys.argv = [str(script_path), *argv]
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    except SystemExit as exc:
        return int(exc.code or 0)
    finally:
        sys.argv = old_argv
    return 0


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Spec2Mol benchmark YAML config.")
    parser.add_argument("--mist-checkpoint", required=True, help="MIST encoder checkpoint.")
    parser.add_argument("--dlm-checkpoint", required=True, help="FRIGID DLM checkpoint.")
    parser.add_argument("--data-dir", help="Dataset root with labels.tsv, split.tsv and spec_files.")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--output-dir", help="Run output directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-shared-cross-attention", action="store_true")
    parser.add_argument("--fp-threshold", type=float)
    parser.add_argument("--softmax-temp", type=float)
    parser.add_argument("--randomness", type=float)


def _add_generation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scaler", choices=["ngboost", "iceberg"], default="ngboost")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--formula-matches", type=int)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--token-model", help="NGBoost token-length model path.")
    parser.add_argument("--sigma-lambda", type=float, default=3.0)
    parser.add_argument("--num-tokens-unmask", type=int, default=1)
    parser.add_argument("--confidence-temperature", type=float, default=1.0)
    parser.add_argument("--encoder-batch-size", type=int, default=1)
    parser.add_argument("--profile-generation", action="store_true")


def _add_iceberg_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--iceberg-gen-ckpt", help="ICEBERG generator checkpoint.")
    parser.add_argument("--iceberg-inten-ckpt", help="ICEBERG intensity checkpoint.")
    parser.add_argument("--iceberg-python-path", default="python")
    parser.add_argument("--iceberg-batch-size", type=int, default=8)
    parser.add_argument("--iceberg-gpu", nargs="+", type=int, default=[0])
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--num-unique-to-refine", type=int, default=16)
    parser.add_argument("--masks-per-molecule", type=int, default=4)


def _append_optional(argv: list[str], args: argparse.Namespace, *names: str) -> None:
    for name in names:
        value = getattr(args, name)
        if value is None:
            continue
        flag = "--" + name.replace("_", "-")
        if isinstance(value, list):
            argv.append(flag)
            argv.extend(str(item) for item in value)
        elif isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv.extend([flag, str(value)])


def _build_base_argv(args: argparse.Namespace, output_dir: str) -> list[str]:
    argv = [
        "--config",
        args.config,
        "--mist-checkpoint",
        args.mist_checkpoint,
        "--dlm-checkpoint",
        args.dlm_checkpoint,
        "--split",
        args.split,
        "--seed",
        str(args.seed),
        "--output-dir",
        output_dir,
    ]
    _append_optional(argv, args, "data_dir", "fp_threshold", "softmax_temp", "randomness")
    if args.use_shared_cross_attention:
        argv.append("--use-shared-cross-attention")
    return argv


def _build_ngboost_argv(args: argparse.Namespace, output_dir: str, max_spectra: int | None) -> list[str]:
    argv = _build_base_argv(args, output_dir)
    if max_spectra is not None:
        argv.extend(["--max-spectra", str(max_spectra)])
    _append_optional(
        argv,
        args,
        "batch_size",
        "formula_matches",
        "max_attempts",
        "token_model",
        "sigma_lambda",
        "num_tokens_unmask",
        "confidence_temperature",
        "encoder_batch_size",
        "profile_generation",
    )
    return argv


def _build_iceberg_argv(args: argparse.Namespace, output_dir: str, max_spectra: int | None) -> list[str]:
    if not args.iceberg_gen_ckpt or not args.iceberg_inten_ckpt:
        raise SystemExit("--scaler iceberg requires --iceberg-gen-ckpt and --iceberg-inten-ckpt")
    argv = _build_base_argv(args, output_dir)
    if max_spectra is not None:
        argv.extend(["--max-spectra", str(max_spectra)])
    _append_optional(
        argv,
        args,
        "batch_size",
        "token_model",
        "sigma_lambda",
        "iceberg_gen_ckpt",
        "iceberg_inten_ckpt",
        "iceberg_python_path",
        "iceberg_batch_size",
        "iceberg_gpu",
        "num_rounds",
        "num_unique_to_refine",
        "masks_per_molecule",
    )
    return argv


def _run_generation(args: argparse.Namespace, *, mode: str, max_spectra: int | None) -> int:
    repo_root = _repo_root()
    output_dir = args.output_dir or _default_output(mode, args.scaler)
    if args.scaler == "iceberg":
        script_name = "spec2mol_scaling.py"
        script_argv = _build_iceberg_argv(args, output_dir, max_spectra)
    else:
        script_name = "benchmark_spec2mol.py"
        script_argv = _build_ngboost_argv(args, output_dir, max_spectra)

    command = ["python", str(repo_root / "scripts" / script_name), *script_argv]
    exit_code = _run_script(repo_root, script_name, script_argv)
    write_manifest(
        output_dir,
        command=command,
        mode=mode,
        scaler=args.scaler,
        repo_root=repo_root,
        inputs={"config": args.config, "data_dir": args.data_dir, "split": args.split},
        checkpoints={
            "mist": args.mist_checkpoint,
            "dlm": args.dlm_checkpoint,
            "iceberg_gen": getattr(args, "iceberg_gen_ckpt", None),
            "iceberg_inten": getattr(args, "iceberg_inten_ckpt", None),
        },
        parameters=vars(args),
        status="success" if exit_code == 0 else "failed",
        exit_code=exit_code,
    )
    return exit_code


def cmd_predict(args: argparse.Namespace) -> int:
    return _run_generation(args, mode="predict", max_spectra=args.max_spectra)


def cmd_benchmark(args: argparse.Namespace) -> int:
    return _run_generation(args, mode="benchmark", max_spectra=args.max_spectra)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="frigid", description="FRIGID prediction and benchmark CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict", help="Run FRIGID on one or more spectra from a configured split.")
    _add_common_args(predict)
    _add_generation_args(predict)
    _add_iceberg_args(predict)
    predict.add_argument("--max-spectra", type=int, default=1)
    predict.set_defaults(func=cmd_predict)

    benchmark = subparsers.add_parser("benchmark", help="Run a reproducible FRIGID benchmark.")
    _add_common_args(benchmark)
    _add_generation_args(benchmark)
    _add_iceberg_args(benchmark)
    benchmark.add_argument("--max-spectra", type=int)
    benchmark.set_defaults(func=cmd_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
