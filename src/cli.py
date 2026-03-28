"""CLI: launch Streamlit chapters with environment defaults."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "src" / "app.py"


def _run_streamlit(env_updates: dict, port: int) -> int:
    env = os.environ.copy()
    env.update(env_updates)
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP),
        f"--server.port={port}",
        "--browser.gatherUsageStats=false",
    ]
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Random geometry and criticality — Chapters A–E.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("chapter-a", help="Carpet, Vicsek, or percolation + SRW viz")
    a.add_argument(
        "--graph",
        choices=("carpet", "vicsek", "percolation"),
        default="carpet",
        help="Graph family (default: carpet).",
    )
    a.add_argument("--depth", type=int, default=4, help="Recursion depth for carpet/Vicsek.")
    a.add_argument(
        "--perc-size",
        type=int,
        default=24,
        help="Grid side for bond percolation.",
    )
    a.add_argument(
        "--p-open",
        type=float,
        default=0.55,
        help="Bond occupation probability for percolation.",
    )
    a.add_argument("--seed", type=int, default=42, help="RNG seed for percolation instance.")
    # Lévy modifier
    a.add_argument(
        "--levy-alpha",
        type=float,
        default=0.0,
        help="Lévy exponent α ∈ (0,2). Set > 0 to enable long-range edges. 0 = disabled.",
    )
    a.add_argument(
        "--levy-p",
        type=float,
        default=0.05,
        help="Edge probability at nearest-neighbour length scale (default 0.05).",
    )
    # Biased walk presets
    a.add_argument(
        "--bias",
        choices=("none", "toward-center", "away-center", "hub-seeking", "hub-avoiding"),
        default="none",
        help="Walk bias preset (default: none = uniform SRW).",
    )
    a.add_argument(
        "--bias-strength",
        type=float,
        default=3.0,
        help="Strength parameter for the bias preset (default 3.0).",
    )
    a.add_argument("--port", type=int, default=8501, help="Streamlit server port.")

    b = sub.add_parser("chapter-b", help="Occupation, first passage, or traps")
    b.add_argument(
        "--mode",
        choices=("occupation", "first_passage", "traps"),
        default="occupation",
        help="Observable mode.",
    )
    b.add_argument("--port", type=int, default=8502, help="Streamlit server port.")

    c = sub.add_parser("chapter-c", help="Synthetic dataset + RandomForest classifier")
    c.add_argument("--port", type=int, default=8503, help="Streamlit server port.")

    d = sub.add_parser("chapter-d", help="DLA on bond-percolation substrates")
    d.add_argument("--perc-size", type=int, default=45, help="Percolation grid side.")
    d.add_argument("--p-open", type=float, default=0.55, help="Bond occupation probability.")
    d.add_argument("--seed", type=int, default=42, help="Substrate RNG seed.")
    d.add_argument("--dla-runs", type=int, default=20, help="Number of ensemble runs.")
    d.add_argument(
        "--dla-max-frac",
        type=float,
        default=0.15,
        help="Stop cluster growth at this fraction of substrate nodes.",
    )
    d.add_argument("--port", type=int, default=8504, help="Streamlit server port.")

    e = sub.add_parser("chapter-e", help="Critical phenomena at the percolation transition")
    e.add_argument(
        "--mode",
        choices=("order_parameter", "cluster_geometry", "finite_size_collapse", "critical_dynamics"),
        default="order_parameter",
        help="Observable mode to open on (default: order_parameter).",
    )
    e.add_argument("--perc-size", type=int, default=28, help="L for dynamics modes.")
    e.add_argument("--n-samples", type=int, default=30, help="Samples per (L, p) point.")
    e.add_argument("--seed", type=int, default=42, help="RNG seed.")
    e.add_argument("--port", type=int, default=8505, help="Streamlit server port.")

    return p


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    args = _parser().parse_args(argv)

    if args.command == "chapter-a":
        env = {
            "FRACTAL_CHAPTER": "A",
            "FRACTAL_GRAPH": args.graph,
            "FRACTAL_DEPTH": str(args.depth),
            "FRACTAL_SEED": str(args.seed),
            "FRACTAL_PERC_SIZE": str(args.perc_size),
            "FRACTAL_P_OPEN": str(args.p_open),
            "FRACTAL_LEVY_ALPHA": str(args.levy_alpha),
            "FRACTAL_LEVY_P": str(args.levy_p),
            "FRACTAL_BIAS": args.bias,
            "FRACTAL_BIAS_STRENGTH": str(args.bias_strength),
        }
        return _run_streamlit(env, args.port)

    if args.command == "chapter-b":
        env = {
            "FRACTAL_CHAPTER": "B",
            "FRACTAL_MODE": args.mode,
        }
        return _run_streamlit(env, args.port)

    if args.command == "chapter-c":
        env = {"FRACTAL_CHAPTER": "C"}
        return _run_streamlit(env, args.port)

    if args.command == "chapter-d":
        env = {
            "FRACTAL_CHAPTER": "D",
            "FRACTAL_PERC_SIZE": str(args.perc_size),
            "FRACTAL_P_OPEN": str(args.p_open),
            "FRACTAL_SEED": str(args.seed),
            "FRACTAL_DLA_RUNS": str(args.dla_runs),
            "FRACTAL_DLA_MAX_FRAC": str(args.dla_max_frac),
        }
        return _run_streamlit(env, args.port)

    if args.command == "chapter-e":
        env = {
            "FRACTAL_CHAPTER": "E",
            "FRACTAL_E_MODE": args.mode,
            "FRACTAL_PERC_SIZE": str(args.perc_size),
            "FRACTAL_E_SAMPLES": str(args.n_samples),
            "FRACTAL_SEED": str(args.seed),
        }
        return _run_streamlit(env, args.port)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
