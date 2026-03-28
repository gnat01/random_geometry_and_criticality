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
        description="Fractal random walks — Chapters A (geometry), B (observables), C (ML).",
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
        help="Grid side for bond percolation (ignored for carpet/Vicsek in env defaults).",
    )
    a.add_argument(
        "--p-open",
        type=float,
        default=0.55,
        help="Bond occupation probability for percolation.",
    )
    a.add_argument("--seed", type=int, default=42, help="RNG seed for percolation instance.")
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

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
