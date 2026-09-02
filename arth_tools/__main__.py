# -*- coding: utf-8 -*-
"""
Command-line tools for the OA image pipeline.

    python -m arth_tools export
    python -m arth_tools split --manifest E:\\ArthAgent\\manifests\\manifest_master.csv
    python -m arth_tools train
    python -m arth_tools eval --run-dir reporting/training/<run_id>
    python -m arth_tools hpo
    python -m arth_tools smoke
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        prog="arth_tools",
        description="OA image tools: export, split, train, eval, hpo, smoke.",
    )
    ap.add_argument(
        "tool",
        nargs="?",
        choices=("export", "split", "train", "eval", "hpo", "smoke"),
        help="Which tool to run",
    )
    args, rest = ap.parse_known_args(argv)
    if args.tool is None:
        ap.print_help()
        print("\nExamples:")
        print("  python -m arth_tools train")
        print("  python -m arth_tools eval --run-dir reporting/training/<run_id>")
        print("  python -m arth_tools smoke")
        return 0

    if args.tool == "export":
        from arth_tools.data.export import main as tool_main
    elif args.tool == "split":
        from arth_tools.data.splits import main as tool_main
    elif args.tool == "train":
        from arth_tools.training.train import main as tool_main
    elif args.tool == "eval":
        from arth_tools.training.evaluate import main as tool_main
    elif args.tool == "hpo":
        from arth_tools.training.hpo import main as tool_main
    else:
        from arth_tools.smoke import main as tool_main

    return int(tool_main(rest) if args.tool != "smoke" else tool_main())


if __name__ == "__main__":
    raise SystemExit(main())
