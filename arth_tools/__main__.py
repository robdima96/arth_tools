# -*- coding: utf-8 -*-
"""
Command-line tools for the OA image pipeline.

    python -m arth_tools prepare --config kl_grade --dicom-root /path/to/xrays
    python -m arth_tools train --config kl_grade
    python -m arth_tools infer --run-dir reporting/training/<run_id> --dicom-root /path/to/new
    python -m arth_tools hka --config hka --dicom-root /path/to/longleg
    python -m arth_tools eval --run-dir reporting/training/<run_id>
    python -m arth_tools ui
    python -m arth_tools hpo --config omeract_synovitis
    python -m arth_tools smoke
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(
        prog="arth_tools",
        description="OA image tools: prepare, export, split, train, eval, infer, hka, hpo, ui, smoke.",
    )
    ap.add_argument(
        "tool",
        nargs="?",
        choices=(
            "prepare",
            "export",
            "split",
            "train",
            "eval",
            "infer",
            "hka",
            "hpo",
            "ui",
            "smoke",
            "roi-prepare",
            "roi-train",
            "roi-eval",
        ),
        help="Which tool to run",
    )
    args, rest = ap.parse_known_args(argv)
    if args.tool is None:
        ap.print_help()
        print("\nExamples:")
        print("  python -m arth_tools prepare --config kl_grade --dicom-root /path/to/xrays")
        print("  python -m arth_tools train --config kl_grade")
        print("  python -m arth_tools hka --config hka --dicom-root /path/to/longleg")
        print("  python -m arth_tools infer --run-dir reporting/training/<run_id> --dicom-root /path/to/new")
        print("  python -m arth_tools eval --run-dir reporting/training/<run_id>")
        print("  python -m arth_tools ui")
        print("  python -m arth_tools smoke")
        return 0

    if args.tool == "prepare":
        from arth_tools.data.prepare import main as tool_main
    elif args.tool == "export":
        from arth_tools.data.export import main as tool_main
    elif args.tool == "split":
        from arth_tools.data.splits import main as tool_main
    elif args.tool == "train":
        from arth_tools.training.train import main as tool_main
    elif args.tool == "eval":
        from arth_tools.training.evaluate import main as tool_main
    elif args.tool == "infer":
        from arth_tools.inference.infer import main as tool_main
    elif args.tool == "hka":
        from arth_tools.hka.run import main as tool_main
    elif args.tool == "hpo":
        from arth_tools.training.hpo import main as tool_main
    elif args.tool == "ui":
        from arth_tools.ui.launch import main as tool_main
    elif args.tool == "roi-prepare":
        from arth_tools.roi.prepare import main as tool_main
    elif args.tool == "roi-train":
        from arth_tools.roi.train import main as tool_main
    elif args.tool == "roi-eval":
        from arth_tools.roi.train import main_eval as tool_main
    else:
        from arth_tools.smoke import main as tool_main

    if args.tool == "smoke":
        return int(tool_main())
    return int(tool_main(rest))


if __name__ == "__main__":
    raise SystemExit(main())
