from __future__ import annotations

import argparse
from pathlib import Path

from tools import RunConfig, run_analysis
from reporting.render import render_all


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CSV -> analysis.json + report.md + report.pdf")
    p.add_argument("--csv", required=True, help="Path to input CSV")
    p.add_argument("--out", default="outputs", help="Output root dir, default: outputs/")
    p.add_argument("--template", default="reporting/template.md", help="Jinja2 markdown template path")
    p.add_argument("--sample_rows", type=int, default=200_000, help="Max rows loaded for analysis")
    p.add_argument("--target", default="", help="目标列名（可选）。指定后会执行最小建模闭环")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = RunConfig(sample_rows=args.sample_rows)
    analysis = run_analysis(args.csv, output_root=args.out, cfg=cfg, target=(args.target or None))
    run_dir = Path(args.out) / analysis["run_id"]
    render_all(
        run_dir=run_dir,
        template_path=Path(args.template),
    )

    print("Done.")
    print(f"- Run dir: {run_dir}")
    print(f"- PDF: {run_dir / 'report.pdf'}")


if __name__ == "__main__":
    main()
