"""
Render pipeline:
analysis.json + Jinja2 Markdown template  -> report.md -> report.pdf (WeasyPrint)

Expected run directory layout (recommended):
outputs/<run_id>/
  analysis.json
  figures/
    *.png
  report.md          (generated)
  report.pdf         (generated)

Dependencies:
  pip install jinja2 markdown weasyprint
Optional (nice-to-have):
  pip install pygments  (only if you enable code highlighting CSS yourself)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader, StrictUndefined

import markdown as md
from weasyprint import HTML, CSS


DEFAULT_CSS = """
@page { size: A4; margin: 18mm 16mm; }

html, body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans",
               "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", Arial, sans-serif;
  font-size: 11pt;
  line-height: 1.5;
  color: #111;
}

h1, h2, h3 {
  line-height: 1.25;
  margin: 0.9em 0 0.35em;
}
h1 { font-size: 20pt; }
h2 { font-size: 15pt; border-bottom: 1px solid #ddd; padding-bottom: 0.2em; }
h3 { font-size: 12.5pt; }

p { margin: 0.4em 0; }
ul { margin: 0.35em 0 0.6em 1.2em; }
li { margin: 0.2em 0; }

code, pre {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  font-size: 10pt;
}
pre {
  padding: 10px 12px;
  background: #f6f8fa;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  white-space: pre-wrap;
}

blockquote {
  margin: 0.7em 0;
  padding: 0.6em 0.8em;
  background: #fafafa;
  border-left: 4px solid #ddd;
}

table {
  border-collapse: collapse;
  width: 100%;
  table-layout: fixed;   /* 关键：固定布局，让表格强制在页面宽度内分配列宽 */
  margin: 0.6em 0 0.8em;
  font-size: 10.5pt;
}
th, td {
  border: 1px solid #e5e7eb;
  padding: 6px 8px;
  vertical-align: top;
  overflow-wrap: anywhere;  /* 允许在任意位置断行，避免长表头/长内容把列撑爆 */
  word-break: break-word;
  white-space: normal;
}
th { background: #f3f4f6; }

img {
  max-width: 100%;
  height: auto;
  margin: 0.35em 0 0.5em;
}

hr { border: 0; border-top: 1px solid #ddd; margin: 1em 0; }
"""


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def render_markdown_from_template(
    analysis_data: Dict[str, Any],
    template_path: Path,
) -> str:
    """
    Renders a Markdown string from a Jinja2 template and analysis dict.
    Uses StrictUndefined to catch missing fields early (helps debugging).
    """
    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        autoescape=False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tpl = env.get_template(template_path.name)
    return tpl.render(**analysis_data)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def markdown_to_html(md_text: str, title: str = "Report") -> str:
    """
    Convert Markdown text to a standalone HTML document.
    """
    body_html = md.markdown(
        md_text,
        extensions=[
            "tables",
            "fenced_code",
            "toc",
            "sane_lists",
        ],
        output_format="html5",
    )

    html_doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{title}</title>
</head>
<body>
{body_html}
</body>
</html>"""
    return html_doc


def render_pdf_with_weasyprint(
    md_path: Path,
    pdf_path: Path,
    base_url: Optional[Path] = None,
    css_text: str = DEFAULT_CSS,
    extra_css_path: Optional[Path] = None,
) -> None:
    """
    Render a PDF from a Markdown file.
    base_url matters for loading relative image paths like ![](figures/xxx.png).
    """
    md_text = md_path.read_text(encoding="utf-8")
    html_doc = markdown_to_html(md_text, title=md_path.stem)

    base_url = base_url or md_path.parent
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    stylesheets = [CSS(string=css_text)]
    if extra_css_path is not None and extra_css_path.exists():
        stylesheets.append(CSS(filename=str(extra_css_path)))

    HTML(string=html_doc, base_url=str(base_url)).write_pdf(
        str(pdf_path),
        stylesheets=stylesheets,
    )


def render_all(
    run_dir: Path,
    template_path: Path,
    out_md_name: str = "report.md",
    out_pdf_name: str = "report.pdf",
    analysis_name: str = "analysis.json",
    extra_css_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    One-shot:
      run_dir/analysis.json -> run_dir/report.md -> run_dir/report.pdf
    Returns output paths.
    """
    run_dir = run_dir.resolve()
    analysis_path = run_dir / analysis_name
    if not analysis_path.exists():
        raise FileNotFoundError(f"analysis.json not found: {analysis_path}")

    analysis_data = load_json(analysis_path)

    md_text = render_markdown_from_template(analysis_data, template_path)
    md_path = run_dir / out_md_name
    write_text(md_path, md_text)

    pdf_path = run_dir / out_pdf_name
    render_pdf_with_weasyprint(
        md_path=md_path,
        pdf_path=pdf_path,
        base_url=run_dir,  # important: so ![](figures/xxx.png) works
        extra_css_path=extra_css_path,
    )

    return {"analysis": analysis_path, "md": md_path, "pdf": pdf_path}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render report.md and report.pdf from analysis.json")
    p.add_argument("--run_dir", type=str, required=True, help="Directory containing analysis.json and figures/")
    p.add_argument("--template", type=str, required=True, help="Path to Jinja2 Markdown template, e.g. reporting/template.md")
    p.add_argument("--out_md", type=str, default="report.md", help="Markdown output filename (within run_dir)")
    p.add_argument("--out_pdf", type=str, default="report.pdf", help="PDF output filename (within run_dir)")
    p.add_argument("--analysis_name", type=str, default="analysis.json", help="analysis json filename (within run_dir)")
    p.add_argument("--extra_css", type=str, default="", help="Optional extra CSS file path")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    run_dir = Path(args.run_dir)
    template_path = Path(args.template)

    extra_css_path = Path(args.extra_css) if args.extra_css else None

    outputs = render_all(
        run_dir=run_dir,
        template_path=template_path,
        out_md_name=args.out_md,
        out_pdf_name=args.out_pdf,
        analysis_name=args.analysis_name,
        extra_css_path=extra_css_path,
    )

    print("OK")
    print(f"- analysis: {outputs['analysis']}")
    print(f"- md:      {outputs['md']}")
    print(f"- pdf:     {outputs['pdf']}")


if __name__ == "__main__":
    main()
