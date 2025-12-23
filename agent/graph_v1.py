from __future__ import annotations

import argparse
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, TypedDict

from langgraph.graph import StateGraph, START, END

import tools as t  


class GraphState(TypedDict, total=False):
    # inputs
    csv_path: str
    output_root: str
    template_path: str
    sample_rows: int

    # runtime paths
    run_id: str
    out_dir: str
    fig_dir: str

    # data objects
    cfg: t.RunConfig
    df_raw: Any
    df_clean: Any
    df_fe: Any
    meta: Dict[str, Any]
    roles: Dict[str, str]
    profile_raw: Dict[str, Any]
    cleaning_log: Dict[str, Any]
    fe_log: Dict[str, Any]
    plots: list[Dict[str, Any]]
    tests: list[Dict[str, Any]]
    insights: list[str]

    # final artifacts
    analysis: Dict[str, Any]
    analysis_path: str
    report_md_path: Optional[str]
    report_pdf_path: Optional[str]


# -------------------------
# Nodes
# -------------------------

def node_init_run(state: GraphState) -> GraphState:
    cfg = t.RunConfig(sample_rows=state.get("sample_rows", 200_000))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

    out_dir = os.path.join(state["output_root"], run_id)
    fig_dir = os.path.join(out_dir, "figures")
    t.ensure_dir(fig_dir)

    return {
        "cfg": cfg,
        "run_id": run_id,
        "out_dir": out_dir,
        "fig_dir": fig_dir,
    }


def node_load_and_profile(state: GraphState) -> GraphState:
    df_raw, meta = t.load_csv(state["csv_path"], state["cfg"])
    roles = t.infer_column_roles(df_raw)
    profile_raw = t.profile_dataset(df_raw, roles, state["cfg"])
    return {
        "df_raw": df_raw,
        "meta": meta,
        "roles": roles,
        "profile_raw": profile_raw,
    }


def node_clean(state: GraphState) -> GraphState:
    df_clean, cleaning_log = t.clean_dataset(state["df_raw"], state["roles"], state["cfg"])
    return {"df_clean": df_clean, "cleaning_log": cleaning_log}


def node_feature_engineering(state: GraphState) -> GraphState:
    df_fe, fe_log = t.feature_engineering(state["df_clean"], state["roles"])
    return {"df_fe": df_fe, "fe_log": fe_log}


def node_plots(state: GraphState) -> GraphState:
    plots = t.make_plots(
        df_raw=state["df_raw"],
        df_clean=state["df_clean"],
        roles=state["roles"],
        out_dir=state["fig_dir"],
        cfg=state["cfg"],
    )
    return {"plots": plots}


def node_stat_tests(state: GraphState) -> GraphState:
    tests = t.run_stat_tests(state["df_clean"], state["roles"], state["cfg"])
    return {"tests": tests}


def node_insights(state: GraphState) -> GraphState:
    insights = t.generate_rule_based_insights(
        profile=state["profile_raw"],
        cleaning_log=state["cleaning_log"],
        tests=state["tests"],
        cfg=state["cfg"],
    )
    return {"insights": insights}


def node_assemble_and_save(state: GraphState) -> GraphState:
    # 构建 analysis dict（对齐现有 run_analysis 输出结构）:contentReference[oaicite:2]{index=2}
    analysis = {
        "run_id": state["run_id"],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "filename": state["meta"]["filename"],
            "n_rows": state["profile_raw"]["n_rows"],
            "n_cols": state["profile_raw"]["n_cols"],
            "memory_mb": state["profile_raw"]["memory_mb"],
            "loaded_rows": state["meta"]["loaded_rows"],
        },
        "columns": state["profile_raw"]["columns"],
        "cleaning": {**state["cleaning_log"], **state.get("fe_log", {"added_features": []})},
        "plots": [
            {**p, "path": os.path.relpath(p["path"], state["out_dir"]).replace("\\", "/")}
            for p in state.get("plots", [])
        ],
        "tests": state.get("tests", []),
        "insights": state.get("insights", []),
    }

    analysis_path = os.path.join(state["out_dir"], "analysis.json")
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    return {"analysis": analysis, "analysis_path": analysis_path}


def node_render_report(state):
    """
    生成 report.md 和 report.pdf (WeasyPrint)。
    兼容两种放置方式：
    1) reporting/render.py  -> from reporting.render import render_all  
    2) 项目根目录 render.py -> import render
    """
    render_all = None

    # reporting/render.py
    try:
        from reporting.render import render_all as _render_all
        render_all = _render_all
    except Exception:
        pass

    from pathlib import Path
    run_dir = Path(state["out_dir"])
    template_path = Path(state["template_path"])

    outputs = render_all(run_dir=run_dir, template_path=template_path)

    # render_all 返回 Dict[str, Path]：{"analysis":..., "md":..., "pdf":...} :contentReference[oaicite:4]{index=4}
    return {
        "report_md_path": str(outputs.get("md")),
        "report_pdf_path": str(outputs.get("pdf")),
    }

# -------------------------
# Build graph
# -------------------------

def build_graph():
    g = StateGraph(GraphState)
    g.add_node("init_run", node_init_run)
    g.add_node("load_and_profile", node_load_and_profile)
    g.add_node("clean", node_clean)
    g.add_node("feature_engineering", node_feature_engineering)
    g.add_node("plots", node_plots)
    g.add_node("stat_tests", node_stat_tests)
    g.add_node("insights", node_insights)
    g.add_node("assemble_and_save", node_assemble_and_save)
    g.add_node("render_report", node_render_report)

    g.add_edge(START, "init_run")
    g.add_edge("init_run", "load_and_profile")
    g.add_edge("load_and_profile", "clean")
    g.add_edge("clean", "feature_engineering")
    g.add_edge("feature_engineering", "plots")
    g.add_edge("plots", "stat_tests")
    g.add_edge("stat_tests", "insights")
    g.add_edge("insights", "assemble_and_save")
    g.add_edge("assemble_and_save", "render_report")
    g.add_edge("render_report", END)

    return g.compile()


# -------------------------
# CLI
# -------------------------

def main():
    p = argparse.ArgumentParser(description="LangGraph v1 pipeline (no LLM): CSV -> analysis.json (+ report.md/pdf)")
    p.add_argument("--csv", required=True, help="Path to input CSV")
    p.add_argument("--out", default="outputs", help="Output root dir")
    p.add_argument("--template", default="reporting/template.md", help="Jinja2 markdown template path")
    p.add_argument("--sample_rows", type=int, default=200_000, help="Max rows to load")
    args = p.parse_args()

    graph = build_graph()

    init_state: GraphState = {
        "csv_path": args.csv,
        "output_root": args.out,
        "template_path": args.template,
        "sample_rows": args.sample_rows,
    }

    final_state = graph.invoke(init_state)

    print("Done.")
    print("Run dir:", final_state.get("out_dir"))
    print("analysis.json:", final_state.get("analysis_path"))
    print("report.md:", final_state.get("report_md_path"))
    print("report.pdf:", final_state.get("report_pdf_path"))


if __name__ == "__main__":
    main()
