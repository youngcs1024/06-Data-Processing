from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, List, TypedDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tools as t  # noqa: E402

from langgraph.graph import StateGraph, START, END  # noqa: E402

# Qwen API key
HARDCODED_DASHSCOPE_API_KEY = ""  


class GraphState(TypedDict, total=False):
    # inputs
    csv_path: str
    output_root: str
    template_path: str

    # config
    cfg: t.RunConfig

    # run dirs
    run_id: str
    out_dir: str
    fig_dir: str

    # data and intermediate results
    df_raw: Any
    df_clean: Any
    df_fe: Any
    meta: Dict[str, Any]
    roles: Dict[str, str]
    profile_raw: Dict[str, Any]
    cleaning_log: Dict[str, Any]
    fe_log: Dict[str, Any]
    plots: List[Dict[str, Any]]
    tests: List[Dict[str, Any]]
    insights: List[str]

    # llm
    use_llm: bool
    qwen_api_key: str
    qwen_model: str
    llm_error: str

    # outputs
    analysis: Dict[str, Any]
    analysis_path: str
    report_md_path: str
    report_pdf_path: str


def resolve_dashscope_api_key(cli_key: str = "") -> str:
    return (
        (cli_key or "").strip()
        or (HARDCODED_DASHSCOPE_API_KEY or "").strip()
        or os.getenv("DASHSCOPE_API_KEY", "").strip()
    )


def _extract_text_from_dashscope_response(resp: Any) -> str:

    if resp is None:
        return ""

    out = getattr(resp, "output", None)
    if out is None and isinstance(resp, dict):
        out = resp.get("output")

    # 1) output.text 
    if hasattr(out, "text") and getattr(out, "text"):
        return str(getattr(out, "text"))

    if isinstance(out, dict):
        # 2) output["text"]
        if out.get("text"):
            return str(out["text"])
        # 3) OpenAI-like: choices[0].message.content
        try:
            choices = out.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                content = msg.get("content")
                if content:
                    return str(content)
        except Exception:
            pass

    # 4) resp.output.choices[0].message.content
    try:
        choices = getattr(out, "choices", None)
        if choices:
            first = choices[0]
            msg = getattr(first, "message", None)
            if msg is not None:
                content = getattr(msg, "content", None)
                if content:
                    return str(content)
    except Exception:
        pass

    try:
        return str(resp)
    except Exception:
        return ""


def qwen_chat_once(
    messages: List[Dict[str, str]],
    api_key: str,
    model: str = "qwen-turbo",
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> str:
    """
    调用通义千问(DashScope)做一次对话生成(使用 dashscope Python SDK)。
    """
    import dashscope  # type: ignore
    from dashscope import Generation  # type: ignore

    dashscope.api_key = api_key

    resp = Generation.call(
        model=model,
        messages=messages,
        result_format="message",
        temperature=temperature,
        max_tokens=max_tokens,
    )

    status = getattr(resp, "status_code", None)
    if status not in (HTTPStatus.OK, 200, "200"):
        msg = getattr(resp, "message", None)
        raise RuntimeError(f"DashScope call failed: status_code={status}, message={msg}")

    return _extract_text_from_dashscope_response(resp).strip()


def _parse_insights_text(text: str, max_items: int = 6) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    # try JSON first
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            items = [str(x).strip() for x in obj if str(x).strip()]
            return items[:max_items]
        if isinstance(obj, dict):
            maybe = obj.get("insights") or obj.get("bullets") or obj.get("points")
            if isinstance(maybe, list):
                items = [str(x).strip() for x in maybe if str(x).strip()]
                return items[:max_items]
    except Exception:
        pass

    # fallback: bullet lines
    lines = [ln.strip() for ln in text.splitlines()]
    bullets: List[str] = []
    for ln in lines:
        if not ln:
            continue
        ln = ln.lstrip("-*•").strip()
        ln = ln.lstrip("0123456789").lstrip(".、))").strip()
        if ln:
            bullets.append(ln)

    # de-dup preserve order
    seen = set()
    out: List[str] = []
    for b in bullets:
        if b not in seen:
            out.append(b)
            seen.add(b)
        if len(out) >= max_items:
            break
    return out


# -------------------------
# LangGraph nodes
# -------------------------
def node_init_run(state: GraphState) -> GraphState:
    cfg = state.get("cfg") or t.RunConfig()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    out_dir = os.path.join(state["output_root"], run_id)
    fig_dir = os.path.join(out_dir, "figures")
    t.ensure_dir(fig_dir)

    api_key = resolve_dashscope_api_key(state.get("qwen_api_key", ""))
    use_llm = bool(state.get("use_llm", True) and api_key)

    return {
        "cfg": cfg,
        "run_id": run_id,
        "out_dir": out_dir,
        "fig_dir": fig_dir,
        "qwen_api_key": api_key,
        "use_llm": use_llm,
    }


def node_load_and_profile(state: GraphState) -> GraphState:
    df_raw, meta = t.load_csv(state["csv_path"], state["cfg"])
    roles = t.infer_column_roles(df_raw)
    profile_raw = t.profile_dataset(df_raw, roles, state["cfg"])
    return {"df_raw": df_raw, "meta": meta, "roles": roles, "profile_raw": profile_raw}


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


def node_insights_rule_based(state: GraphState) -> GraphState:
    insights = t.generate_rule_based_insights(
        profile=state["profile_raw"],
        cleaning_log=state["cleaning_log"],
        tests=state.get("tests", []),
        cfg=state["cfg"],
    )
    return {"insights": insights}


def node_llm_insights(state: GraphState) -> GraphState:
    """
    用 LLM 把“规则洞见”润色 + 补充(严格基于已有统计结果，禁止编造)。
    """
    try:
        cfg = state["cfg"]

        llm_payload = {
            "dataset": {
                "filename": state["meta"]["filename"],
                "n_rows": state["profile_raw"]["n_rows"],
                "n_cols": state["profile_raw"]["n_cols"],
                "loaded_rows": state["meta"].get("loaded_rows"),
            },
            "top_missing": state["profile_raw"].get("top_missing", [])[: getattr(cfg, "topk_cols", 12)],
            "tests": state.get("tests", []),
            "rule_based_insights": state.get("insights", []),
            "cleaning_summary": {
                "dropped_cols": state.get("cleaning_log", {}).get("dropped_cols", []),
                "imputation": state.get("cleaning_log", {}).get("imputation", {}),
                "type_conversions": state.get("cleaning_log", {}).get("type_conversions", []),
                "added_features": state.get("fe_log", {}).get("added_features", []),
            },
        }

        system = (
            "你是严谨的数据分析师。你只能根据用户提供的 JSON 信息写结论，"
            "不能凭空编造数字/显著性/因果解释。"
            "如果信息不足，请明确说“无法从当前数据推断”。"
            "输出必须是 JSON 数组，数组元素是中文字符串，每条不超过 60 字，最多 6 条。"
        )

        user = (
            "请根据下面的分析摘要，输出“可复述的洞见要点”(偏业务、可操作)，并可对规则洞见做润色。"
            "注意：不得编造任何未出现在摘要里的数值结论。\n\n"
            + json.dumps(llm_payload, ensure_ascii=False, indent=2)
        )

        text = qwen_chat_once(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            api_key=state["qwen_api_key"],
            model=state.get("qwen_model", "qwen-turbo"),
            temperature=0.2,
            max_tokens=800,
        )
        llm_insights = _parse_insights_text(text, max_items=6)

        merged: List[str] = []
        seen = set()
        for it in (state.get("insights", []) + llm_insights):
            it = str(it).strip()
            if not it or it in seen:
                continue
            merged.append(it)
            seen.add(it)

        return {"insights": merged, "llm_error": ""}
    except Exception as e:
        return {"llm_error": str(e)}


def node_assemble_and_save(state: GraphState) -> GraphState:
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
        "llm": {
            "used": bool(state.get("use_llm")),
            "model": state.get("qwen_model", ""),
            "error": state.get("llm_error", ""),
        },
    }

    analysis_path = os.path.join(state["out_dir"], "analysis.json")
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    return {"analysis": analysis, "analysis_path": analysis_path}


def node_render_report(state: GraphState) -> GraphState:
    """
    生成 report.md 和 report.pdf (WeasyPrint)。
    优先使用 reporting/render.py 的 render_all;否则尝试根目录 render.py。
    """
    render_all = None

    try:
        from reporting.render import render_all as _render_all  # type: ignore
        render_all = _render_all
    except Exception:
        pass

    if render_all is None:
        try:
            import render  # type: ignore
            render_all = render.render_all
        except Exception as e:
            raise RuntimeError(
                "Cannot import render_all. Please ensure you have reporting/render.py or render.py"
            ) from e

    run_dir = Path(state["out_dir"])
    template_path = Path(state["template_path"])

    outputs = render_all(run_dir=run_dir, template_path=template_path)
    return {"report_md_path": str(outputs.get("md")), "report_pdf_path": str(outputs.get("pdf"))}


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
    g.add_node("insights_rule", node_insights_rule_based)
    g.add_node("llm_insights", node_llm_insights)
    g.add_node("assemble_and_save", node_assemble_and_save)
    g.add_node("render_report", node_render_report)

    g.add_edge(START, "init_run")
    g.add_edge("init_run", "load_and_profile")
    g.add_edge("load_and_profile", "clean")
    g.add_edge("clean", "feature_engineering")
    g.add_edge("feature_engineering", "plots")
    g.add_edge("plots", "stat_tests")
    g.add_edge("stat_tests", "insights_rule")

    def _route_after_insights(state: GraphState) -> str:
        return "llm_insights" if state.get("use_llm") else "assemble_and_save"

    g.add_conditional_edges(
        "insights_rule",
        _route_after_insights,
        {
            "llm_insights": "llm_insights",
            "assemble_and_save": "assemble_and_save",
        },
    )

    g.add_edge("llm_insights", "assemble_and_save")
    g.add_edge("assemble_and_save", "render_report")
    g.add_edge("render_report", END)

    return g.compile()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LangGraph v2: Auto EDA + Report (optional Qwen insights)")
    p.add_argument("--csv", dest="csv_path", type=str, required=True, help="Input CSV path")
    p.add_argument("--out", dest="output_root", type=str, default="outputs", help="Output root dir")
    p.add_argument("--template", dest="template_path", type=str, required=True, help="Jinja2 Markdown template path")

    # RunConfig overrides (optional)
    p.add_argument("--sample_rows", type=int, default=200_000)
    p.add_argument("--topk_cols", type=int, default=12)
    p.add_argument("--corr_max_features", type=int, default=12)

    # Qwen
    p.add_argument("--qwen_api_key", type=str, default="", help="DashScope API Key (optional)")
    p.add_argument("--qwen_model", type=str, default="qwen-turbo", help="e.g. qwen-turbo / qwen-plus / qwen-max")
    p.add_argument("--no_llm", action="store_true", help="Disable LLM refinement (rule-based only)")
    return p


def main() -> None:
    args = build_argparser().parse_args()

    cfg = t.RunConfig(
        sample_rows=args.sample_rows,
        topk_cols=args.topk_cols,
        corr_max_features=args.corr_max_features,
    )

    app = build_graph()
    final_state = app.invoke(
        {
            "csv_path": args.csv_path,
            "output_root": args.output_root,
            "template_path": args.template_path,
            "cfg": cfg,
            "qwen_api_key": args.qwen_api_key,
            "qwen_model": args.qwen_model,
            "use_llm": (not args.no_llm),
        }
    )

    print("Done.")
    print(f"Run dir: {final_state.get('out_dir')}")
    print(f"analysis.json: {final_state.get('analysis_path')}")
    print(f"report.md: {final_state.get('report_md_path')}")
    print(f"report.pdf: {final_state.get('report_pdf_path')}")
    if final_state.get("llm_error"):
        print(f"[LLM skipped/failed] {final_state['llm_error']}")


if __name__ == "__main__":
    main()
