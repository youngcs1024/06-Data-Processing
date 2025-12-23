from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Literal

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tools as t  # noqa: E402
from reporting import render as r  # noqa: E402

from langgraph.graph import StateGraph, START, END  # noqa: E402


# -------------------------
# Qwen (DashScope OpenAI-compatible) client (HTTP)
# -------------------------

DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_ENDPOINT = "/chat/completions"

# Qwen API key
HARDCODED_DASHSCOPE_API_KEY = ""  

def qwen_chat_once(
    messages: List[Dict[str, str]],
    model: str,
    api_key: str,
    base_url: str = DEFAULT_QWEN_BASE_URL,
    temperature: float = 0.2,
    max_tokens: int = 900,
    timeout_s: int = 120,
) -> str:
    """
    Call DashScope OpenAI-compatible Chat Completions endpoint via HTTP.
    Returns assistant content string.
    """
    url = base_url.rstrip("/") + DEFAULT_QWEN_ENDPOINT
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(data, ensure_ascii=False, indent=2)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort extraction of a JSON object from arbitrary LLM output.
    """
    if not text:
        return None

    # direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # fenced code block ```json ... ```
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # first {...} block (greedy but bounded)
    m = re.search(r"(\{[\s\S]*\})", text)
    if m:
        chunk = m.group(1).strip()
        # try to cut to last "}"
        last = chunk.rfind("}")
        if last != -1:
            chunk = chunk[: last + 1]
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


# -------------------------
# Extra analyses (pure pandas; no need to modify tools.py)
# -------------------------

def _find_col(analysis: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    cols = [c.get("name") for c in analysis.get("columns", []) if isinstance(c, dict)]
    canon = {re.sub(r"[^a-z0-9]+", "", str(c).lower()): c for c in cols if c}
    for cand in candidates:
        k = re.sub(r"[^a-z0-9]+", "", cand.lower())
        if k in canon:
            return canon[k]
    return None


def extra_analysis_tables(
    csv_path: str,
    analysis: Dict[str, Any],
    sample_rows: int,
) -> List[Dict[str, Any]]:
    """
    Generate a few "business-like" tables to make report richer:
    - Payment Type summary
    - Company (top) summary
    - Time-of-day / DOW summary if timestamps exist
    """
    df = t.pd.read_csv(csv_path, nrows=sample_rows, low_memory=False)

    # try parse timestamps if exist
    ts_start = _find_col(analysis, ["Trip Start Timestamp", "start_time", "timestamp", "datetime"])
    if ts_start and ts_start in df.columns:
        df[ts_start] = t.pd.to_datetime(df[ts_start], errors="coerce")

    pay_col = _find_col(analysis, ["Payment Type", "payment_type"])
    comp_col = _find_col(analysis, ["Company", "company"])
    fare_col = _find_col(analysis, ["Fare", "fare"])
    tip_col = _find_col(analysis, ["Tips", "tip", "tips"])
    total_col = _find_col(analysis, ["Trip Total", "total"])
    miles_col = _find_col(analysis, ["Trip Miles", "miles"])
    sec_col = _find_col(analysis, ["Trip Seconds", "seconds", "duration"])

    tables: List[Dict[str, Any]] = []

    def _safe_num(c: Optional[str]) -> Optional[str]:
        return c if c and c in df.columns else None

    fare_col = _safe_num(fare_col)
    tip_col = _safe_num(tip_col)
    total_col = _safe_num(total_col)
    miles_col = _safe_num(miles_col)
    sec_col = _safe_num(sec_col)

    metrics = [c for c in [fare_col, tip_col, total_col, miles_col, sec_col] if c]

    # 1) Payment Type summary  —— 拆成两张表（金额相关 / 行程相关），避免 PDF 过宽溢出
    if pay_col and pay_col in df.columns and metrics:
        g = df.groupby(pay_col, dropna=False)

        # 先按 count 取 Top 12（两张表保持同一批支付方式行）
        base = t.pd.DataFrame({"count": g.size()}).sort_values("count", ascending=False).head(12)

        # A2 拆表：钱相关 / 行程相关
        money_metrics = [c for c in [fare_col, tip_col, total_col] if c]
        trip_metrics = [c for c in [miles_col, sec_col] if c]

        if money_metrics:
            out_money = base.copy()
            for m in money_metrics:
                out_money[f"{m}_mean"] = g[m].mean(numeric_only=True)
                out_money[f"{m}_median"] = g[m].median(numeric_only=True)

            # 小费率：用均值/均值（和你原来的逻辑一致）
            if tip_col and total_col:
                out_money["tip_rate_mean"] = (
                    g[tip_col].mean(numeric_only=True)
                    / (g[total_col].mean(numeric_only=True) + 1e-9)
                )

            out_money = out_money.reset_index()

            tables.append({
                "title": f"分组汇总：{pay_col}（金额相关 Top 12）",
                "columns": list(out_money.columns),
                "rows": out_money.fillna("").values.tolist(),
                "note": "用于补充“不同支付方式在费用/小费/总额上的差异”。",
            })

        if trip_metrics:
            out_trip = base.copy()
            for m in trip_metrics:
                out_trip[f"{m}_mean"] = g[m].mean(numeric_only=True)
                out_trip[f"{m}_median"] = g[m].median(numeric_only=True)

            out_trip = out_trip.reset_index()

            tables.append({
                "title": f"分组汇总：{pay_col}（行程相关 Top 12）",
                "columns": list(out_trip.columns),
                "rows": out_trip.fillna("").values.tolist(),
                "note": "用于补充“不同支付方式在里程/时长上的差异”。",
            })

    # 2) Company summary (top companies)
    if comp_col and comp_col in df.columns and metrics:
        top_companies = df[comp_col].astype("object").fillna("Unknown").value_counts().head(10).index.tolist()
        d2 = df[df[comp_col].astype("object").fillna("Unknown").isin(top_companies)]
        g = d2.groupby(comp_col, dropna=False)
        out = t.pd.DataFrame({"count": g.size()})
        for m in metrics:
            out[f"{m}_mean"] = g[m].mean(numeric_only=True)
        out = out.sort_values("count", ascending=False).reset_index()
        tables.append({
            "title": f"公司汇总：{comp_col}（Top 10 by count）",
            "columns": list(out.columns),
            "rows": out.fillna("").values.tolist(),
            "note": "用于补充“头部公司在费用/里程/小费上的差异”。",
        })

    # 3) Time patterns
    if ts_start and ts_start in df.columns and metrics:
        dt = df[ts_start]
        df["_hour"] = dt.dt.hour
        df["_dow"] = dt.dt.dayofweek
        g1 = df.groupby("_hour")
        out1 = t.pd.DataFrame({"count": g1.size()})
        for m in metrics:
            out1[f"{m}_mean"] = g1[m].mean(numeric_only=True)
        out1 = out1.reset_index().sort_values("_hour")
        tables.append({
            "title": f"时间模式：按小时（{ts_start}）",
            "columns": list(out1.columns),
            "rows": out1.fillna("").values.tolist(),
            "note": "用于补充“高峰时段/时段费用差异”。",
        })

        g2 = df.groupby("_dow")
        out2 = t.pd.DataFrame({"count": g2.size()})
        for m in metrics:
            out2[f"{m}_mean"] = g2[m].mean(numeric_only=True)
        out2 = out2.reset_index().sort_values("_dow")
        tables.append({
            "title": f"时间模式：按星期（0=周一…6=周日）（{ts_start}）",
            "columns": list(out2.columns),
            "rows": out2.fillna("").values.tolist(),
            "note": "用于补充“工作日 vs 周末差异”。",
        })

    return tables


def table_to_markdown(title: str, columns: List[str], rows: List[List[Any]], max_rows: int = 12) -> str:
    cols = [str(c) for c in columns]
    shown = rows[:max_rows]
    # stringify + lightweight rounding
    def fmt(x: Any) -> str:
        if isinstance(x, float):
            if abs(x) >= 1000:
                return f"{x:.2f}"
            return f"{x:.4f}"
        return str(x)

    lines = []
    lines.append(f"### {title}")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for rrow in shown:
        lines.append("| " + " | ".join(fmt(v) for v in rrow) + " |")
    if len(rows) > max_rows:
        lines.append(f"\n> 仅展示前 {max_rows} 行，完整可在代码里导出或提高展示上限。")
    return "\n".join(lines)


# -------------------------
# Report patching: inject LLM section even if template has no placeholder
# -------------------------

def inject_llm_and_tables_into_md(
    md_text: str,
    llm_md: str,
    tables: List[Dict[str, Any]],
) -> str:
    if not llm_md and not tables:
        return md_text

    # rename rule-based section title a bit (optional)
    # "## 5. 规则洞见（无LLM）" -> "## 5. 规则洞见（规则引擎）"
    md_text = re.sub(r"^##\s*5\.\s*规则洞见（无LLM）\s*$", "## 5. 规则洞见（规则引擎）", md_text, flags=re.M)

    # avoid duplicate injection
    if re.search(r"^##\s*6\.\s*LLM\s*洞见", md_text, flags=re.M):
        return md_text

    parts = []
    parts.append("\n\n## 6. LLM 洞见\n")
    if llm_md:
        parts.append(llm_md.strip() + "\n")
    if tables:
        parts.append("\n## 7. 额外分析表（Agent 自动补充）\n")
        for tb in tables:
            parts.append(table_to_markdown(tb.get("title", "Table"), tb.get("columns", []), tb.get("rows", [])))
            note = tb.get("note", "")
            if note:
                parts.append(f"\n> {note}\n")
            parts.append("\n")
    return md_text.rstrip() + "\n" + "".join(parts)


# -------------------------
# LangGraph state + nodes
# -------------------------

class AgentState(TypedDict, total=False):
    csv: str
    out: str
    template: str
    sample_rows: int
    # LLM
    use_llm: bool
    qwen_api_key: str
    qwen_model: str
    qwen_base_url: str
    # loop
    iteration: int
    max_iter: int
    # outputs
    analysis: Dict[str, Any]
    run_dir: str
    llm_plan: Dict[str, Any]
    llm_insights_md: str
    extra_tables: List[Dict[str, Any]]
    report_md: str
    report_pdf: str
    done: bool


def node_base_analysis(state: AgentState) -> AgentState:
    cfg = t.RunConfig(sample_rows=int(state["sample_rows"]))
    analysis = t.run_analysis(state["csv"], output_root=state["out"], cfg=cfg)

    run_dir = str((Path(state["out"]) / analysis["run_id"]).resolve())
    return {
        "analysis": analysis,
        "run_dir": run_dir,
        "iteration": 0,
        "extra_tables": [],
        "llm_insights_md": "",
        "done": False,
    }


def node_llm_plan(state: AgentState) -> AgentState:
    if not state.get("use_llm", True):
        return {"llm_plan": {"extra_analyses": ["payment_summary", "company_summary", "time_patterns"], "stop": True}}

    analysis = state["analysis"]
    compact = {
        "dataset": analysis.get("dataset", {}),
        "top_missing_cols": sorted(
            [{"name": c.get("name"), "missing_rate": c.get("missing_rate"), "role": c.get("role")}
             for c in analysis.get("columns", []) if isinstance(c, dict)],
            key=lambda x: (x.get("missing_rate") or 0),
            reverse=True
        )[:8],
        "tests": analysis.get("tests", []),
        "rule_insights": analysis.get("insights", []),
    }

    system = (
        "你是数据分析Agent。你将看到一个自动EDA/清洗/统计检验的结构化摘要。"
        "你的任务：规划下一步要补充哪些分析，让报告更“像人写的”。"
        "只输出 JSON 对象，不要输出其它多余文本。"
    )
    user = f"""
给定摘要（JSON）：
{json.dumps(compact, ensure_ascii=False, indent=2)}

请输出一个 JSON，格式如下：
{{
  "extra_analyses": ["payment_summary","company_summary","time_patterns"],
  "llm_insights_md": "用Markdown写：3-6条关键洞见 + 2条数据质量风险 + 2条可行动建议",
  "stop": false
}}

约束：
- extra_analyses 只能从以下枚举中选：payment_summary, company_summary, time_patterns
- stop=false 表示你希望再执行这些 extra_analyses 后再反思一次；stop=true 表示不需要额外分析直接生成报告
"""

    api_key = state.get("qwen_api_key") or os.getenv("DASHSCOPE_API_KEY") or HARDCODED_DASHSCOPE_API_KEY
    if not api_key:
        # no key => behave like no-llm mode
        return {"llm_plan": {"extra_analyses": ["payment_summary", "company_summary", "time_patterns"], "stop": True},
                "llm_insights_md": ""}

    text = qwen_chat_once(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=state.get("qwen_model", "qwen-turbo"),
        api_key=api_key,
        base_url=state.get("qwen_base_url", DEFAULT_QWEN_BASE_URL),
        temperature=0.2,
        max_tokens=900,
    )
    obj = _extract_json_object(text) or {}
    plan = {
        "extra_analyses": obj.get("extra_analyses", ["payment_summary", "company_summary", "time_patterns"]),
        "stop": bool(obj.get("stop", False)),
    }
    llm_md = obj.get("llm_insights_md", "")

    return {"llm_plan": plan, "llm_insights_md": llm_md}


def node_execute_extras(state: AgentState) -> AgentState:
    plan = state.get("llm_plan") or {}
    want = set(plan.get("extra_analyses") or [])

    # extra_analysis_tables 里会尽量生成这三类表；这里用 want 过滤一下
    tables_all = extra_analysis_tables(
        csv_path=state["csv"],
        analysis=state["analysis"],
        sample_rows=int(state["sample_rows"]),
    )

    def keep_table(tb: Dict[str, Any]) -> bool:
        title = (tb.get("title") or "").lower()
        if "payment" in title:
            return "payment_summary" in want
        if "公司" in title or "company" in title:
            return "company_summary" in want
        if "时间" in title or "hour" in title or "dow" in title:
            return "time_patterns" in want
        return True

    tables = [tb for tb in tables_all if keep_table(tb)]
    return {"extra_tables": tables}


def node_llm_reflect(state: AgentState) -> AgentState:
    """
    Ask LLM whether more loops are needed. If no LLM, stop.
    """
    it = int(state.get("iteration", 0))
    max_iter = int(state.get("max_iter", 1))
    if not state.get("use_llm", True):
        return {"done": True}

    api_key = state.get("qwen_api_key") or os.getenv("DASHSCOPE_API_KEY") or HARDCODED_DASHSCOPE_API_KEY
    if not api_key:
        return {"done": True}

    # If reached max iterations => stop
    if it >= max_iter - 1:
        return {"done": True}

    # Build reflection prompt
    tables = state.get("extra_tables") or []
    brief_tables = [{"title": tb.get("title"), "n_rows": len(tb.get("rows", [])), "note": tb.get("note", "")} for tb in tables]
    system = "你是数据分析报告的审阅者。判断是否还需要额外分析来提升洞见质量。只输出 JSON。"
    user = f"""
当前已完成：
- LLM洞见（Markdown）长度：{len(state.get("llm_insights_md",""))}
- 额外表数量：{len(tables)}
- 额外表摘要：{json.dumps(brief_tables, ensure_ascii=False, indent=2)}

请输出 JSON：
{{
  "stop": true,
  "reason": "一句话原因",
  "extra_analyses": ["payment_summary","company_summary","time_patterns"]
}}

约束：
- 如果 stop=false，则 extra_analyses 只能从枚举中选，并且尽量与上次不同（补齐短板）。
"""
    text = qwen_chat_once(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        model=state.get("qwen_model", "qwen-turbo"),
        api_key=api_key,
        base_url=state.get("qwen_base_url", DEFAULT_QWEN_BASE_URL),
        temperature=0.2,
        max_tokens=500,
    )
    obj = _extract_json_object(text) or {"stop": True, "reason": "fallback", "extra_analyses": []}
    stop = bool(obj.get("stop", True))

    if stop:
        return {"done": True}

    # continue loop: update plan, increment iteration
    new_plan = {"extra_analyses": obj.get("extra_analyses", []), "stop": False}
    return {"llm_plan": new_plan, "iteration": it + 1, "done": False}


def node_render(state: AgentState) -> AgentState:
    run_dir = Path(state["run_dir"])
    analysis_path = run_dir / "analysis.json"

    # Reload analysis.json to ensure consistency, then enrich it
    analysis = state["analysis"]
    analysis["llm"] = {
        "enabled": bool(state.get("use_llm", True)),
        "model": state.get("qwen_model", ""),
        "iteration": int(state.get("iteration", 0)),
    }
    analysis["llm_insights_md"] = state.get("llm_insights_md", "")
    analysis["extra_tables"] = state.get("extra_tables", [])

    analysis_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")

    # Render markdown from template
    template_path = Path(state["template"])
    md_text = r.render_markdown_from_template(analysis, template_path)

    # Patch: inject LLM section + tables even if template doesn't reference them
    md_text = inject_llm_and_tables_into_md(md_text, analysis.get("llm_insights_md", ""), analysis.get("extra_tables", []))

    md_path = run_dir / "report.md"
    md_path.write_text(md_text, encoding="utf-8")

    # PDF
    pdf_path = run_dir / "report.pdf"
    r.render_pdf_with_weasyprint(md_path=md_path, pdf_path=pdf_path, base_url=run_dir)

    return {"report_md": str(md_path), "report_pdf": str(pdf_path), "done": True}


def should_continue(state: AgentState) -> Literal["continue", "render"]:
    if state.get("done"):
        return "render"
    # if LLM plan says stop => render
    plan = state.get("llm_plan") or {}
    if bool(plan.get("stop", False)):
        return "render"
    return "continue"


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("base_analysis", node_base_analysis)
    g.add_node("llm_plan", node_llm_plan)
    g.add_node("execute_extras", node_execute_extras)
    g.add_node("llm_reflect", node_llm_reflect)
    g.add_node("render", node_render)

    g.add_edge(START, "base_analysis")
    g.add_edge("base_analysis", "llm_plan")
    g.add_edge("llm_plan", "execute_extras")
    g.add_edge("execute_extras", "llm_reflect")

    # Conditional loop: reflect -> (continue => plan again) or (render)
    g.add_conditional_edges(
        "llm_reflect",
        should_continue,
        {
            "continue": "llm_plan",
            "render": "render",
        },
    )

    g.add_edge("render", END)
    return g.compile()


# -------------------------
# CLI
# -------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LangGraph Agent v3: analysis -> plan -> extra -> reflect -> render")
    p.add_argument("--csv", type=str, required=True, help="Path to CSV file")
    p.add_argument("--out", type=str, default="outputs", help="Output root directory")
    p.add_argument("--template", type=str, required=True, help="Jinja2 Markdown template path, e.g. reporting/template.md")

    p.add_argument("--sample_rows", type=int, default=200_000, help="Max rows to load for analysis (EDA-friendly)")

    # LLM settings
    p.add_argument("--no_llm", action="store_true", help="Disable LLM (rule-based only)")
    p.add_argument("--qwen_api_key", type=str, default="", help="DashScope API key (optional, can hardcode in code)")
    p.add_argument("--qwen_model", type=str, default="qwen-turbo", help="Qwen model name, e.g. qwen-turbo / qwen-plus")
    p.add_argument("--qwen_base_url", type=str, default=DEFAULT_QWEN_BASE_URL, help="DashScope base_url (OpenAI-compatible)")

    # agent loop
    p.add_argument("--max_iter", type=int, default=2, help="Max agent iterations (plan/reflect loops)")

    return p


def main():
    args = build_argparser().parse_args()

    graph = build_graph()

    init_state: AgentState = {
        "csv": args.csv,
        "out": args.out,
        "template": args.template,
        "sample_rows": int(args.sample_rows),

        "use_llm": (not args.no_llm),
        "qwen_api_key": args.qwen_api_key.strip(),
        "qwen_model": args.qwen_model.strip(),
        "qwen_base_url": args.qwen_base_url.strip(),

        "max_iter": int(args.max_iter),
    }

    final_state = graph.invoke(init_state)

    print("Done.")
    print(f"Run dir: {final_state.get('run_dir')}")
    print(f"analysis.json: {Path(final_state.get('run_dir','')) / 'analysis.json'}")
    print(f"report.md: {final_state.get('report_md')}")
    print(f"report.pdf: {final_state.get('report_pdf')}")


if __name__ == "__main__":
    main()
