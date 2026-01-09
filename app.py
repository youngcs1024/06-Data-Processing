import time
import uuid
from pathlib import Path

import streamlit as st

from tools import RunConfig, run_analysis

# 你的 render_all 若不在 reporting/render.py，请改成：from render import render_all
from reporting.render import render_all


st.set_page_config(page_title="CSV → 自动分析报告", layout="wide")
st.title("CSV → analysis.json → report.md / report.pdf")

# -------------------------
# UI inputs
# -------------------------
uploaded = st.file_uploader("上传 CSV 文件", type=["csv"])

out_root = st.text_input("输出目录", value="outputs")
template_path = st.text_input("模板路径", value="reporting/template.md")

sample_rows = st.number_input(
    "sample_rows（最多载入行数）",
    min_value=1000,
    value=200_000,
    step=10_000,
)

TARGET_FIXED = "Trip Seconds"

use_langgraph = st.checkbox("使用 LangGraph_v3（推荐展示 Agent）", value=True)

use_llm = st.checkbox("启用 LLM（仅 LangGraph 模式有效）", value=False)
qwen_api_key = ""
qwen_model = "qwen-turbo"
qwen_base_url = ""

max_iter = st.number_input("max_iter（LangGraph 循环次数）", min_value=1, value=2, step=1)

# 只有在 LangGraph + LLM 时才显示这些输入
if use_langgraph and use_llm:
    qwen_api_key = st.text_input("DashScope API Key", type="password")
    qwen_model = st.text_input("Qwen model（默认 qwen-turbo）", value="qwen-turbo")
    qwen_base_url = st.text_input(
        "Qwen base_url（默认 DashScope OpenAI-compatible）",
        value="",  # 为空时后面自动用 graph_v3.DEFAULT_QWEN_BASE_URL
    )

run_btn = st.button("生成报告", type="primary", disabled=(uploaded is None))

# -------------------------
# Helpers
# -------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _load_graph_v3():
    """
    兼容两种仓库结构：
    1) agent/graph_v3.py  -> from agent import graph_v3
    2) graph_v3.py        -> import graph_v3
    """
    try:
        from agent import graph_v3 as g3  # type: ignore
        return g3
    except Exception:
        import graph_v3 as g3  # type: ignore
        return g3

# session state keys
for k in ["last_run_dir", "last_md_path", "last_pdf_path", "last_error"]:
    st.session_state.setdefault(k, "")

# -------------------------
# Run
# -------------------------
if run_btn and uploaded is not None:
    # basic checks
    tp = Path(template_path)
    if not tp.exists():
        st.error(f"模板文件不存在：{tp}")
    else:
        try:
            uploads_dir = Path("uploads")
            _ensure_dir(uploads_dir)

            csv_path = uploads_dir / f"{int(time.time())}_{uuid.uuid4().hex[:6]}.csv"
            csv_path.write_bytes(uploaded.getvalue())

            _ensure_dir(Path(out_root))

            with st.spinner("运行中：分析/画图/统计检验/渲染 PDF ..."):
                if use_langgraph:
                    g3 = _load_graph_v3()

                    # base_url：用户没填就用 graph_v3 默认
                    base_url = (qwen_base_url or getattr(g3, "DEFAULT_QWEN_BASE_URL", "")).strip()

                    graph = g3.build_graph()

                    init_state = {
                        "csv": str(csv_path),
                        "out": str(out_root),
                        "template": str(tp),
                        "sample_rows": int(sample_rows),
                        "target": TARGET_FIXED,

                        "use_llm": bool(use_llm),
                        "qwen_api_key": (qwen_api_key or "").strip(),
                        "qwen_model": (qwen_model or "qwen-turbo").strip(),
                        "qwen_base_url": base_url,

                        "max_iter": int(max_iter),
                    }

                    final_state = graph.invoke(init_state)

                    run_dir = Path(final_state["run_dir"])
                    md_path = Path(final_state["report_md"])
                    pdf_path = Path(final_state["report_pdf"])

                else:
                    cfg = RunConfig(sample_rows=int(sample_rows))
                    analysis = run_analysis(
                        str(csv_path),
                        output_root=str(out_root),
                        cfg=cfg,
                        target=TARGET_FIXED,
                    )
                    run_dir = Path(out_root) / analysis["run_id"]

                    outputs = render_all(run_dir=run_dir, template_path=tp)
                    md_path = outputs["md"]
                    pdf_path = outputs["pdf"]

            # store to session_state to survive reruns
            st.session_state["last_run_dir"] = str(run_dir)
            st.session_state["last_md_path"] = str(md_path)
            st.session_state["last_pdf_path"] = str(pdf_path)
            st.session_state["last_error"] = ""

            st.success(f"完成：{run_dir}")

        except Exception as e:
            st.session_state["last_error"] = repr(e)
            st.error("运行失败，请检查错误信息：")
            st.exception(e)

# -------------------------
# Display last result (persistent)
# -------------------------
if st.session_state.get("last_error"):
    st.warning(f"上一次运行错误：{st.session_state['last_error']}")

md_path_str = st.session_state.get("last_md_path", "")
pdf_path_str = st.session_state.get("last_pdf_path", "")
run_dir_str = st.session_state.get("last_run_dir", "")

if md_path_str and Path(md_path_str).exists():
    st.subheader("report.md（渲染预览）")
    md_text = Path(md_path_str).read_text(encoding="utf-8", errors="ignore")
    st.markdown(md_text)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "下载 report.md",
            data=md_text.encode("utf-8"),
            file_name="report.md",
            mime="text/markdown",
        )
    with col2:
        if pdf_path_str and Path(pdf_path_str).exists():
            st.download_button(
                "下载 report.pdf",
                data=Path(pdf_path_str).read_bytes(),
                file_name="report.pdf",
                mime="application/pdf",
            )
        else:
            st.info("PDF 尚未生成或路径不存在。")

    # 可选：展示 figures
    if run_dir_str:
        figs_dir = Path(run_dir_str) / "figures"
        if figs_dir.exists():
            fig_files = sorted(figs_dir.glob("*.png"))
            if fig_files:
                with st.expander("查看生成的图表（figures/*.png）", expanded=False):
                    for fp in fig_files:
                        st.image(str(fp), caption=fp.name, use_container_width=True)
