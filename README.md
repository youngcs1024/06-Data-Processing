# AI 驱动的大规模数据分析代理（LangGraph + 统计工具链）

一个端到端的自动化数据分析项目：输入 CSV，自动完成清洗、EDA、可视化、统计检验、建模（含 Statsmodels 诊断），并输出 `analysis.json / report.md / report.pdf`。同时提供基于 Streamlit 的 Demo（上传 CSV → 一键生成报告）。

---

## 功能概览

* **端到端自动生成**：`CSV → analysis.json → report.md → report.pdf`
* **大数据友好**：支持 `sample_rows` 抽样上限，避免超大表导致运行过慢/内存压力
* **自动清洗与特征工程**

* 去重、缺失率阈值删列
* 类型推断（numeric/categorical/datetime/text）
* 对 datetime 生成 year/month/dow 等派生特征
* **自动可视化**

* 缺失率 TopK
* 数值分布直方图
* 相关性热力图（过滤经纬度/编码列）
* 类别 TopN 条形图
* **统计检验**

* Pearson 相关、t-test/ANOVA、卡方检验（含候选过滤与优先级）
* **最小建模闭环**

* 指定 `--target` 后自动判断回归/分类，训练基线模型并输出指标：

* 回归：R² / MAE
* 分类：Accuracy / F1(macro)
* 回归任务额外输出 **Statsmodels OLS 摘要 + VIF Top10**（失败不影响主链路）
* **LLM Agent（可选）**

* LangGraph：`base_analysis → llm_plan → execute_extras → llm_reflect → render`
* LLM 输出严格 JSON 计划，工具侧执行增量汇总表（支付方式/公司/时间模式等），降低幻觉风险
* 支持禁用 LLM 或 Key 缺失自动降级

---

## 目录结构

```
.
├─ app.py                       # Streamlit Demo（上传 CSV → 报告）
├─ main.py                      # Baseline CLI（不走 LangGraph）
├─ requirements.txt
├─ tools.py                     # 清洗/EDA/统计检验/建模/产出 analysis.json
├─ agent/
│  └─ graph_v3.py               # LangGraph Agent v3（可选 LLM）
├─ reporting/
│  ├─ render.py                 # Jinja2 模板渲染 + WeasyPrint 导出 PDF
│  └─ template.md               # 报告模板
└─ scripts/
   └─ download_chicago_taxi.py  # 下载 Chicago Taxi Trips 子集
```

---

## 环境配置

### 1) 创建虚拟环境（推荐）

**macOS / Linux**

```
python -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2) 安装依赖

```
pip install -r requirements.txt
```

---

## 数据下载

项目提供下载脚本，默认下载 **Chicago Taxi Trips** 的一个 demo 子集（默认 20 万行）。

```
python scripts/download_chicago_taxi_demo.py --out data/chicago_taxi_demo.csv --limit 200000
```

可选参数示例：

* 指定年份过滤（默认 year=2023）：

```
python scripts/download_chicago_taxi_demo.py --out data/chicago_taxi_demo.csv --year 2024 --limit 200000
```

* 若 SODA 接口被限流/403，可强制走全量 CSV 下载端点（流式截断前 N 行）：

```
python scripts/download_chicago_taxi_demo.py --out data/chicago_taxi_demo.csv --force_full_download --limit 200000
```

* 可选：设置 `SOCRATA_APP_TOKEN` 减少 403/限流概率：

```
export SOCRATA_APP_TOKEN="YOUR_TOKEN"
```

---

## 运行方式一：Baseline CLI

生成 `analysis.json / report.md / report.pdf`：

```
python main.py \
  --csv data/chicago_taxi_demo.csv \
  --out outputs \
  --template reporting/template.md \
  --sample_rows 200000 \
  --target "Trip Seconds"
```

运行结束后会在 `outputs/<run_id>/` 下生成：

```
outputs/<run_id>/
  analysis.json
  figures/*.png
  report.md
  report.pdf
```

---

## 运行方式二：LangGraph Agent（可选 LLM）

### 1) 不启用 LLM（纯工具链 + Agent 流水线）

```
python agent/graph_v3.py \
  --csv data/chicago_taxi_demo.csv \
  --out outputs \
  --template reporting/template.md \
  --sample_rows 200000 \
  --target "Trip Seconds" \
  --no_llm \
  --max_iter 2
```

### 2) 启用 LLM

方式 A：环境变量

```
export DASHSCOPE_API_KEY="YOUR_DASHSCOPE_KEY"
```

然后运行（不带 `--no_llm`）：

```
python agent/graph_v3.py \
  --csv data/chicago_taxi_demo.csv \
  --out outputs \
  --template reporting/template.md \
  --sample_rows 200000 \
  --target "Trip Seconds" \
  --max_iter 2
```

方式 B：命令行显式传 Key

```
python agent/graph_v3.py \
  --csv data/chicago_taxi_demo.csv \
  --out outputs \
  --template reporting/template.md \
  --sample_rows 200000 \
  --target "Trip Seconds" \
  --qwen_api_key "YOUR_DASHSCOPE_KEY" \
  --qwen_model "qwen-turbo" \
  --max_iter 2
```

---

## 运行方式三：Streamlit Demo（上传 CSV → 一键生成报告）

启动 Demo：

```
streamlit run app.py
```

打开页面后：

1. 上传你的 CSV 文件
2. 选择输出目录、模板路径（默认 `reporting/template.md`）
3. 选择是否使用 LangGraph\_v3、是否启用 LLM（可选）
4. 点击 **生成报告**
5. 页面内可直接预览 `report.md`，并下载 `report.md / report.pdf`
6. 可在页面展开查看生成的图表（`figures/*.png`）

---

## 参数说明（常用）

* `--csv`：输入 CSV 路径
* `--out`：输出根目录（默认 `outputs/`）
* `--template`：报告模板路径
* `--sample_rows`：最多读入多少行用于分析
* `--target`：目标列名
* LangGraph 相关：

* `--no_llm`：禁用 LLM
* `--qwen_api_key / --qwen_model / --qwen_base_url`
* `--max_iter`：Agent 最大迭代次数（计划/反思循环上限）

---

## 产物说明

每次运行会在 `outputs/<run_id>/` 下生成：

* `analysis.json`：结构化分析结果（概览、清洗日志、图表、检验、洞见、建模、LLM 增量等）
* `figures/*.png`：EDA 图表
* `report.md`：Jinja2 模板渲染后的 Markdown 报告
* `report.pdf`：WeasyPrint 导出的 PDF 报告

---
