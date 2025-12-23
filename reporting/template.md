# 自动化数据分析报告

- Run ID: {{ run_id }}
- 生成时间: {{ generated_at }}
- 数据集: {{ dataset.filename }}
- 行/列: {{ dataset.n_rows }} / {{ dataset.n_cols }}
- 本次载入行数(用于分析): {{ dataset.loaded_rows }}

## 1. 数据概览

### 1.1 原始数据缺失率最高的列（Top 10）
{% set cols_sorted = columns | sort(attribute="missing_rate", reverse=True) %}
{% for c in cols_sorted[:10] %}
- {{ c.name }}: {{ (c.missing_rate*100) | round(2) }}%（dtype={{ c.dtype }}，role={{ c.role }}）
{% endfor %}

## 2. 清洗与特征工程日志
- 去除重复行: {{ cleaning.dropped_duplicates }}
- 删除列: {{ cleaning.dropped_columns | length }}

### 2.1 类型转换（共 {{ cleaning.type_conversions | length }} 项）
{% if cleaning.type_conversions | length == 0 %}
无。
{% else %}
{% for x in cleaning.type_conversions %}
- {{ x.column }} -> {{ x.to }}
{% endfor %}
{% endif %}

### 2.2 缺失值填补（共 {{ cleaning.imputations | length }} 列）
{% if cleaning.imputations | length == 0 %}
无。
{% else %}
| 列 | 方法 | 填充值 |
|---|---|---|
{% for imp in cleaning.imputations[:12] %}
| {{ imp.column }} | {{ imp.method }} | {{ imp.value }} |
{% endfor %}
{% if cleaning.imputations | length > 12 %}
> 仅展示前12列填补记录，完整记录见 analysis.json。
{% endif %}
{% endif %}

### 2.3 新增特征（共 {{ cleaning.added_features | length }} 个）
{% if cleaning.added_features | length == 0 %}
无。
{% else %}
{% for f in cleaning.added_features %}
- {{ f }}
{% endfor %}
{% endif %}

## 3. 图表
{% for p in plots %}
### Figure {{ loop.index }}：{{ p.title }}
![]({{ p.path }})
{{ p.caption }}
{% endfor %}

## 4. 统计检验
{% if tests|length == 0 %}
无（数据类型限制或样本不足）。
{% else %}
{% for t in tests %}
- {{ t.test }}：{{ t.x }} vs {{ t.y }}，stat={{ t.statistic | round(4) }}，p={{ t.p_value_fmt if t.p_value_fmt is defined else t.p_value }}
  - {{ t.note }}
{% endfor %}
{% endif %}



{# ---- LLM flags ---- #}
{% set llm_info = llm if llm is defined else {} %}
{% set llm_enabled = llm_info.get("enabled", llm_info.get("used", false)) %}
{% set llm_model = llm_info.get("model", "") %}
{% set llm_error = llm_info.get("error", "") %}

{# llm_insights_md：不存在就当空字符串 #}
{% set llm_text = (llm_insights_md if llm_insights_md is defined else "") %}
{% set llm_has_text = (llm_text | trim | length > 0) %}

{# “真正使用LLM增强”= 启用 + 有文本 + 没报错 #}
{% set llm_used = llm_enabled and llm_has_text and ((llm_error | trim) == "") %}

## 5. 洞见总结{% if llm_used %}（含LLM增强：{{ llm_model }}）{% else %}（仅规则）{% endif %}

{% if llm_enabled and ((llm_error | trim) != "") %}
> LLM 调用失败：{{ llm_error }}
{% endif %}

### 5.1 规则洞见（来自自动统计/规则引擎）
{% if insights is defined and insights|length > 0 %}
{% for s in insights %}
- {{ s }}
{% endfor %}
{% else %}
无（本次未生成规则洞见）。
{% endif %}

{% if llm_enabled and ((llm_error | trim) == "") %}
### 5.2 LLM 洞见摘要
{% if llm_has_text %}
{{ llm_text }}
{% else %}
无（本次 LLM 摘要为空；详细洞见见第 6 章）。
{% endif %}
{% endif %}
