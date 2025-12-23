from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# -------------------------
# Config / helpers
# -------------------------

@dataclass
class RunConfig:
    sample_rows: int = 200_000          # 读入最多多少行用于分析(大数据友好)
    topk_cols: int = 12                 # 缺失/绘图时取TopK列
    topk_categories: int = 10           # 类别TopN
    max_numeric_plots: int = 3          # 数值列画几个分布图
    max_cat_plots: int = 3              # 类别列画几个TopN条形图
    corr_max_features: int = 12 #20         # 相关性最多取多少个数值列(避免爆炸)
    missing_drop_threshold: float = 0.95  # 缺失率过高的列直接丢弃
    random_seed: int = 42


def _safe_float(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sanitize_filename(name: str, max_len: int = 80) -> str:
    """
    把列名变成安全文件名：空格/特殊字符 -> 下划线，并限制长度。
    """
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name)).strip("_")
    if not s:
        s = "col"
    return s[:max_len]


# -------------------------
# 1) Load
# -------------------------

def load_csv(csv_path: str, cfg: RunConfig) -> Tuple[pd.DataFrame, Dict]:
    """
    大数据友好：只读前 sample_rows 行(足够做EDA/检验/画图)
    """
    df = pd.read_csv(csv_path, nrows=cfg.sample_rows, low_memory=False)
    meta = {
        "filename": os.path.basename(csv_path),
        "loaded_rows": int(len(df)),
        "loaded_cols": int(df.shape[1]),
    }
    return df, meta


# -------------------------
# 2) Infer roles (improved)
# -------------------------
from typing import Dict
import pandas as pd

def _tokens(col: str) -> list[str]:
    # 把列名切成 token，避免子串误伤：
    # "Pickup Centroid Latitude" -> ["pickup","centroid","latitude"]
    return [t for t in re.split(r"[^a-z0-9]+", str(col).lower()) if t]

def _is_latlon_column_name(col: str) -> bool:
    """
    用 token 判断经纬度列，避免子串误伤：
    - "Pickup Centroid Latitude" -> latitude
    - "Pickup Centroid Longitude" -> longitude
    - 也兼容 lat/lon/lng
    """
    toks = set(_tokens(col))
    return (
        "latitude" in toks
        or "longitude" in toks
        or "lat" in toks
        or "lon" in toks
        or "lng" in toks
    )

def _is_code_like_column_name(col: str) -> bool:
    """
    判断列名是否像“编码/区域ID”而不是连续变量。
    关键：用 token 匹配，避免把 centroid 误判成 id。
    """
    toks = set(_tokens(col))

    # tract / geoid / zip / zone / code / id 这类 token 基本都是编码意义
    if "tract" in toks:
        return True
    if "geoid" in toks:
        return True
    if "zipcode" in toks or "zip" in toks:
        return True
    if "zone" in toks:
        return True
    if "code" in toks:
        return True
    if "id" in toks:
        return True

    # "community area" 会拆成 community + area
    if "community" in toks and "area" in toks:
        return True

    # # 可选：单独的 area 也可能是编码（如果你觉得误伤多，可以删掉这行）
    # if "area" in toks:
    #     return True

    return False

def infer_column_roles(df: pd.DataFrame) -> Dict[str, str]:
    roles: Dict[str, str] = {}

    time_keywords = (
        "date", "time", "timestamp", "datetime",
        "created", "updated", "start", "end"
    )

    for col in df.columns:
        s = df[col]

        # bool 作为类别
        if pd.api.types.is_bool_dtype(s):
            roles[col] = "categorical"
            continue

        # --------- 数值列先判断是不是“编码列” ---------
        if pd.api.types.is_numeric_dtype(s):
            col_lower = str(col).lower()

            # 经纬度/坐标列永远视为 numeric
            if ("latitude" in col_lower) or ("longitude" in col_lower) or ("lat" in col_lower) or ("lon" in col_lower) or ("lng" in col_lower):
                roles[col] = "numeric"
            elif _is_code_like_column_name(col):
                roles[col] = "categorical"   # tract/area 等编码列
            else:
                roles[col] = "numeric"
            continue

        # datetime dtype
        if pd.api.types.is_datetime64_any_dtype(s):
            roles[col] = "datetime"
            continue

        # pandas category dtype
        if pd.api.types.is_categorical_dtype(s):
            roles[col] = "categorical"
            continue

        # object：轻量推断
        if s.dtype == "object":
            s_nonnull = s.dropna().astype(str)
            sample = s_nonnull.head(2000)

            # datetime 尝试：只有列名像时间才解析(减少 warning/加速)
            col_lower = str(col).lower()
            maybe_time = any(k in col_lower for k in time_keywords)
            if maybe_time and len(sample) > 0:
                dt = pd.to_datetime(sample, errors="coerce")
                if float(dt.notna().mean()) > 0.8:
                    roles[col] = "datetime"
                    continue

            # numeric 尝试
            if len(sample) > 0:
                num = pd.to_numeric(sample, errors="coerce")
                if float(num.notna().mean()) > 0.8:
                    # 如果列名像编码,即便能转数字也当 categorical
                    roles[col] = "categorical" if _is_code_like_column_name(col) else "numeric"
                    continue

            # text vs categorical
            nunique = int(s.nunique(dropna=True))
            ratio = nunique / max(1, len(s))
            roles[col] = "text" if (ratio > 0.5 or nunique > 200) else "categorical"
            continue

        # fallback
        roles[col] = "categorical"

    return roles
# -------------------------
# 3) Profile
# -------------------------

def profile_dataset(df: pd.DataFrame, roles: Dict[str, str], cfg: RunConfig) -> Dict:
    cols_profile = []
    for col in df.columns:
        s = df[col]
        missing_rate = float(s.isna().mean())
        nunique = int(s.nunique(dropna=True))
        item = {
            "name": col,
            "role": roles.get(col, "unknown"),
            "dtype": str(s.dtype),
            "missing_rate": missing_rate,
            "n_unique": nunique,
        }
        if roles.get(col) == "numeric":
            s_num = pd.to_numeric(s, errors="coerce")
            desc = s_num.describe(percentiles=[0.25, 0.5, 0.75])
            item["summary"] = {
                "mean": _safe_float(desc.get("mean")),
                "std": _safe_float(desc.get("std")),
                "min": _safe_float(desc.get("min")),
                "p25": _safe_float(desc.get("25%")),
                "median": _safe_float(desc.get("50%")),
                "p75": _safe_float(desc.get("75%")),
                "max": _safe_float(desc.get("max")),
            }
        elif roles.get(col) in ("categorical",):
            vc = s.astype("object").fillna("NaN").value_counts().head(cfg.topk_categories)
            item["top_values"] = [{"value": str(k), "count": int(v)} for k, v in vc.items()]
        cols_profile.append(item)

    dataset_profile = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "memory_mb": float(df.memory_usage(deep=True).sum() / (1024 ** 2)),
        "columns": cols_profile,
    }
    return dataset_profile


# -------------------------
# 4) Cleaning
# -------------------------

def clean_dataset(df: pd.DataFrame, roles: Dict[str, str], cfg: RunConfig) -> Tuple[pd.DataFrame, Dict]:
    df_clean = df.copy()
    log = {"dropped_duplicates": 0, "dropped_columns": [], "imputations": [], "type_conversions": []}

    # 4.1 去重
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    log["dropped_duplicates"] = int(before - len(df_clean))

    # 4.2 丢弃缺失率极高列
    missing_rates = df_clean.isna().mean()
    drop_cols = [c for c, r in missing_rates.items() if float(r) >= cfg.missing_drop_threshold]
    if drop_cols:
        df_clean = df_clean.drop(columns=drop_cols)
        for c in drop_cols:
            log["dropped_columns"].append({"name": c, "reason": f"missing_rate>={cfg.missing_drop_threshold}"})
            roles.pop(c, None)

    # 4.3 类型转换：对推断为 datetime/numeric 的 object 做转换
    for col in list(df_clean.columns):
        role = roles.get(col)
        if role == "datetime":
            if not pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                converted = pd.to_datetime(df_clean[col], errors="coerce")
                df_clean[col] = converted
                log["type_conversions"].append({"column": col, "to": "datetime"})
        elif role == "numeric":
            if not pd.api.types.is_numeric_dtype(df_clean[col]):
                converted = pd.to_numeric(df_clean[col], errors="coerce")
                df_clean[col] = converted
                log["type_conversions"].append({"column": col, "to": "numeric"})

    # 4.4 缺失值填补（关键改造）
    for col in df_clean.columns:
        s = df_clean[col]
        if s.isna().sum() == 0:
            continue

        role = roles.get(col, "unknown")
        is_code_like = _is_code_like_column_name(col)

        # A) 编码列 / 类别列：统一用显式缺失类别，避免 mode/median 造成偏置
        #    注意：即使它的 dtype 是 numeric，我们也把它当 categorical 来处理。
        if is_code_like or role in ("categorical", "text"):
            fill_value = "Missing"
            df_clean[col] = s.astype("object").fillna(fill_value)
            log["imputations"].append({"column": col, "method": "missing_category", "value": fill_value})
            continue

        # B) 真正的连续数值列：用 median（稳健）
        if pd.api.types.is_numeric_dtype(s) or role == "numeric":
            s_num = pd.to_numeric(s, errors="coerce")
            fill_value = float(s_num.median(skipna=True)) if s_num.notna().any() else 0.0
            df_clean[col] = s_num.fillna(fill_value)
            log["imputations"].append({"column": col, "method": "median", "value": fill_value})
            continue

        # C) datetime：保持不填（也可按需求填众数日期）
        if role == "datetime":
            pass
    
    return df_clean, log

# -------------------------
# 5) Feature engineering (minimal)
# -------------------------

def feature_engineering(df: pd.DataFrame, roles: Dict[str, str]) -> Tuple[pd.DataFrame, Dict]:
    df_fe = df.copy()
    log = {"added_features": []}

    # 对 datetime 列提取 year/month/dayofweek
    for col, role in roles.items():
        if role == "datetime" and col in df_fe.columns and pd.api.types.is_datetime64_any_dtype(df_fe[col]):
            df_fe[f"{col}_year"] = df_fe[col].dt.year
            df_fe[f"{col}_month"] = df_fe[col].dt.month
            df_fe[f"{col}_dow"] = df_fe[col].dt.dayofweek
            log["added_features"].extend([f"{col}_year", f"{col}_month", f"{col}_dow"])

    return df_fe, log


# -------------------------
# 6) Plots
# -------------------------

def make_plots(df_raw: pd.DataFrame, df_clean: pd.DataFrame, roles: Dict[str, str], out_dir: str, cfg: RunConfig) -> List[Dict]:
    ensure_dir(out_dir)
    plots = []

    # 6.1 缺失率TopK(raw)
    miss = df_raw.isna().mean().sort_values(ascending=False).head(cfg.topk_cols)
    fig_path = os.path.join(out_dir, "missing_topk.png")
    plt.figure()
    miss[::-1].plot(kind="barh")
    plt.title("Missing Rate Top Columns (raw)")
    plt.xlabel("missing rate")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    plots.append({"id": "missing_topk", "title": "缺失率最高的列(TopK)", "path": fig_path, "caption": "基于原始数据的缺失率统计。"})
    
    # 6.2 数值分布(选几个方差较大的数值列)
    num_cols = [c for c, r in roles.items() if r == "numeric" and c in df_clean.columns]
    if num_cols:
        variances = df_clean[num_cols].var(numeric_only=True).sort_values(ascending=False)
        pick = list(variances.head(cfg.max_numeric_plots).index)
        for col in pick:
            safe = sanitize_filename(col)
            fig_path = os.path.join(out_dir, f"hist_{safe}.png")
            plt.figure()
            df_clean[col].dropna().hist(bins=40)
            plt.title(f"Distribution: {col}")
            plt.xlabel(col)
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()
            plots.append({"id": f"hist_{col}", "title": f"数值列分布：{col}", "path": fig_path, "caption": "清洗后数据的直方图分布。"})
    
    # 6.3 相关性热力图（方差TopN + 排除 lat/lon + 排除编码列）
    if len(num_cols) >= 2:
        # ① 过滤掉经纬度/编码列
        cand = [
            c for c in num_cols
            if (c in df_clean.columns)
            and (not _is_latlon_column_name(c))
            and (not _is_code_like_column_name(c))
        ]

        if len(cand) >= 2:
            # ② 转数值并计算方差（避免 object 混进来）
            df_num = df_clean[cand].apply(pd.to_numeric, errors="coerce")

            # （可选但推荐）过滤掉“几乎全空”的列
            nonnull_rate = df_num.notna().mean()
            df_num = df_num.loc[:, nonnull_rate >= 0.95]

            # ③ 方差TopN（去掉方差为0/NaN的列）
            var = df_num.var(numeric_only=True).replace([np.inf, -np.inf], np.nan).dropna()
            var = var[var > 0]

            pick = list(var.sort_values(ascending=False).head(cfg.corr_max_features).index)

            if len(pick) >= 2:
                corr = df_num[pick].corr(numeric_only=True)

                fig_path = os.path.join(out_dir, "corr_heatmap.png")
                plt.figure(figsize=(8, 6))
                plt.imshow(corr.values, aspect="auto")
                plt.title("Correlation Heatmap (numeric, variance-top)")
                plt.xticks(range(len(pick)), pick, rotation=90, fontsize=7)
                plt.yticks(range(len(pick)), pick, fontsize=7)
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(fig_path, dpi=150)
                plt.close()

                plots.append({
                    "id": "corr_heatmap",
                    "title": "数值特征相关性热力图(方差TopN,已过滤经纬度/编码列）",
                    "path": fig_path,
                    "caption": "清洗后数值列相关性。列集合按方差TopN选择, 并排除经纬度/编码列。"
                })

    # 6.4 类别TopN条形图
    cat_cols = [
        c for c, r in roles.items()
        if r == "categorical" and c in df_clean.columns and (not _is_code_like_column_name(c))
    ]
    # 按 nunique 从小到大选几个(Payment Type 通常会被选到)
    cat_cols = sorted(cat_cols, key=lambda c: df_clean[c].nunique(dropna=True))
    for col in cat_cols[: cfg.max_cat_plots]:
        vc = df_clean[col].astype("object").value_counts().head(cfg.topk_categories)
        safe = sanitize_filename(col)
        fig_path = os.path.join(out_dir, f"cat_top_{safe}.png")
        plt.figure()
        vc[::-1].plot(kind="barh")
        plt.title(f"Top Categories: {col}")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
        plots.append({"id": f"cat_{col}", "title": f"类别TopN：{col}", "path": fig_path, "caption": "清洗后类别频次TopN。"})
    
    return plots


# -------------------------
# 7) Statistical tests (minimal automatic)
# -------------------------
def _canon(s: str) -> str:
    """列名归一化：忽略大小写/空格/特殊字符,方便匹配"""
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def pick_preferred_column(cols: list[str], preferred: list[str]) -> str | None:
    """
    在 cols 中优先挑 preferred 里出现的列(不区分大小写/空格/符号)
    """
    cols_map = {_canon(c): c for c in cols}
    for p in preferred:
        key = _canon(p)
        if key in cols_map:
            return cols_map[key]
    return None

def filter_cat_candidates(df: pd.DataFrame, cols: list[str], max_nunique: int = 50) -> list[str]:
    """
    过滤掉不适合做卡方/分组检验的类别列：
    - 编码列(tract/area/id/code 等)优先排除
    - 基数过大(类别太多)排除,避免列联表爆炸
    """
    kept = []
    for c in cols:
        # 避开编码列(你如果已实现 _is_code_like_column_name,就用它)
        try:
            if _is_code_like_column_name(c):
                continue
        except NameError:
            pass

        nunique = int(df[c].nunique(dropna=True))
        if nunique <= max_nunique and nunique >= 2:
            kept.append(c)
    return kept

def filter_num_candidates(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """
    过滤掉不适合做数值检验的 numeric 列：
    - 编码列(tract/area/id/code 等)排除
    - 常数/近常数排除
    """
    kept = []
    for c in cols:
        try:
            if _is_code_like_column_name(c):
                continue
        except NameError:
            pass

        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().sum() < 30:
            continue
        if float(s.nunique(dropna=True)) <= 1:
            continue
        kept.append(c)
    return kept


def format_p_value(p: float) -> str:
    """
    把极小 p 值从 0.0 / 下溢,格式化成更专业的展示形式
    """
    if p is None:
        return "NA"
    try:
        p = float(p)
    except Exception:
        return "NA"
    if np.isnan(p):
        return "NA"

    # 避免出现 0.0(通常是下溢)
    if p == 0.0 or p < 1e-300:
        return "<1e-300"

    # 常用显示规则：小于0.001用科学计数法,否则用定点
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def is_trivial_total_component_pair(x: str, y: str) -> bool:
    """
    过滤掉 'total' 与其组成项(fare/tips/extras/tolls 等)的组合：
    例如 Fare vs Trip Total、Tips vs Trip Total 等
    """
    xl, yl = x.lower(), y.lower()

    # total 的关键词(按你数据列名可扩展)
    total_keys = ["total"]
    # 组成项关键词(Chicago Taxi Trips 里常见)
    component_keys = ["fare", "tip", "tips", "extras", "tolls"]

    x_is_total = any(k in xl for k in total_keys)
    y_is_total = any(k in yl for k in total_keys)

    x_is_comp = any(k in xl for k in component_keys)
    y_is_comp = any(k in yl for k in component_keys)

    # total vs component => trivial
    return (x_is_total and y_is_comp) or (y_is_total and x_is_comp)


def is_valid_numeric_pair(x: str, y: str) -> bool:
    """
    数值-数值相关候选过滤器：返回 True 表示允许用于 Pearson 检验
    """
    if x == y:
        return False

    # 规则0：过滤经纬度列（避免选到 Lat vs Lon 这种洞见弱的组合）
    if _is_latlon_column_name(x) or _is_latlon_column_name(y):
        return False

    # 规则1：过滤 total 与其组成项的组合
    if is_trivial_total_component_pair(x, y):
        return False

    # 规则2：过滤明显编码列(tract/area/id/code 等)
    if _is_code_like_column_name(x) or _is_code_like_column_name(y):
        return False

    return True


def run_stat_tests(df: pd.DataFrame, roles: Dict[str, str], cfg: RunConfig, target: Optional[str] = None) -> List[Dict]:
    """
    自动做几类检验：
    - numeric vs numeric：Pearson 相关显著性(挑相关性最高的一对)
    - categorical vs numeric：若存在类别列 & 数值列,做 t-test/ANOVA(挑一个组合)
    - categorical vs categorical：卡方(挑一个组合)
    """
    tests: List[Dict] = []

    num_cols = [c for c, r in roles.items() if r == "numeric" and c in df.columns]
    cat_cols = [c for c, r in roles.items() if r == "categorical" and c in df.columns]

    # 7.1 numeric-numeric Pearson：挑绝对相关最大的一对(加过滤)
    if len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True).abs()

        # 取上三角所有 pair,按相关性从大到小排序
        pairs = []
        cols = list(corr.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                x, y = cols[i], cols[j]
                v = float(corr.iloc[i, j])
                if np.isnan(v):
                    continue
                pairs.append((v, x, y))

        pairs.sort(key=lambda t: t[0], reverse=True)

        chosen = None
        for v, x, y in pairs:
            if is_valid_numeric_pair(x, y):
                chosen = (x, y)
                break

        # 如果全被过滤掉,就退回到相关最高的那一对
        if chosen is None and pairs:
            _, x, y = pairs[0]
            chosen = (x, y)

        if chosen is not None:
            x, y = chosen
            xvals = df[x].astype(float)
            yvals = df[y].astype(float)
            r, p = stats.pearsonr(xvals, yvals)
            tests.append({
                "test": "pearsonr",
                "x": x, "y": y,
                "statistic": float(r),
                "p_value": float(p),
                "p_value_fmt": format_p_value(p),  # 如果你已加 p 格式化
                "note": "数值-数值相关显著性检验(Pearson,已过滤 trivial total/component 列对)。"
            })

    # 7.2 categorical-numeric：t-test(两类) 或 ANOVA(多类)
    if cat_cols and num_cols:
        # 先过滤候选(避免 tract/area 之类编码列干扰)
        cat_candidates = filter_cat_candidates(df, cat_cols, max_nunique=50)
        num_candidates = filter_num_candidates(df, num_cols)

        # 优先选 Payment Type；数值列优先 Trip Seconds(更像“时长差异”分析)
        ccol = pick_preferred_column(cat_candidates, ["Payment Type"]) or (cat_candidates[0] if cat_candidates else None)
        ncol = pick_preferred_column(num_candidates, ["Trip Seconds", "Trip Miles", "Fare"]) or (num_candidates[0] if num_candidates else None)

        if ccol is not None and ncol is not None:
            groups = [g[ncol].values for _, g in df.groupby(ccol) if len(g) >= 5]
            if len(groups) == 2:
                t, p = stats.ttest_ind(groups[0], groups[1], equal_var=False, nan_policy="omit")
                tests.append({
                    "test": "t_test", "x": ccol, "y": ncol,
                    "statistic": float(t),
                    "p_value": float(p),
                    "p_value_fmt": format_p_value(p),
                    "note": "两组均值差异检验(Welch t-test)。"
                })
            elif len(groups) >= 3:
                f, p = stats.f_oneway(*groups[:5])  # 限制组数,避免太慢
                tests.append({
                    "test": "anova", "x": ccol, "y": ncol,
                    "statistic": float(f),
                    "p_value": float(p),
                    "p_value_fmt": format_p_value(p),
                    "note": "多组均值差异检验(ANOVA,组数做了上限)。"
                })

    # 7.3 categorical-categorical：卡方
    # 7.3 categorical-categorical：卡方
    if len(cat_cols) >= 2:
        cat_candidates = filter_cat_candidates(df, cat_cols, max_nunique=80)

        # 优先 Payment Type vs Company
        x = pick_preferred_column(cat_candidates, ["Payment Type"])
        y = pick_preferred_column(cat_candidates, ["Company"])

        # 如果没有凑齐，就从候选里挑两个（基数从小到大）
        if x is None or y is None or x == y:
            cat_sorted = sorted(cat_candidates, key=lambda c: df[c].nunique(dropna=True))
            if len(cat_sorted) >= 2:
                x, y = cat_sorted[0], cat_sorted[1]
            else:
                x, y = None, None

        if x is not None and y is not None and x != y:
            ct = pd.crosstab(df[x], df[y])
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            tests.append({
                "test": "chi2", "x": x, "y": y,
                "statistic": float(chi2),
                "p_value": float(p),
                "p_value_fmt": format_p_value(p),
                "note": "类别-类别独立性检验（卡方）。"
            })

    return tests


# -------------------------
# 8) Rule-based insights (no LLM)
# -------------------------

from typing import Dict, List

def generate_rule_based_insights(profile: Dict, cleaning_log: Dict, tests: List[Dict], cfg: RunConfig) -> List[str]:
    insights: List[str] = []

    # 1) 缺失率高的列（基于 profile，通常是 raw）
    cols = profile.get("columns", [])
    if cols:
        miss_sorted = sorted(cols, key=lambda x: float(x.get("missing_rate", 0.0)), reverse=True)
        if miss_sorted and float(miss_sorted[0].get("missing_rate", 0.0)) > 0.3:
            top = miss_sorted[0]
            insights.append(
                f"缺失率最高的列是 `{top.get('name','')}`（missing_rate={float(top.get('missing_rate',0.0)):.2%}），"
                f"建议确认采集流程或考虑剔除/重采样。"
            )

    # 2) 去重
    dropped = int(cleaning_log.get("dropped_duplicates", 0) or 0)
    if dropped > 0:
        insights.append(f"检测到并移除了重复行 {dropped} 条，后续分析基于去重后的数据。")

    # 3) 显著性检验：用 p_value_fmt（避免 p=0）
    for t in (tests or []):
        p = t.get("p_value", 1.0)
        try:
            p_float = float(p)
        except Exception:
            p_float = 1.0

        if p_float < 0.05:
            # 优先用格式化后的 p 值
            p_show = t.get("p_value_fmt")
            if not p_show:
                # 兼容旧数据：如果没有 p_value_fmt，就用 format_p_value
                try:
                    p_show = format_p_value(p_float)  # format_p_value 在 tools.py 里定义
                except Exception:
                    # 最后兜底
                    p_show = str(p)

            insights.append(
                f"统计检验 `{t.get('test','')}` 显示 `{t.get('x','')}` 与 `{t.get('y','')}` "
                f"存在显著关系（p={p_show}）。"
            )

    if not insights:
        insights.append("未发现特别突出的缺失/重复/显著关系信号（可能需要指定目标列或更细分的业务问题）。")

    return insights


# -------------------------
# 9) Orchestration
# -------------------------

def run_analysis(csv_path: str, output_root: str = "outputs", cfg: Optional[RunConfig] = None) -> Dict:
    cfg = cfg or RunConfig()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

    out_dir = os.path.join(output_root, run_id)
    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(fig_dir)

    df_raw, meta = load_csv(csv_path, cfg)
    roles = infer_column_roles(df_raw)

    profile_raw = profile_dataset(df_raw, roles, cfg)
    df_clean, cleaning_log = clean_dataset(df_raw, roles, cfg)
    df_fe, fe_log = feature_engineering(df_clean, roles)

    plots = make_plots(df_raw, df_clean, roles, fig_dir, cfg)
    tests = run_stat_tests(df_clean, roles, cfg)
    insights = generate_rule_based_insights(profile_raw, cleaning_log, tests, cfg)

    analysis = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "filename": meta["filename"],
            "n_rows": profile_raw["n_rows"],
            "n_cols": profile_raw["n_cols"],
            "memory_mb": profile_raw["memory_mb"],
            "loaded_rows": meta["loaded_rows"]
        },
        "columns": profile_raw["columns"],
        "cleaning": {**cleaning_log, **fe_log},
        "plots": [
            {**p, "path": os.path.relpath(p["path"], out_dir).replace("\\", "/")}
            for p in plots
        ],
        "tests": tests,
        "insights": insights,
    }

    # save analysis.json
    with open(os.path.join(out_dir, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)

    return analysis
