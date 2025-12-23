from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Optional

import requests

DEFAULT_DATASET_ID = "wrvz-psew"

SODA_CSV_URL = "https://data.cityofchicago.org/resource/{dataset_id}.csv"
FULL_DOWNLOAD_URL = "https://data.cityofchicago.org/api/views/{dataset_id}/rows.csv?accessType=DOWNLOAD"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download a demo subset of Chicago Taxi Trips as CSV (with fallback).")
    p.add_argument("--out", default="data/chicago_taxi_demo.csv", help="Output CSV path")
    p.add_argument("--dataset_id", default=DEFAULT_DATASET_ID, help="Dataset id, default: wrvz-psew")

    # how many rows to save into demo file (excluding header)
    p.add_argument("--limit", type=int, default=200_000, help="Number of rows to save to the output CSV")

    # SODA query options
    p.add_argument("--year", type=int, default=2023, help="Filter by year on trip_start_timestamp (SODA only)")
    p.add_argument("--no_year_filter", action="store_true", help="Disable year filter even in SODA mode")
    p.add_argument("--timeout", type=int, default=180, help="HTTP timeout seconds")

    # behavior
    p.add_argument("--force_full_download", action="store_true",
                   help="Skip SODA and directly use the full CSV download endpoint (stream + cut).")
    p.add_argument("--soda_limit", type=int, default=200_000,
                   help="SODA $limit (usually same as --limit)")
    p.add_argument("--offset", type=int, default=0, help="SODA $offset")

    return p.parse_args()


def _browser_headers() -> dict:
    # 403 有时跟“太像爬虫”有关，尽量伪装成浏览器请求头
    return {
        "Accept": "text/csv,application/json;q=0.9,*/*;q=0.8",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://data.cityofchicago.org/",
        "Connection": "keep-alive",
    }


def try_download_via_soda(
    out_path: Path,
    dataset_id: str,
    limit: int,
    offset: int,
    year: Optional[int],
    year_filter: bool,
    timeout: int,
) -> None:
    """
    Try Socrata SODA endpoint with optional App Token. Raises HTTPError on failure.
    """
    url = SODA_CSV_URL.format(dataset_id=dataset_id)
    params = {
        "$limit": str(limit),
        "$offset": str(offset),
        "$order": "trip_start_timestamp DESC",
    }

    if year_filter and year is not None:
        y = year
        params["$where"] = (
            f"trip_start_timestamp >= '{y}-01-01T00:00:00.000' "
            f"AND trip_start_timestamp < '{y+1}-01-01T00:00:00.000'"
        )

    headers = _browser_headers()

    token = os.getenv("SOCRATA_APP_TOKEN", "").strip()
    if token:
        # 官方推荐用 X-App-Token 头传 token
        headers["X-App-Token"] = token

    print("Requesting (SODA):", url)
    print("Params:", params)
    if token:
        print("Using SOCRATA_APP_TOKEN from env (X-App-Token).")
    else:
        print("No SOCRATA_APP_TOKEN found; may be rate-limited / blocked (403).")

    with requests.get(url, params=params, headers=headers, stream=True, timeout=timeout) as r:
        # 让上层看到具体 HTTP 状态码（比如 403）
        r.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)

    print(f"Saved demo CSV via SODA: {out_path}")


def download_and_cut_full_csv(
    out_path: Path,
    dataset_id: str,
    limit_rows: int,
    timeout: int,
) -> None:
    """
    Download from the official full CSV download endpoint, but stream and keep only:
      header + first `limit_rows` rows.
    This avoids downloading the entire (very large) file.
    """
    url = FULL_DOWNLOAD_URL.format(dataset_id=dataset_id)
    headers = _browser_headers()

    print("Requesting (FULL DOWNLOAD, stream+cut):", url)
    print(f"Will save header + first {limit_rows} rows.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # iter_lines 会按行产出（假设 CSV 记录不跨行，绝大多数数据集是这样）
    with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
        r.raise_for_status()

        lines = r.iter_lines(decode_unicode=True)

        # 读 header
        header_line = next(lines, None)
        if header_line is None:
            raise RuntimeError("Empty response: cannot read CSV header.")

        # 用 csv 规范写出，避免编码/换行问题
        with out_path.open("w", encoding="utf-8", newline="") as f_out:
            writer = csv.writer(f_out)

            header = next(csv.reader([header_line]))
            writer.writerow(header)

            kept = 0
            for line in lines:
                if not line:
                    continue
                row = next(csv.reader([line]))
                writer.writerow(row)
                kept += 1
                if kept >= limit_rows:
                    break

    print(f"Saved demo CSV via full-download stream+cut: {out_path}")


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)

    if args.force_full_download:
        download_and_cut_full_csv(
            out_path=out_path,
            dataset_id=args.dataset_id,
            limit_rows=args.limit,
            timeout=args.timeout,
        )
        return

    # 默认：先 SODA（可 year filter），失败（403 等）再 fallback
    try:
        try_download_via_soda(
            out_path=out_path,
            dataset_id=args.dataset_id,
            limit=args.soda_limit,
            offset=args.offset,
            year=args.year,
            year_filter=(not args.no_year_filter),
            timeout=args.timeout,
        )
    except requests.HTTPError as e:
        status = getattr(e.response, "status_code", None)
        print(f"SODA failed with HTTPError (status={status}): {e!r}", file=sys.stderr)
        print("Falling back to full CSV download (stream+cut)...", file=sys.stderr)

        download_and_cut_full_csv(
            out_path=out_path,
            dataset_id=args.dataset_id,
            limit_rows=args.limit,
            timeout=args.timeout,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Download failed:", repr(e), file=sys.stderr)
        sys.exit(1)
