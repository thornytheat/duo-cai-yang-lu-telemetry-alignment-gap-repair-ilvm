"""小时摘要与分段摘要：min / max / mean / p95 + 缺口格数 + carried 格数。

- 小时摘要维持 1 小时桶口径，并标注该小时跨越了哪些采样档位分段，
  缺口/carried 在分段边界重算后的格子上统计；
- 分段摘要按每个连续档位分段汇总，质量报告据此按「各段可观测能力」解释覆盖。

p95 用线性插值分位数（与 statistics.quantiles(method='inclusive') 口径一致，但手写以便
无 numpy 也能精确控制边界）。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from . import settings


def percentile(values: list[float], p: float) -> float | None:
    """线性插值分位数，p ∈ [0, 100]。"""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    if p <= 0:
        return ordered[0]
    if p >= 100:
        return ordered[-1]
    position = (len(ordered) - 1) * p / 100.0
    lo = int(position)
    hi = min(lo + 1, len(ordered) - 1)
    frac = position - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def _iso(epoch: float) -> str:
    tz = timezone(timedelta(hours=8))
    return datetime.fromtimestamp(epoch, tz=tz).isoformat(timespec="seconds")


def _stats_for_group(group: list[dict]) -> dict:
    values = [c["value"] for c in group if c["value"] is not None]
    return {
        "cells": len(group),
        "value_cells": len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
        "p95": percentile(values, 95),
        "gap_cells": sum(1 for c in group if c["source"] == "empty"),
        "unrepaired_cells": sum(1 for c in group if c["source"] == "empty"),
        "carried_cells": sum(1 for c in group if c["carried"]),
        "interpolated_cells": sum(1 for c in group if c["source"] == "interpolated"),
        "raw_cells": sum(1 for c in group if c["source"] in ("agg", "sample")),
    }


def hourly_summary(cells: list[dict], cells_per_hour: int = 60) -> list[dict]:
    """按小时聚合网格（只统计有值的格；空/未修复格计入缺口数）。

    额外给出该小时覆盖到的档位分段（interval_s 列表），便于核对跨档位小时。
    """
    buckets: dict[int, list[dict]] = {}
    for cell in cells:
        hour = cell["bucket"] // cells_per_hour
        buckets.setdefault(hour, []).append(cell)

    out: list[dict] = []
    for hour in sorted(buckets):
        group = sorted(buckets[hour], key=lambda c: c["bucket"])
        item = _stats_for_group(group)
        tiers: list[int] = []
        for c in group:
            iv = c.get("interval_s")
            if iv is not None and iv not in tiers:
                tiers.append(iv)
        item.update(
            {
                "hour_index": hour,
                "hour_start": _iso(hour * cells_per_hour * settings.BUCKET_SECONDS),
                "segment_intervals": tiers,
                "cross_segment": len(tiers) > 1,
            }
        )
        out.append(item)
    return out


def segment_summary(cells: list[dict], segment_plans: list[dict]) -> list[dict]:
    """按连续档位分段汇总覆盖与可观测能力（质量报告按此解释每段覆盖）。"""
    by_seg: dict[int, list[dict]] = {}
    for cell in cells:
        key = cell.get("segment_index")
        if key is None:
            continue
        by_seg.setdefault(key, []).append(cell)

    out: list[dict] = []
    for plan in segment_plans:
        idx = plan["index"]
        group = by_seg.get(idx, [])
        stats = _stats_for_group(group)
        expected = plan["end_bucket"] - plan["start_bucket"] + 1
        valid = stats["value_cells"]
        raw_cov = stats["raw_cells"] / expected if expected else 0.0
        coverage = valid / expected if expected else 0.0
        out.append(
            {
                "segment_index": idx,
                "interval_s": plan["interval_s"],
                "tier": plan["tier"],
                "start_time": plan["first_time"],
                "end_time": plan["last_time"],
                "start_bucket": plan["start_bucket"],
                "end_bucket": plan["end_bucket"],
                "points": plan["points"],
                "absorbed_runs": plan["absorbed_runs"],
                "absorbed_sparse_runs": plan.get("absorbed_sparse_runs", 0),
                "expected_cells": expected,
                "observability": _observability_text(plan, stats),
                **stats,
                "coverage": coverage,
                "raw_coverage": raw_cov,
            }
        )
    return out


def _observability_text(plan: dict, stats: dict) -> str:
    """按档位可观测能力解释该段：高频档桶内可聚合，低频档每桶至多一个观测点。"""
    interval = plan["interval_s"]
    if plan["tier"] == "high":
        per_bucket = max(1, settings.BUCKET_SECONDS // max(1, interval))
        base = f"高频档 {interval}s：每分钟桶期望约 {per_bucket} 个原始点，桶内聚合 mean/max/min"
    else:
        base = (
            f"低频档 {interval}s：每分钟桶至多 1 个观测点，缺报桶只能 hold 沿用或留空，"
            "不具备桶内聚合能力"
        )
    if stats["carried_cells"]:
        base += f"；本档 {stats['carried_cells']} 格为沿用值"
    if stats["gap_cells"]:
        base += f"；{stats['gap_cells']} 格无观测（未修复）"
    if plan.get("absorbed_runs", 0) > 0:
        base += "；短暂补传突发已并入本档，未改变分段档位"
    elif plan.get("absorbed_sparse_runs", 0) > 0:
        base += "；个别孤立稀疏点已并入本档，未改变分段档位"
    return base