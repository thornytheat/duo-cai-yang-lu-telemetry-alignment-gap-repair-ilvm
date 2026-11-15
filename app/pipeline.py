"""编排：一条 (设备, 指标) 的对齐流水线。

1 数据导入（序列与基准选择）→ 2 漂移估计 → 3 网格重采样（分段）→ 4 缺口判定
→ 5 小时/分段摘要 → 6 质量评分。路由只调用本模块，不写业务规则。

本模块同时暴露无副作用的纯函数部件（compute_grid / compute_drift / assemble_payload），
供 publication 发布器按「分段作业」断点续跑：每个档位分段是一个可独立重算/复用的作业，
全部分段完成后再组装并原子切换发布版本。
"""
from __future__ import annotations

import time

from . import drift, gaps, grid, quality, settings, summary
from .telemetry import Dataset, Series


def reference_for(dataset: Dataset, metric: str, target_device: str) -> tuple[Series | None, str | None]:
    """同一指标下采样能力最强的序列作为时钟基准；目标设备本身即基准时返回 (自身, 自身)。

    排序键考虑**档位分段**而不只是整段中位数：优先取「最高档位最密」的序列
    （min segment interval），其次整段中位数，最后设备号——在 1s/5s/60s 间切换的
    多档位序列不会因为大量 60s 段拉高中位数而失去基准资格。
    """
    candidates = [
        ser for (device, met), ser in dataset.series.items() if met == metric
    ]
    if not candidates:
        return None, None
    start_bucket, end_bucket = bucket_window(dataset)

    def key(ser: Series) -> tuple[int, int, str]:
        plans = grid.segment_plans(ser.points, ser.interval_s, start_bucket, end_bucket)
        finest = min((p["interval_s"] for p in plans), default=ser.interval_s)
        return (finest, ser.interval_s, ser.device_id)

    best = min(candidates, key=key)
    if best.device_id == target_device:
        return best, best.device_id
    return best, best.device_id


def bucket_window(dataset: Dataset) -> tuple[int, int]:
    return (
        int(dataset.window_start // settings.BUCKET_SECONDS),
        int(dataset.window_end // settings.BUCKET_SECONDS),
    )


def compute_grid(dataset: Dataset, series: Series) -> tuple[list[dict], list[dict], list[dict], dict]:
    """纯计算：分段规划 → 分段网格 → 分段缺口重算。

    返回 (segment_plans, cells, gap_result, grid_stats)。cells 已就地完成 hold 与插补。
    发布器可对每个 segment_index 单独调用一次本函数并只持久化对应桶区间，实现续跑。
    """
    segment_plans, cells = grid_stage(dataset, series)
    gap_result = gaps_stage(cells)
    grid_stats = grid.stats(cells)
    return segment_plans, cells, gap_result, grid_stats


def grid_stage(dataset: Dataset, series: Series) -> tuple[list[dict], list[dict]]:
    """分段规划 + 网格重采样（含段内/跨锚点 hold），但**不做缺口判定与插补**。

    缺口前的格子是确定性的中间态，发布器把它作为分段作业结果落库，
    缺口阶段在组装时统一重放，保证续跑与一次性流水线结果一致。
    """
    start_bucket, end_bucket = bucket_window(dataset)
    segment_plans = grid.segment_plans(series.points, series.interval_s, start_bucket, end_bucket)
    cells = grid.resample(series.points, series.interval_s, start_bucket, end_bucket)
    return segment_plans, cells


def gaps_stage(cells: list[dict]) -> dict:
    """缺口判定（含分段边界 hold 与线性插补，就地改 cells），返回缺口结论。"""
    gap_result = gaps.find_gaps(cells)
    gaps.apply_repair(cells)
    return gap_result


def compute_drift(
    dataset: Dataset,
    series: Series,
    threshold_s: float,
) -> tuple[dict, Series | None, str | None, bool]:
    """纯计算：漂移估计（基准设备返回免校时结论）。"""
    ref_series, ref_device = reference_for(dataset, series.metric, series.device_id)
    self_reference = ref_series is not None and ref_series.device_id == series.device_id
    if self_reference:
        drift_result = {
            "available": False,
            "reason": "该指标最高频序列来自本设备，作为线级时钟基准，不能与自己比对",
            "threshold_s": threshold_s,
            "offset_s": None,
            "rate_ppm": None,
            "rmse_s": None,
            "anchors": 0,
            "drifted": None,
            "verdict": "本设备是该指标的时钟基准，无需漂移校正",
            "suggestion": "基准设备时间视为可信；可对其他设备执行对齐以查看偏移。",
        }
    else:
        drift_result = drift.estimate_drift(
            ref_series.points, series.points, dataset.window_start, threshold_s
        )
        drift_result["available"] = True
        drift_result["reference_device"] = ref_device
        drift_result["reference_interval_s"] = ref_series.interval_s
    return drift_result, ref_series, ref_device, self_reference


def assemble_payload(
    dataset: Dataset,
    series: Series,
    threshold_s: float,
    stages: list[dict] | None = None,
    total_ms: float | None = None,
) -> dict:
    """纯计算：跑完整条流水线并组装结果 payload（不落库、无发布语义）。"""
    stages = stages if stages is not None else []
    started = time.perf_counter()

    def stage(name: str, detail: str, begin: float) -> None:
        stages.append({"name": name, "detail": detail, "ms": round((time.perf_counter() - begin) * 1000, 2)})

    t0 = time.perf_counter()
    stage("数据导入", f"{len(series.points)} 点 / 采样率 {series.interval_s}s", t0)

    t0 = time.perf_counter()
    drift_result, ref_series, ref_device, self_reference = compute_drift(dataset, series, threshold_s)
    if self_reference:
        stage("漂移估计", "基准设备，跳过自比", t0)
    else:
        stage("漂移估计", f"锚点 {drift_result['anchors']} 个，offset {drift_result['offset_s']:+.1f}s", t0)

    t0 = time.perf_counter()
    start_bucket, end_bucket = bucket_window(dataset)
    segment_plans, cells, gap_result, grid_stats = compute_grid(dataset, series)
    tier_text = " / ".join(f"{p['interval_s']}s×{p['buckets']}格" for p in segment_plans)
    stage(
        "网格重采样",
        f"{len(segment_plans)} 个档位分段（{tier_text}）；"
        f"{grid_stats['cells']} 格（聚合 {grid_stats['agg_cells']} / 采样 {grid_stats['sample_cells']} / 沿用 {grid_stats['held_cells']}）",
        t0,
    )

    t0 = time.perf_counter()
    stage("缺口判定", f"缺口 {gap_result['count']} 处，插补 {gap_result['interpolated_cells']} 格，未修复 {gap_result['unrepaired_cells']} 格", t0)

    t0 = time.perf_counter()
    hourly = summary.hourly_summary(cells)
    segment_stats = summary.segment_summary(cells, segment_plans)
    stage("小时摘要", f"{len(hourly)} 个小时桶 / {len(segment_stats)} 个档位分段", t0)

    t0 = time.perf_counter()
    expected = dataset.bucket_count()
    coverage = quality.coverage_report(cells, expected, gap_result, segment_stats)
    stage("质量评分", f"覆盖率 {coverage['coverage']:.4f} → {coverage['grade']} 级（{len(segment_stats)} 段）", t0)

    if total_ms is None:
        total_ms = round((time.perf_counter() - started) * 1000, 2)
    return {
        "dataset_id": dataset.dataset_id,
        "device_id": series.device_id,
        "metric": series.metric,
        "metric_label": settings.METRIC_LABEL.get(series.metric, series.metric),
        "metric_unit": settings.METRIC_UNIT.get(series.metric, ""),
        "interval_s": series.interval_s,
        "threshold_s": threshold_s,
        "window": {
            "start": grid.bucket_start_iso(start_bucket),
            "end": grid.bucket_start_iso(end_bucket),
            "cells": expected,
            "bucket_seconds": settings.BUCKET_SECONDS,
        },
        "drift": drift_result,
        "grid": grid_stats,
        "segments": segment_plans,
        "segment_stats": segment_stats,
        "gaps": gap_result,
        "coverage": coverage,
        "hourly": hourly,
        "preview": cells[:60],
        "cells": cells,
        "reference": {
            "device_id": ref_device,
            "interval_s": ref_series.interval_s if ref_series else None,
            "self_reference": self_reference,
        },
        "stages": stages,
        "total_ms": total_ms,
    }


def run_alignment(
    dataset: Dataset,
    device_id: str,
    metric: str,
    threshold_s: float = settings.DEFAULT_OFFSET_THRESHOLD_S,
) -> dict:
    """兼容入口：组装一条序列的完整对齐结果（不含发布信息）。"""
    series = dataset.series_for(device_id, metric)
    if series is None:
        raise KeyError(f"设备 {device_id} 没有指标 {metric}")
    payload = assemble_payload(dataset, series, threshold_s)
    payload.pop("cells", None)  # 旧契约只给前 60 格预览，完整网格留在发布层
    return payload


def build_report(dataset: Dataset, threshold_s: float = settings.DEFAULT_OFFSET_THRESHOLD_S) -> dict:
    rows: list[dict] = []
    for (device, metric), series in sorted(dataset.series.items()):
        payload = run_alignment(dataset, device, metric, threshold_s)
        rows.append(
            {
                "device_id": device,
                "metric": metric,
                "metric_label": payload["metric_label"],
                "interval_s": series.interval_s,
                "coverage": payload["coverage"],
                "drift": payload["drift"],
                "gaps": payload["gaps"],
                "segments": payload["segments"],
            }
        )
    return {
        "dataset_id": dataset.dataset_id,
        "threshold_s": threshold_s,
        "rows": rows,
        "summary": quality.report_summary(rows),
        "devices": quality.device_rollup(rows),
        "csv": quality.report_csv(rows),
        "segments_csv": quality.segment_csv(rows),
    }
