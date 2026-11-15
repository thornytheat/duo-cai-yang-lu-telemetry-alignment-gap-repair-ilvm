"""分段发布器：断点续跑、原子切换、幂等重跑。

一条 (设备, 指标) 序列的对齐结果按「连续采样档位分段」拆成独立作业：

- 每个档位分段是 segment_jobs 表里的一行，处理成功才落 result_json（done）；
- 中断后续跑：done 的作业**直接复用、绝不重算**，只重放 pending/failed 的分段；
- 全部分段 done 后，在**单个 SQLite 事务**里组装完整 payload、置 published 并把
  active 指针从旧版切到新版（原子切换）；任一作业失败则版本标 failed，旧版保留；
- pub_id 由输入内容哈希决定，重跑同一批数据得到同一个 pub_id、同一份结果（幂等）。

worker 可注入（默认 DefaultWorker），测试通过替换 worker 模拟某个分段失败/进程中断，
不需要 monkeypatch 内部函数。
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

from . import pipeline, settings, store
from .telemetry import Dataset, Series


class PublicationError(RuntimeError):
    """发布流程失败（作业计算错误等），message 可直接回显。"""


@dataclass
class SegmentTask:
    pub_id: str
    dataset_id: str
    device_id: str
    metric: str
    threshold_s: float
    segment_index: int
    interval_s: int


@dataclass
class DriftTask:
    pub_id: str
    dataset_id: str
    device_id: str
    metric: str
    threshold_s: float


class DefaultWorker:
    """默认作业执行器：纯内存计算，无外部副作用。可被测试替身替换。"""

    def compute_segment(self, dataset: Dataset, series: Series, task: SegmentTask) -> dict:
        # 缺口前的确定性中间态：分段规划 + 重采样（含段内 hold），不做缺口判定。
        segment_plans, raw_cells = pipeline.grid_stage(dataset, series)
        plan = next((p for p in segment_plans if p["index"] == task.segment_index), None)
        if plan is None:
            raise PublicationError(f"分段 {task.segment_index} 不存在（输入可能已变化）")
        seg_cells = [dict(c) for c in raw_cells if c.get("segment_index") == task.segment_index]

        # 缺口结论在副本上计算，只落结论（items）；格子补值在组装阶段统一确定性重放，
        # 这样中断续跑、逐段重算与一次性流水线得到的格子完全一致。
        probe = [dict(c) for c in raw_cells]
        gap_result = pipeline.gaps_stage(probe)
        seg_gaps = [g for g in gap_result["items"] if g.get("segment_index") == task.segment_index]
        return {
            "segment": plan,
            "cells": seg_cells,
            "gaps": seg_gaps,
        }

    def compute_drift(self, dataset: Dataset, series: Series, task: DriftTask) -> dict:
        drift_result, _ref, _ref_device, _self = pipeline.compute_drift(
            dataset, series, task.threshold_s
        )
        return {"drift": drift_result}


def content_hash(dataset: Dataset, series: Series, threshold_s: float) -> str:
    """输入内容指纹：数据集 sha + 序列点 + 阈值。同输入必同哈希（重跑结果不变的基础）。"""
    h = hashlib.sha256()
    h.update(dataset.sha256.encode("utf-8"))
    h.update(series.device_id.encode())
    h.update(series.metric.encode())
    h.update(f"{threshold_s:.6f}".encode())
    for ts, value in series.points:
        h.update(f"{ts:.3f}|{value!r};".encode())
    return h.hexdigest()


def make_pub_id(content: str) -> str:
    return "pub-" + hashlib.sha256(content.encode()).hexdigest()[:12]


def ensure_publication(
    dataset: Dataset,
    device_id: str,
    metric: str,
    threshold_s: float,
    worker: DefaultWorker | None = None,
) -> dict:
    """幂等入口：返回该输入对应的发布版本记录（不触发计算）。

    已存在（running/published/failed）时原样返回；不存在则按当前分段规划登记作业。
    """
    worker = worker or DefaultWorker()
    series = dataset.series_for(device_id, metric)
    if series is None:
        raise KeyError(f"设备 {device_id} 没有指标 {metric}")
    digest = content_hash(dataset, series, threshold_s)
    existing = store.get_publication_by_hash(
        dataset.dataset_id, device_id, metric, threshold_s, digest
    )
    if existing is not None:
        return existing

    start_bucket, end_bucket = pipeline.bucket_window(dataset)
    plans = pipeline.grid.segment_plans(
        series.points, series.interval_s, start_bucket, end_bucket
    )
    pub_id = make_pub_id(f"{dataset.dataset_id}|{device_id}|{metric}|{threshold_s}|{digest}")
    return store.create_publication(
        pub_id,
        dataset.dataset_id,
        device_id,
        metric,
        threshold_s,
        digest,
        segment_plans=plans,
    )


def run_publication(
    dataset: Dataset,
    device_id: str,
    metric: str,
    threshold_s: float = settings.DEFAULT_OFFSET_THRESHOLD_S,
    worker: DefaultWorker | None = None,
) -> dict:
    """执行（或断点续跑）一次发布，成功返回 active 发布记录（含完整 payload）。

    - 已 published 且 active：直接返回，不做任何重算；
    - running / failed：只处理 pending/failed 的分段作业，done 的复用落库结果；
    - 全部分段完成后单事务原子切换；中途异常标记 failed 并抛出，旧 active 不受影响。
    """
    worker = worker or DefaultWorker()
    series = dataset.series_for(device_id, metric)
    if series is None:
        raise KeyError(f"设备 {device_id} 没有指标 {metric}")

    pub = ensure_publication(dataset, device_id, metric, threshold_s, worker)
    pub_id = pub["pub_id"]

    # 已发布且是当前 active：幂等返回（重跑同一批结果不变）。
    if pub["status"] == store.PUB_PUBLISHED and pub["is_active"]:
        return pub

    jobs = store.list_segment_jobs(pub_id)
    stages: list[dict] = []
    started = time.perf_counter()

    def record(name: str, detail: str, begin: float) -> None:
        stages.append({"name": name, "detail": detail, "ms": round((time.perf_counter() - begin) * 1000, 2)})

    try:
        t0 = time.perf_counter()
        record("数据导入", f"{len(series.points)} 点 / 采样率 {series.interval_s}s", t0)

        t0 = time.perf_counter()
        # 漂移是整序列级作业，先于分段作业（与 6 步流水线顺序一致）。
        drift_task = DriftTask(
            pub_id=pub_id,
            dataset_id=dataset.dataset_id,
            device_id=device_id,
            metric=metric,
            threshold_s=threshold_s,
        )
        drift_payload = worker.compute_drift(dataset, series, drift_task)
        record("漂移估计", "整序列级作业", t0)

        t0 = time.perf_counter()
        for job in jobs:
            if job["status"] == store.JOB_DONE:
                continue  # 已完成分段不重复重算
            task = SegmentTask(
                pub_id=pub_id,
                dataset_id=dataset.dataset_id,
                device_id=device_id,
                metric=metric,
                threshold_s=threshold_s,
                segment_index=job["segment_index"],
                interval_s=job["interval_s"],
            )
            result = worker.compute_segment(dataset, series, task)
            store.mark_segment_done(pub_id, job["segment_index"], result)
        done_jobs = [j for j in store.list_segment_jobs(pub_id) if j["status"] == store.JOB_DONE]
        record(
            "网格重采样",
            f"{len(done_jobs)} 个档位分段作业（断点续跑复用已完成分段）",
            t0,
        )

        t0 = time.perf_counter()
        payload = _assemble(
            dataset, series, threshold_s, pub_id, drift_payload["drift"],
            stages=stages, total_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        record(
            "缺口判定",
            f"缺口 {payload['gaps']['count']} 处，插补 {payload['gaps']['interpolated_cells']} 格，"
            f"未修复 {payload['gaps']['unrepaired_cells']} 格（分段边界重算）",
            t0,
        )
        record("小时摘要", f"{len(payload['hourly'])} 个小时桶 / {len(payload['segments'])} 个档位分段", t0)
        t0 = time.perf_counter()
        record("质量评分", f"覆盖率 {payload['coverage']['coverage']:.4f} → {payload['coverage']['grade']} 级", t0)
        # 组装阶段追加的两条记录需要回写到已构建 payload。
        payload["stages"] = stages
        payload["total_ms"] = round((time.perf_counter() - started) * 1000, 2)
        published = store.publish_atomically(pub_id, payload)
        return published
    except Exception as exc:  # noqa: BLE001 - 失败落库后原样抛出，由路由转 4xx/5xx
        store.mark_publication_failed(pub_id, str(exc))
        raise PublicationError(f"发布 {pub_id} 失败：{exc}") from exc


def _assemble(
    dataset: Dataset,
    series: Series,
    threshold_s: float,
    pub_id: str,
    drift_result: dict,
    stages: list[dict] | None = None,
    total_ms: float = 0.0,
) -> dict:
    """从已落库的分段作业结果组装完整 payload（不重新跑网格计算）。"""
    from . import grid as grid_mod
    from . import quality as quality_mod
    from . import summary as summary_mod

    stages = list(stages or [])
    jobs = store.list_segment_jobs(pub_id)
    done = [j for j in jobs if j["status"] == store.JOB_DONE]
    if len(done) != len(jobs):
        raise PublicationError("分段作业未全部完成，无法组装发布 payload")

    segment_plans: list[dict] = []
    cells: list[dict] = []
    gap_items: list[dict] = []
    for job in sorted(done, key=lambda j: j["segment_index"]):
        result = json.loads(job["result_json"])
        segment_plans.append(result["segment"])
        cells.extend(result["cells"])
        gap_items.extend(result["gaps"])
    cells.sort(key=lambda c: c["bucket"])
    cell_by_bucket = {c["bucket"]: c for c in cells}

    # 重放分段边界 hold（与 gaps._apply_boundary_hold 同规则）：跨档位、间距 ≤ hold 上限
    # 的段尾空桶沿用前一观测值，并打 cross_segment；长断采保持 empty 留给缺口登记。
    n = len(cells)
    i = 0
    while i < n:
        if cells[i]["source"] != "empty":
            i += 1
            continue
        start = i
        while i < n and cells[i]["source"] == "empty":
            i += 1
        end = i - 1
        prev_idx, next_idx = start - 1, i if i < n else None
        if prev_idx < 0 or next_idx is None:
            continue
        prev_cell, next_cell = cells[prev_idx], cells[next_idx]
        if prev_cell.get("segment_index") == next_cell.get("segment_index"):
            continue
        if prev_cell["source"] not in ("agg", "sample") or next_cell["source"] not in ("agg", "sample"):
            continue
        if next_cell["bucket"] - prev_cell["bucket"] > settings.HOLD_MAX_BRACKET_BUCKETS:
            continue
        for j in range(start, end + 1):
            cells[j].update(
                value=prev_cell["value"], mean=prev_cell["value"],
                max=prev_cell["value"], min=prev_cell["value"],
                source="hold", carried=True, filled=True, cross_segment=True,
            )

    # 重放确定性插补：只依赖 gap 项里记录的前后锚点桶与值，不重新跑缺口判定。
    for item in gap_items:
        if item["kind"] != "interpolated":
            continue
        prev_cell = cell_by_bucket.get(item["start_bucket"] - 1)
        next_cell = cell_by_bucket.get(item["end_bucket"] + 1)
        if prev_cell is None or next_cell is None:
            continue
        v0 = prev_cell["value"]
        v1 = next_cell["value"]
        span = next_cell["bucket"] - prev_cell["bucket"]
        for offset, bucket in enumerate(
            range(item["start_bucket"], item["end_bucket"] + 1), start=1
        ):
            value = v0 + (v1 - v0) * offset / span
            cell = cell_by_bucket[bucket]
            cell.update(
                value=value, mean=value, max=value, min=value,
                source="interpolated", filled=True,
            )

    # 缺口插补会就地把部分 empty 改成 interpolated：在最终格子上重算统计，
    # 与内存流水线（先 find_gaps 再 stats）口径保持一致。
    grid_stats = grid_mod.stats(cells)
    interpolated = sum(
        g["cells"] for g in gap_items if g["kind"] == "interpolated"
    )
    unrepaired = sum(
        g["cells"] for g in gap_items if g["kind"] == "unrepaired"
    )
    boundary = sum(g["cells"] for g in gap_items if g["kind"] == "unrepaired" and g["boundary"])
    gap_result = {
        "items": sorted(gap_items, key=lambda g: g["start_bucket"]),
        "count": len(gap_items),
        "interpolated_cells": interpolated,
        "unrepaired_cells": unrepaired,
        "boundary_cells": boundary,
    }
    hourly = summary_mod.hourly_summary(cells)
    segment_stats = summary_mod.segment_summary(cells, segment_plans)
    expected = dataset.bucket_count()
    coverage = quality_mod.coverage_report(cells, expected, gap_result, segment_stats)

    ref_series, ref_device = pipeline.reference_for(dataset, series.metric, series.device_id)
    self_reference = ref_series is not None and ref_series.device_id == series.device_id
    start_bucket, end_bucket = pipeline.bucket_window(dataset)
    return {
        "dataset_id": dataset.dataset_id,
        "device_id": series.device_id,
        "metric": series.metric,
        "metric_label": settings.METRIC_LABEL.get(series.metric, series.metric),
        "metric_unit": settings.METRIC_UNIT.get(series.metric, ""),
        "interval_s": series.interval_s,
        "threshold_s": threshold_s,
        "window": {
            "start": grid_mod.bucket_start_iso(start_bucket),
            "end": grid_mod.bucket_start_iso(end_bucket),
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
        "reference": {
            "device_id": ref_device,
            "interval_s": ref_series.interval_s if ref_series else None,
            "self_reference": self_reference,
        },
        "stages": [],
        "total_ms": 0.0,
        "publication": {
            "pub_id": pub_id,
            "status": store.PUB_PUBLISHED,
            "segment_jobs": [
                {
                    "segment_index": j["segment_index"],
                    "interval_s": j["interval_s"],
                    "status": j["status"],
                    "attempts": j["attempts"],
                }
                for j in jobs
            ],
        },
    }
