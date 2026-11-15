"""缺口判定与修复（按连续采样档位分段重算）。

判定：网格内没有原始点（source='empty'），且同分段内前后都有原始点 → gap。
修复：gap 长度 <= 3 格 → 线性插补（interpolated）；> 3 格 → unrepaired。
边界：
- 序列首尾空缺不外推——开头/结尾的空桶只登记为 unrepaired(boundary)，绝不填值；
- **插补绝不跨分段**：每个分段内部各自找前后锚点，分段交界处的空桶按所在分段的
  边界空缺处理（boundary=True），不会拿另一个档位的值插出「看起来连续」的曲线。
- hold 沿用（含跨分段 hold）已在 grid 阶段处理，不算缺口。
"""
from __future__ import annotations

from . import settings


def find_gaps(cells: list[dict]) -> dict:
    """在（已 hold、已带分段标记的）网格上按分段找出缺口并给出修复结论。"""
    _apply_boundary_hold(cells)
    gaps: list[dict] = []

    # 按分段分组（cells 时间有序，segment_index 相同的桶连续出现）。
    groups: dict[int | None, list[int]] = {}
    order: list[int | None] = []
    for i, cell in enumerate(cells):
        key = cell.get("segment_index")
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(i)

    if not groups:
        return _summary(gaps)

    for key in order:
        idxs = groups[key]
        raw_local = [i for i in idxs if cells[i]["source"] in ("agg", "sample")]

        if not raw_local:
            # 整段没有任何原始点：整段登记为未修复（可能是边界段或长断采覆盖整段）。
            gaps.append(
                _gap_item(
                    cells,
                    start_idx=idxs[0],
                    end_idx=idxs[-1],
                    kind="unrepaired",
                    reason="该采样档位分段内没有任何原始点，不外推填值",
                    prev_idx=None,
                    next_idx=None,
                )
            )
            continue

        # 段首空缺：相对该分段不外推
        first = raw_local[0]
        if first > idxs[0]:
            gaps.append(
                _gap_item(
                    cells,
                    start_idx=idxs[0],
                    end_idx=first - 1,
                    kind="unrepaired",
                    reason="分段开头空缺，首端不外推",
                    prev_idx=None,
                    next_idx=first,
                )
            )

        for prev, nxt in zip(raw_local, raw_local[1:]):
            run_len = nxt - prev - 1
            if run_len == 0:
                continue
            bracket = cells[nxt]["bucket"] - cells[prev]["bucket"]
            if bracket <= settings.HOLD_MAX_BRACKET_BUCKETS:
                continue  # 已由 hold 沿用（carried），不算缺口
            kind = "interpolated" if run_len <= settings.INTERPOLATE_MAX_CELLS else "unrepaired"
            reason = (
                f"连续 {run_len} 格无原始点（前后原始格相距 {bracket} 分钟），"
                + ("允许线性插补（≤3 格）" if kind == "interpolated" else "超过 3 格，标记未修复")
            )
            gaps.append(
                _gap_item(
                    cells,
                    start_idx=prev + 1,
                    end_idx=nxt - 1,
                    kind=kind,
                    reason=reason,
                    prev_idx=prev,
                    next_idx=nxt,
                )
            )

        # 段尾空缺：相对该分段不外推（跨档 hold 已在 _apply_boundary_hold 处理，
        # 能沿用的空桶此刻不再是 empty；剩下的才是真正的段尾未修复空缺）。
        last = raw_local[-1]
        if last < idxs[-1] and cells[idxs[-1]]["source"] == "empty":
            tail_start = last + 1
            while tail_start <= idxs[-1] and cells[tail_start]["source"] != "empty":
                tail_start += 1
            if tail_start <= idxs[-1]:
                gaps.append(
                    _gap_item(
                        cells,
                        start_idx=tail_start,
                        end_idx=idxs[-1],
                        kind="unrepaired",
                        reason="分段结尾空缺，尾端不外推",
                        prev_idx=last,
                        next_idx=None,
                    )
                )

    # 同桶不会被两个分段处理（桶归属唯一）；按起始桶排序保证输出稳定。
    gaps.sort(key=lambda g: g["start_bucket"])
    return _summary(gaps)


def _apply_boundary_hold(cells: list[dict]) -> None:
    """分段边界的弱补值：段尾空桶若能被「本段最后一个原始点」与「下一段第一个原始点」
    在 HOLD_MAX_BRACKET_BUCKETS 格内夹住，则沿用本段最后观测值（hold），并显式标记
    cross_segment=True。

    这与 grid._apply_hold 的区别：那里只处理同一对原始锚点之间的空桶，而分段交界处的
    空桶在分段内只有单侧锚点，需要这里结合相邻分段判定。仍然**绝不插值跨段**——
    只做「沿用最后观测值」，且间距超过 hold 上限的长断采保持 empty 交给缺口登记。
    """
    n = len(cells)
    i = 0
    while i < n:
        if cells[i]["source"] != "empty":
            i += 1
            continue
        # 找到一段连续 empty 的起点/终点
        start = i
        while i < n and cells[i]["source"] == "empty":
            i += 1
        end = i - 1
        prev_idx = start - 1
        next_idx = i if i < n else None
        if prev_idx < 0 or next_idx is None:
            continue  # 序列首尾：不外推
        prev_cell = cells[prev_idx]
        next_cell = cells[next_idx]
        if prev_cell.get("segment_index") == next_cell.get("segment_index"):
            continue  # 同段内的空桶已由 grid 的 hold/后续缺口逻辑覆盖
        if prev_cell["source"] not in ("agg", "sample") or next_cell["source"] not in ("agg", "sample"):
            continue
        bracket = next_cell["bucket"] - prev_cell["bucket"]
        if bracket > settings.HOLD_MAX_BRACKET_BUCKETS:
            continue  # 长断采：不沿用，登记为段尾未修复
        hold_value = prev_cell["value"]
        for j in range(start, end + 1):
            cells[j]["value"] = hold_value
            cells[j]["mean"] = hold_value
            cells[j]["max"] = hold_value
            cells[j]["min"] = hold_value
            cells[j]["source"] = "hold"
            cells[j]["carried"] = True
            cells[j]["filled"] = True
            cells[j]["cross_segment"] = True


def _summary(gaps: list[dict]) -> dict:
    interpolated = 0
    unrepaired = 0
    boundary = 0
    for item in gaps:
        if item["kind"] == "interpolated":
            interpolated += item["cells"]
        else:
            unrepaired += item["cells"]
            if item["boundary"]:
                boundary += item["cells"]
    return {
        "items": gaps,
        "count": len(gaps),
        "interpolated_cells": interpolated,
        "unrepaired_cells": unrepaired,
        "boundary_cells": boundary,
    }


def _gap_item(
    cells: list[dict],
    start_idx: int,
    end_idx: int,
    kind: str,
    reason: str,
    prev_idx: int | None,
    next_idx: int | None,
) -> dict:
    run = cells[start_idx : end_idx + 1]
    length = len(run)
    boundary = prev_idx is None or next_idx is None
    filled: list[float] = []
    if kind == "interpolated" and prev_idx is not None and next_idx is not None:
        v0 = cells[prev_idx]["value"]
        v1 = cells[next_idx]["value"]
        span = cells[next_idx]["bucket"] - cells[prev_idx]["bucket"]
        for offset, cell in enumerate(run, start=1):
            value = v0 + (v1 - v0) * offset / span
            cell["value"] = value
            cell["mean"] = value
            cell["max"] = value
            cell["min"] = value
            cell["source"] = "interpolated"
            cell["filled"] = True
            filled.append(value)
    return {
        "start_bucket": run[0]["bucket"],
        "start_time": run[0]["bucket_start"],
        "end_bucket": run[-1]["bucket"],
        "end_time": run[-1]["bucket_start"],
        "cells": length,
        "kind": kind,
        "reason": reason,
        "boundary": boundary,
        "segment_index": run[0].get("segment_index"),
        "interval_s": run[0].get("interval_s"),
        "prev_time": cells[prev_idx]["bucket_start"] if prev_idx is not None else None,
        "next_time": cells[next_idx]["bucket_start"] if next_idx is not None else None,
        "filled_values": filled,
    }


def apply_repair(cells: list[dict]) -> list[dict]:
    """返回补齐后的网格（find_gaps 已就地写入插补值，这里做一次显式返回便于测试）。"""
    return cells
