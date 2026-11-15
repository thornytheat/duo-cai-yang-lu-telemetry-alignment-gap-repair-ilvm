"""连续采样档位分段：识别一条序列在时间轴上切换过的采样档位（cadence tier）。

背景：旧实现用整段相邻间隔的**中位数**给整条序列定一个采样率，设备在 1s/5s/60s
上报之间切换时，切换前后的分钟格被按同一种方式解读，缺口、hold 沿用与小时摘要
对不上。本模块只做一件事——把一条序列切成若干「连续同档位」分段，后续网格、缺口、
摘要与质量报告均按分段口径重算。

鲁棒性（对应需求「短暂补传或个别稀疏点不能带偏整段档位」）：
- 档位按相邻采样间隔的固定带宽识别（1s/5s/60s/120s/180s 五档），带宽之外的间隔
  归入最接近的已知档，避免一个离群间隔凭空造出一个新档位；
- 先做「点级标签 + 游程」，再把短于 SEGMENT_MIN_SPAN_S 秒或短于
  SEGMENT_MIN_POINTS 个点的游程并入相邻档位（补传突发、个别稀疏点都不会切段）；
- 分段输出是纯函数、无随机量，同一份输入重跑结果完全一致（发布幂等的基础）。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from . import settings


def _iso(epoch: float) -> str:
    tz = timezone(timedelta(hours=8))
    return datetime.fromtimestamp(epoch, tz=tz).isoformat(timespec="seconds")

# 已知采样档位带宽（秒）：间隔落在 (lo, hi] 即归入该档。
# 档位顺序即「频率从高到低」，tier 序号越小越密。
CADENCE_BANDS: tuple[tuple[int, int, int], ...] = (
    # (档位间隔秒, 带宽下界(不含), 带宽上界(含))
    (1, 0, 2),
    (5, 2, 15),
    (60, 15, 90),
    (120, 90, 150),
    (180, 150, 240),
)
FALLBACK_BAND_INTERVAL = 60


@dataclass
class Segment:
    """一个连续同档位分段（点下标闭区间 [first_index, last_index]）。"""

    index: int
    interval_s: int            # 该档标称间隔（1/5/60/120/180）
    tier: str                  # high | low
    first_index: int
    last_index: int
    first_ts: float
    last_ts: float
    points: int
    start_bucket: int = 0      # 覆盖的分钟桶闭区间（由 assign_buckets 填充）
    end_bucket: int = 0
    absorbed_runs: int = 0     # 被平滑吸收的高频突发游程数（真正的秒级补传）
    absorbed_sparse_runs: int = 0  # 被平滑吸收的孤立稀疏游程数（断采边缘单点等）
    merged_into: int | None = None  # 未赢得任何桶时，并入党段的 index

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "interval_s": self.interval_s,
            "tier": self.tier,
            "first_ts": self.first_ts,
            "last_ts": self.last_ts,
            "first_time": _iso(self.first_ts),
            "last_time": _iso(self.last_ts),
            "points": self.points,
            "start_bucket": self.start_bucket,
            "end_bucket": self.end_bucket,
            "buckets": self.end_bucket - self.start_bucket + 1,
            "absorbed_runs": self.absorbed_runs,
            "absorbed_sparse_runs": self.absorbed_sparse_runs,
        }


def is_high_rate(interval_s: int) -> bool:
    """高频档：采样间隔小于 1 分钟桶（桶内可聚合）；否则为低频档（sample/hold 口径）。"""
    return interval_s < settings.BUCKET_SECONDS


def tier_for_interval(interval_s: int) -> str:
    return "high" if is_high_rate(interval_s) else "low"


def band_for_delta(delta: float) -> tuple[int, int, int]:
    """把一个相邻间隔（秒）归入已知档位带宽，返回 (interval, lo, hi)。"""
    for band in CADENCE_BANDS:
        interval, lo, hi = band
        if lo < delta <= hi:
            return band
    # 带宽之外（极端长间隔等）：归入最接近的标称档，不造新档位。
    nearest = min(CADENCE_BANDS, key=lambda b: abs(delta - b[0]))
    return nearest


def interval_for_delta(delta: float) -> int:
    return band_for_delta(delta)[0]


def _point_labels(points: list[tuple[float, float]], fallback_interval: int) -> list[int]:
    """每个点标注其「所在档位」：用它到**后一个点**的间隔归类（前向间隔）。

    为什么用前向而不是后向：短暂补传是「插进正常 cadence 网格里的密集点」，
    补传点相对前一正常点的到达间隔偏大，但它到下一个点（另一个补传点或回归的正常点）
    很密——前向间隔才能把补传突发标成密集档，进而被短游程平滑吸收而不切段。
    末点没有后继，沿用倒数第二个间隔（单点序列用 fallback）。
    """
    n = len(points)
    labels = [fallback_interval] * n
    if n < 2:
        return labels
    last_delta = points[-1][0] - points[-2][0]
    labels[-1] = interval_for_delta(last_delta) if last_delta > 0 else fallback_interval
    for i in range(n - 1):
        delta = points[i + 1][0] - points[i][0]
        labels[i] = interval_for_delta(delta) if delta > 0 else labels[i - 1] if i else fallback_interval
    return labels


def _runs(labels: list[int]) -> list[list[int]]:
    """游程编码：返回每个同标签游程的下标列表。"""
    out: list[list[int]] = []
    for i, label in enumerate(labels):
        if out and labels[out[-1][0]] == label:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def _span_seconds(points: list[tuple[float, float]], idxs: list[int]) -> float:
    if len(idxs) == 1:
        return 0.0
    return points[idxs[-1]][0] - points[idxs[0]][0]


def _is_short(points: list[tuple[float, float]], idxs: list[int], labels: list[int]) -> bool:
    """短游程：持续不足最小跨度，且点数不足最小点数。

    满足尺寸条件的游程无论高频/低频都并入邻段——一个真正的低频档位（如 60s cadence）
    至少需要 3 个点、跨越 ≥120s 才能体现「每分钟桶至多 1 个观测点」的可观测能力，
    1~2 个点、不足 60s 的低频游程只可能是补传尾巴/孤立稀疏点，不构成独立分段。
    持续很宽但点数稀少的游程（AND 的另一面）会保留，避免把真实长间隔缺采误并。
    """
    return (
        len(idxs) < settings.SEGMENT_MIN_POINTS
        and _span_seconds(points, idxs) <= settings.SEGMENT_MIN_SPAN_S
    )


def detect_segments(
    points: list[tuple[float, float]],
    fallback_interval: int = FALLBACK_BAND_INTERVAL,
) -> list[Segment]:
    """把排序后的点列切成连续同档位分段（已做短游程平滑）。

    返回的分段按时间有序、下标互不重叠且完整覆盖 points。
    """
    n = len(points)
    if n == 0:
        return []
    labels = _point_labels(points, fallback_interval)
    raw_runs = _runs(labels)
    # 被短游程平滑吸收的原始游程：分别记录「高频突发」与「孤立稀疏点」，
    # 同档相邻合并不计入；断采边缘的稀疏点与真正的秒级补传在页面上区别解释。
    smoothed_dense: set[int] = set()
    smoothed_sparse: set[int] = set()

    # 短游程吸收
    cur = [list(r) for r in raw_runs]
    changed = True
    while changed and len(cur) > 1:
        changed = False
        for pos, idxs in enumerate(cur):
            if not _is_short(points, idxs, labels):
                continue
            if is_high_rate(labels[idxs[0]]):
                smoothed_dense.add(idxs[0])
            else:
                smoothed_sparse.add(idxs[0])
            if pos > 0:
                cur[pos - 1].extend(idxs)
                del cur[pos]
            else:
                cur[1] = idxs + cur[1]
                del cur[0]
            changed = True
            break

    # 相邻同档合并
    compact: list[list[int]] = []
    for idxs in cur:
        if compact and labels[compact[-1][0]] == labels[idxs[0]]:
            compact[-1].extend(idxs)
        else:
            compact.append(list(idxs))

    segments: list[Segment] = []
    for k, idxs in enumerate(compact):
        idxs.sort()
        first_i, last_i = idxs[0], idxs[-1]
        interval = labels[first_i]
        # absorbed_runs = 落入本分段、且确实被短游程平滑吸收的原始游程数（0 表示没有异常突发）。
        absorbed_dense = sum(
            1 for r in raw_runs
            if r[0] in smoothed_dense and first_i <= r[0] <= last_i
        )
        absorbed_sparse = sum(
            1 for r in raw_runs
            if r[0] in smoothed_sparse and first_i <= r[0] <= last_i
        )
        seg = Segment(
            index=k,
            interval_s=interval,
            tier=tier_for_interval(interval),
            first_index=first_i,
            last_index=last_i,
            first_ts=points[first_i][0],
            last_ts=points[last_i][0],
            points=len(idxs),
            absorbed_runs=absorbed_dense,
            absorbed_sparse_runs=absorbed_sparse,
        )
        segments.append(seg)
    return segments


def assign_buckets(
    segments: list[Segment],
    points: list[tuple[float, float]],
    start_bucket: int,
    end_bucket: int,
) -> list[Segment]:
    """把分钟桶归属到分段：桶内多数点所属分段赢得该桶（平票给更早的分段）；
    没有任何点的桶（断采/边界空缺）并入**前一个**分段，保证分段连续覆盖整个窗口，
    且每个桶恰好属于一个分段——缺口在分段边界重算时有明确的责任段。
    """
    if not segments:
        return segments
    bucket = settings.BUCKET_SECONDS

    # 1) 有点的桶：分段按桶内点数投票，平票给更早（index 更小）的分段。
    counts: dict[int, dict[int, int]] = {}
    for seg in segments:
        for i in range(seg.first_index, seg.last_index + 1):
            b = int(points[i][0] // bucket)
            if b < start_bucket or b > end_bucket:
                continue
            slot = counts.setdefault(b, {})
            slot[seg.index] = slot.get(seg.index, 0) + 1
    bucket_owner: dict[int, int] = {
        b: min(slot.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        for b, slot in counts.items()
    }

    # 2) 空缺桶归前一个分段；开头空缺归第一个分段
    #    （首尾不外推的填值规则在 gaps 模块判定，这里只定归属）。
    last_owner = segments[0].index
    for b in range(start_bucket, end_bucket + 1):
        if b in bucket_owner:
            last_owner = bucket_owner[b]
        else:
            bucket_owner[b] = last_owner

    # 3) 由归属结果反推每段连续桶区间；没有赢得任何桶的分段（极短的低频游程）
    #    并入时间上相邻的分段，保证每个输出分段都拥有非空连续桶区间。
    owned: dict[int, list[int]] = {seg.index: [] for seg in segments}
    for b in range(start_bucket, end_bucket + 1):
        owned[bucket_owner[b]].append(b)
    for seg in segments:
        buckets = owned[seg.index]
        if buckets:
            seg.start_bucket = buckets[0]
            seg.end_bucket = buckets[-1]
            continue
        # 该分段没有任何桶：把它的桶挂到前一个（否则后一个）有效分段上。
        target = _nearest_owned(segments, owned, seg.index)
        if target is not None:
            seg.start_bucket = target.start_bucket
            seg.end_bucket = target.end_bucket
            seg.merged_into = target.index
    return segments


def _nearest_owned(segments: list[Segment], owned: dict[int, list[int]], index: int) -> Segment | None:
    for delta in (1, -1):
        j = index + delta
        while 0 <= j < len(segments):
            if owned[segments[j].index]:
                return segments[j]
            j += delta
    return None


def segment_dicts(segments: list[Segment]) -> list[dict]:
    return [s.to_dict() for s in segments]
