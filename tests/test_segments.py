"""连续采样档位分段：1s/5s/60s 切换识别、短补传/孤立稀疏点不带偏整段、桶归属。"""
from __future__ import annotations

from app import grid, segments, settings

BASE = 1_750_000_000.0 // 60 * 60
B0 = int(BASE // 60)


def tiered_points(schedule: list[tuple[int, int]]) -> list[tuple[float, float]]:
    """按 (持续秒数, 档位秒) 顺序生成点列。"""
    points: list[tuple[float, float]] = []
    t = 0.0
    for span, cadence in schedule:
        end = t + span
        while t < end:
            points.append((BASE + t, float(cadence)))
            t += cadence
    return points


def test_detects_three_cadence_tiers_in_order():
    # 2 分钟 1s → 2 分钟 5s → 3 分钟 60s（60s 段 3 个点、跨度 120s，满足最小段条件）
    points = tiered_points([(120, 1), (120, 5), (180, 60)])
    parts = segments.detect_segments(points, fallback_interval=60)
    assert [p.interval_s for p in parts] == [1, 5, 60]
    assert [p.tier for p in parts] == ["high", "high", "low"]
    assert parts[0].first_index == 0
    assert parts[-1].last_index == len(points) - 1
    # 下标完整覆盖、互不重叠
    covered = [i for p in parts for i in range(p.first_index, p.last_index + 1)]
    assert covered == list(range(len(points)))


def test_short_burst_retransmit_does_not_create_segment():
    # 5s 档位中间插入 2 个秒级补传点（短于 60s、少于 3 点）→ 不切段
    points = tiered_points([(180, 5)])
    points.insert(40, (BASE + 200.3, 1.0))
    points.insert(41, (BASE + 201.1, 1.0))
    parts = segments.detect_segments(points, fallback_interval=5)
    assert len(parts) == 1
    assert parts[0].interval_s == 5
    # 秒级补传突发被记为 dense 吸收（区别于断采边缘的孤立稀疏点）
    assert parts[0].absorbed_runs >= 1


def test_isolated_sparse_point_inside_high_rate_is_absorbed():
    # 1s 档位里夹一个 60s 间隔的孤立稀疏点，仍保持单一 1s 分段
    points = [(BASE + float(i), 1.0) for i in range(90)]
    points.insert(60, (BASE + 59.0 + 60.0, 1.0))  # 单点 60s 间隔
    parts = segments.detect_segments(points, fallback_interval=1)
    assert len(parts) == 1
    assert parts[0].interval_s == 1
    # 孤立稀疏点走 sparse 计数，不与秒级补传突发混淆
    assert parts[0].absorbed_sparse_runs >= 1
    assert parts[0].absorbed_runs == 0


def test_sustained_switch_is_not_smoothed_away():
    # 足够长的 60s 降级段（>60s 且 ≥3 点）必须被切成独立分段
    points = tiered_points([(120, 1), (180, 60), (120, 1)])
    parts = segments.detect_segments(points, fallback_interval=1)
    assert [p.interval_s for p in parts] == [1, 60, 1]
    low = parts[1]
    assert low.points >= 3
    assert low.last_ts - low.first_ts >= settings.SEGMENT_MIN_SPAN_S


def test_assign_buckets_covers_full_window_without_gaps():
    points = tiered_points([(120, 1), (180, 60), (120, 5)])
    parts = segments.detect_segments(points, fallback_interval=60)
    segments.assign_buckets(parts, points, B0, B0 + 6)
    owners = []
    for p in parts:
        owners.extend(range(p.start_bucket, p.end_bucket + 1))
    assert owners == list(range(B0, B0 + 7))  # 每桶恰好归属一个分段
    assert parts[0].start_bucket == B0


def test_empty_buckets_in_outage_are_owned_by_previous_segment():
    # 1s 段后断 4 分钟再来 60s：断采空桶归前一段，分段仍连续
    points = tiered_points([(120, 1)])
    points += [(BASE + 120 + 240 + float(i) * 60, 60.0) for i in range(4)]
    parts = segments.detect_segments(points, fallback_interval=1)
    segments.assign_buckets(parts, points, B0, B0 + 9)
    first = parts[0]
    # 断采的 4 个空桶（第 2..5 桶）并入第一段
    assert first.end_bucket >= B0 + 5


def test_resample_tags_every_cell_with_segment_and_interval():
    points = tiered_points([(120, 1), (180, 60), (120, 5)])
    cells = grid.resample(points, interval_s=1, start_bucket=B0, end_bucket=B0 + 6)
    for cell in cells:
        assert cell["segment_index"] is not None
        assert cell["interval_s"] in (1, 5, 60)
    # 高频档桶为 agg，低频档桶为 sample/hold/empty
    low_cells = [c for c in cells if c["interval_s"] == 60]
    assert all(c["source"] in ("sample", "hold", "empty") for c in low_cells)


def test_detect_segments_is_deterministic():
    points = tiered_points([(120, 1), (120, 5), (120, 60), (120, 5)])
    first = segments.detect_segments(points, fallback_interval=60)
    second = segments.detect_segments(points, fallback_interval=60)
    assert [p.to_dict() for p in first] == [p.to_dict() for p in second]


def test_empty_and_single_point_series():
    assert segments.detect_segments([]) == []
    one = segments.detect_segments([(BASE, 1.0)], fallback_interval=5)
    assert len(one) == 1 and one[0].interval_s == 5
