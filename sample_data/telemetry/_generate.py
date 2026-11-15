#!/usr/bin/env python3
"""生成产线 A 三台设备 6 小时遥测样例（确定性，无随机种子漂移）。

场景事实（与 README / TESTING.md 一致）：
- LINE-A-01：1s 采样，温度（唯一全程 1s 高频的设备）
- LINE-A-02：5s 采样，温度 + 压力；02:10–02:55 采集器降级为 180s → 1 分钟网格需 hold（carried）
- LINE-A-03：60s 采样，温度 + 压力；设备时钟偏移 +37s、走时偏快 55ppm；
  05:00–05:50 归档器降级为 120s（carried）；压力通道 01:40–02:05 断采
- LINE-A-04：一条温度序列在 1s/5s/60s 三档间切换（TIER_SCHEDULE，共 5 个连续档位分段），
  时钟正常；段内插入短暂补传突发/孤立稀疏点（TIER_BURSTS），验证这些异常不会切段
- 全线两处缺采（LINE-A-01/02/03 的所有指标）：03:00–03:02（3 格）、04:10–04:14（5 格）
- LINE-A-02 温度 05:10–05:21 断采（12 格）
- LINE-A-03 压力 01:40–02:05 断采（25 格）

数值形态为三角波（分段线性），便于用阈值穿越点做时钟漂移配准；
噪声幅度远小于斜率，不影响穿越时刻的线性插值精度。
"""
from __future__ import annotations

import csv
import math
import pathlib
from datetime import datetime, timedelta, timezone

TZ = timezone(timedelta(hours=8))
DAY = datetime(2026, 5, 12, 0, 0, 0, tzinfo=TZ)
SPAN_S = 6 * 3600

DEVICES = {
    "LINE-A-01": {"interval_s": 1, "metrics": ["temperature"], "offset_s": 0.0, "rate_ppm": 0.0},
    "LINE-A-02": {"interval_s": 5, "metrics": ["temperature", "pressure"], "offset_s": 0.0, "rate_ppm": 0.0},
    "LINE-A-03": {"interval_s": 60, "metrics": ["temperature", "pressure"], "offset_s": 37.0, "rate_ppm": 55.0},
    # LINE-A-04：同一条温度序列在 1s/5s/60s 三个档位间来回切换（见 TIER_SCHEDULE），
    # 用来验证「按连续采样档位分段」而不是整段中位数；时钟正常，只上报温度。
    "LINE-A-04": {"interval_s": 1, "metrics": ["temperature"], "offset_s": 0.0, "rate_ppm": 0.0},
}

# LINE-A-04 的档位时间表：(起始相对秒, 结束相对秒, 该档 cadence 秒)。
# 每段都足够长（≥3 点且跨度 ≥60s），段内还刻意插入秒级补传突发/孤立稀疏点，
# 验证这些短暂异常不会被切成独立档位（见 tests/test_segments.py）。
TIER_SCHEDULE = {
    "LINE-A-04": [
        (0 * 3600, 1 * 3600 + 30 * 60, 1),       # 00:00–01:30 高频 1s
        (1 * 3600 + 30 * 60, 3 * 3600, 5),       # 01:30–03:00 高频 5s
        (3 * 3600, 4 * 3600 + 30 * 60, 60),      # 03:00–04:30 低频 60s
        (4 * 3600 + 30 * 60, 5 * 3600, 5),       # 04:30–05:00 回到 5s
        (5 * 3600, 6 * 3600, 1),                 # 05:00–06:00 回到 1s
    ],
}

# LINE-A-04 档位段内的短暂异常（不应切段）：(时刻, 连续补传点数, 补传间隔秒)
TIER_BURSTS = {
    "LINE-A-04": [
        (2 * 3600 + 10 * 60, 2, 1),    # 5s 段中的秒级补传突发
        (3 * 3600 + 20 * 60, 1, 1),    # 60s 段中的单个秒级补传点
        (5 * 3600 + 25 * 60, 1, 60),   # 1s 段中的孤立稀疏点
    ],
}

# 指标形态：三角波 + 极小噪声（噪声 < 0.05% 幅值）
METRIC_SHAPE = {
    "temperature": {"base": 21.0, "amp": 4.0, "period_s": 8 * 60, "noise": 0.002, "decimals": 3},
    "pressure": {"base": 3.20, "amp": 0.40, "period_s": 6 * 60, "noise": 0.0008, "decimals": 4},
}

# 缺采窗口（线性分钟窗口，[start, end)），按 (device, metric) 或 "ALL"
OUTAGES = [
    ("ALL", "ALL", 3 * 3600, 3 * 3600 + 3 * 60),
    ("ALL", "ALL", 4 * 3600 + 10 * 60, 4 * 3600 + 15 * 60),
    ("LINE-A-02", "temperature", 5 * 3600 + 10 * 60, 5 * 3600 + 22 * 60),
    ("LINE-A-03", "pressure", 3600 + 40 * 60, 2 * 3600 + 5 * 60),
]

# 降级窗口（采集器把 cadence 调大，需要 1 分钟网格 hold）
DEGRADED = [
    ("LINE-A-02", "temperature", 2 * 3600 + 10 * 60, 2 * 3600 + 55 * 60, 180),
    ("LINE-A-03", "temperature", 5 * 3600, 5 * 3600 + 50 * 60, 120),
    ("LINE-A-03", "pressure", 5 * 3600, 5 * 3600 + 50 * 60, 120),
]


def triangle(t: float, base: float, amp: float, period: float) -> float:
    phase = (t % period) / period
    return base + amp * (4 * abs(phase - 0.5) - 1)


def noise(t: float, amp: float, salt: int) -> float:
    """确定性伪噪声：正弦叠加，无随机数依赖。"""
    return amp * (math.sin(t * 0.71 + salt) * 0.6 + math.sin(t * 1.93 + salt * 2.3) * 0.4)


def in_window(t: float, start: float, end: float) -> bool:
    return start <= t < end


def outage_for(device: str, metric: str, t: float) -> bool:
    for dev, met, start, end in OUTAGES:
        if dev not in ("ALL", device):
            continue
        if met not in ("ALL", metric):
            continue
        if in_window(t, start, end):
            return True
    return False


def cadence_for(device: str, metric: str, t: float, default: int) -> int:
    if device in TIER_SCHEDULE:
        for start, end, cadence in TIER_SCHEDULE[device]:
            if in_window(t, start, end):
                return cadence
        return default
    for dev, met, start, end, cadence in DEGRADED:
        if dev == device and met == metric and in_window(t, start, end):
            return cadence
    return default


def burst_points_for(device: str) -> set[float]:
    """档位段内的短暂补传/稀疏异常时刻（落在常规 cadence 网格之外的额外点）。"""
    out: set[float] = set()
    for anchor, count, step in TIER_BURSTS.get(device, []):
        for k in range(count):
            out.add(anchor + k * step + 0.3)  # 偏移 0.3s，确保不与常规点重合
    return out


def build_rows() -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for device, spec in DEVICES.items():
        offset = spec["offset_s"]
        drift = spec["rate_ppm"] / 1_000_000.0
        bursts = burst_points_for(device)
        for metric in spec["metrics"]:
            shape = METRIC_SHAPE[metric]
            salt = 1 if metric == "temperature" else 7
            t = 0
            while t < SPAN_S:
                cadence = cadence_for(device, metric, t, spec["interval_s"])
                if not outage_for(device, metric, t):
                    value = triangle(t, shape["base"], shape["amp"], shape["period_s"])
                    value += noise(t, shape["noise"], salt)
                    # 设备本地时钟 = 真实时刻 + 固定偏移 + 走时误差
                    local = t + offset + drift * t
                    ts = (DAY + timedelta(seconds=local)).isoformat(timespec="milliseconds")
                    rows.append((device, ts, metric, f"{value:.{shape['decimals']}f}"))
                t += cadence
            # 档位段内短暂异常点（补传/孤立点）：只对该设备存在的指标生成
            for bt in sorted(bursts):
                if bt >= SPAN_S or outage_for(device, metric, bt):
                    continue
                value = triangle(bt, shape["base"], shape["amp"], shape["period_s"])
                value += noise(bt, shape["noise"], salt)
                local = bt + offset + drift * bt
                ts = (DAY + timedelta(seconds=local)).isoformat(timespec="milliseconds")
                rows.append((device, ts, metric, f"{value:.{shape['decimals']}f}"))
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def main() -> None:
    here = pathlib.Path(__file__).resolve().parent
    files = {
        "line-a-01-1s.csv": "LINE-A-01",
        "line-a-02-5s.csv": "LINE-A-02",
        "line-a-03-60s.csv": "LINE-A-03",
        "line-a-04-tiered.csv": "LINE-A-04",
    }
    rows = build_rows()
    for name, device in files.items():
        subset = [r for r in rows if r[0] == device]
        with (here / name).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["device_id", "ts", "metric", "value"])
            writer.writerows(subset)
        print(f"{name}: {len(subset)} rows")


if __name__ == "__main__":
    main()
