"""分段发布器：断点续跑、已完成分段不重算、原子切换、失败保留旧版、重跑幂等。"""
from __future__ import annotations

import pytest

from app import pipeline, publication, quality, seed, store
from app.publication import DefaultWorker, SegmentTask


@pytest.fixture(scope="module")
def dataset():
    return seed.ensure_seed()["dataset"]


@pytest.fixture(autouse=True)
def _clean_publications():
    # 每个用例独立的发布状态：直接清表（conftest 已把 DB 指向临时目录）。
    store.init_db()
    with store._connect() as conn:
        conn.execute("DELETE FROM segment_jobs")
        conn.execute("DELETE FROM publications")
    yield


class FailNthSegmentWorker(DefaultWorker):
    """第 fail_index 个分段首次计算时抛错，用于模拟进程中断；之后恢复正常。"""

    def __init__(self, fail_index: int):
        self.fail_index = fail_index
        self.calls: list[int] = []
        self.failed_once = False

    def compute_segment(self, dataset, series, task: SegmentTask) -> dict:
        self.calls.append(task.segment_index)
        if not self.failed_once and task.segment_index == self.fail_index:
            self.failed_once = True
            raise RuntimeError(f"模拟分段 {task.segment_index} 计算中断")
        return super().compute_segment(dataset, series, task)


def _multi_segment_target(dataset):
    """挑一个档位分段数 >=2 的 (设备, 指标)。"""
    sb, eb = pipeline.bucket_window(dataset)
    for (dev, met), series in sorted(dataset.series.items()):
        plans = pipeline.grid.segment_plans(series.points, series.interval_s, sb, eb)
        if len(plans) >= 2:
            return dev, met, len(plans)
    pytest.skip("样例中没有多分段序列")


def test_full_publish_matches_in_memory_pipeline(dataset):
    dev, met, _ = _multi_segment_target(dataset)
    pub = publication.run_publication(dataset, dev, met, 5.0)
    assert pub["status"] == store.PUB_PUBLISHED
    assert pub["is_active"] is True
    memory = pipeline.run_alignment(dataset, dev, met, 5.0)
    payload = pub["payload"]
    for key in ("gaps", "coverage", "hourly", "segments", "grid", "segment_stats", "drift"):
        assert payload[key] == memory[key]
    # 所有作业都是 done，且记录了分段数
    jobs = store.list_segment_jobs(pub["pub_id"])
    assert jobs and all(j["status"] == store.JOB_DONE for j in jobs)
    assert len(jobs) == pub["segment_count"]


def test_resume_skips_completed_segments_and_publishes(dataset):
    dev, met, seg_count = _multi_segment_target(dataset)
    worker = FailNthSegmentWorker(fail_index=1)

    with pytest.raises(publication.PublicationError):
        publication.run_publication(dataset, dev, met, 5.0, worker=worker)

    pub_id = publication.ensure_publication(dataset, dev, met, 5.0)["pub_id"]
    failed_pub = store.get_publication(pub_id)
    assert failed_pub["status"] == store.PUB_FAILED
    jobs = store.list_segment_jobs(pub_id)
    done = {j["segment_index"] for j in jobs if j["status"] == store.JOB_DONE}
    assert 1 not in done and len(done) >= 1  # 失败分段之前的作业已落库

    # 续跑：新 worker 不再失败；已完成分段绝不再调用 compute_segment
    resume_worker = FailNthSegmentWorker(fail_index=-999)
    pub = publication.run_publication(dataset, dev, met, 5.0, worker=resume_worker)
    assert pub["status"] == store.PUB_PUBLISHED and pub["is_active"]
    assert set(resume_worker.calls).isdisjoint(done)
    assert set(resume_worker.calls) | done == set(range(seg_count))


def test_rerun_same_batch_returns_same_pub_id_and_results(dataset):
    dev, met, _ = _multi_segment_target(dataset)
    first = publication.run_publication(dataset, dev, met, 5.0)
    second = publication.run_publication(dataset, dev, met, 5.0)
    assert first["pub_id"] == second["pub_id"]
    assert first["payload"] == second["payload"]
    # 第二次没有任何作业被重新执行（全是 done，直接复用）
    jobs = store.list_segment_jobs(first["pub_id"])
    assert all(j["attempts"] == 1 for j in jobs)


def test_failed_run_keeps_previously_active_version(dataset):
    dev, met, _ = _multi_segment_target(dataset)
    old = publication.run_publication(dataset, dev, met, 5.0)
    assert old["is_active"] is True

    # 用不同阈值构造一个新版本，并让它在分段阶段失败 → 旧版保持 active
    worker = FailNthSegmentWorker(fail_index=0)
    with pytest.raises(publication.PublicationError):
        publication.run_publication(dataset, dev, met, 0.005, worker=worker)

    old_reloaded = store.get_publication(old["pub_id"])
    assert old_reloaded["is_active"] is True
    assert old_reloaded["status"] == store.PUB_PUBLISHED
    # 失败版本存在且不是 active
    pubs = [p for p in store.list_publications(dataset.dataset_id) if p["device_id"] == dev and p["metric"] == met]
    failed = [p for p in pubs if p["status"] == store.PUB_FAILED]
    assert failed and all(p["is_active"] is False for p in failed)


def test_successful_new_version_atomically_swaps_active_pointer(dataset):
    dev, met, _ = _multi_segment_target(dataset)
    old = publication.run_publication(dataset, dev, met, 5.0)
    new = publication.run_publication(dataset, dev, met, 0.005)
    assert new["pub_id"] != old["pub_id"]
    assert new["is_active"] is True
    assert store.get_publication(old["pub_id"])["is_active"] is False
    active = store.get_active_publication(dataset.dataset_id, dev, met)
    assert active["pub_id"] == new["pub_id"]


def test_publication_payload_contains_segment_and_publish_metadata(dataset):
    dev, met, _ = _multi_segment_target(dataset)
    pub = publication.run_publication(dataset, dev, met, 5.0)
    payload = pub["payload"]
    assert payload["publication"]["pub_id"] == pub["pub_id"]
    assert len(payload["publication"]["segment_jobs"]) == pub["segment_count"]
    assert payload["coverage"]["segment_count"] == pub["segment_count"]
    assert payload["coverage"]["segment_coverage"]
    weakest = payload["coverage"]["weakest_segment"]
    assert weakest["observability"]


def test_report_csv_contains_segment_and_publication_columns(dataset):
    rows = []
    for (dev, met), series in sorted(dataset.series.items()):
        publication.run_publication(dataset, dev, met, 5.0)
        payload = pipeline.run_alignment(dataset, dev, met, 5.0)
        pub = store.get_active_publication(dataset.dataset_id, dev, met)
        rows.append(
            {
                "device_id": dev,
                "metric": met,
                "interval_s": series.interval_s,
                "coverage": payload["coverage"],
                "drift": payload["drift"],
                "gaps": payload["gaps"],
                "publish_status": pub["status"],
            }
        )
    csv_text = quality.report_csv(rows)
    header = csv_text.splitlines()[0]
    for col in ("segment_count", "segment_intervals", "publish_status", "cross_segment_cells"):
        assert col in header
    seg_csv = quality.segment_csv(rows)
    assert "interval_s" in seg_csv.splitlines()[0] and "observability" in seg_csv.splitlines()[0]
    # 每个分段一行
    expected_seg_rows = sum(r["coverage"]["segment_count"] for r in rows)
    assert len(seg_csv.strip().splitlines()) == expected_seg_rows + 1
