import sys
from types import SimpleNamespace

from scripts.evaluate_marlin_mces import publish_clearml_mces


def test_publish_clearml_mces_uses_paper_parity_namespace(monkeypatch):
    scalar_calls = []

    class FakeLogger:
        def report_scalar(self, **kwargs):
            scalar_calls.append(kwargs)

    class FakeTask:
        @staticmethod
        def get_task(*, task_id):
            assert task_id == "evaluation-task"
            return SimpleNamespace(get_logger=lambda: FakeLogger())

    monkeypatch.setitem(sys.modules, "clearml", SimpleNamespace(Task=FakeTask))
    publish_clearml_mces(
        {"rows": 64, "mces_top1": 12.5, "mces_top10": 8.0},
        task_id="evaluation-task",
        iteration=30000,
    )

    assert [(call["title"], call["series"], call["iteration"]) for call in scalar_calls] == [
        ("Paper parity", "MCES@1 (returned)", 30000),
        ("Paper parity", "MCES@10 (returned)", 30000),
    ]
