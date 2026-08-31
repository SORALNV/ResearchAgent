from __future__ import annotations

import base64
import io
import json
import time
import zipfile

import pytest

from harness.compute.bundle import build_source_bundle, extract_source_bundle
from harness.compute.worker_api_portable import PortableWorkerJobManager
from harness.platform.models import Domain, JobRecord, JobSpec, ResourceRequest


def test_source_bundle_round_trip_and_path_traversal_rejection(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "run.py").write_text("print('ok')\n", encoding="utf-8")
    (source / "config.json").write_text('{"x": 1}\n', encoding="utf-8")
    (source / "data").mkdir()
    (source / "data" / "large.csv").write_text("not bundled", encoding="utf-8")
    bundle = build_source_bundle(source)
    destination = tmp_path / "destination"
    result = extract_source_bundle(bundle, destination)
    assert result["file_count"] == 2
    assert (destination / "run.py").read_text(encoding="utf-8") == "print('ok')\n"
    assert not (destination / "data").exists()

    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w") as archive:
        archive.writestr("../escape.txt", "bad")
    raw = memory.getvalue()
    malicious = {
        "encoding": "base64+zip",
        "sha256": __import__("hashlib").sha256(raw).hexdigest(),
        "data": base64.b64encode(raw).decode("ascii"),
    }
    with pytest.raises(ValueError):
        extract_source_bundle(malicious, tmp_path / "malicious")


def test_portable_worker_executes_bundle_without_worker_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_WORKER_TOKEN", "worker-secret")
    source = tmp_path / "job-source"
    source.mkdir()
    (source / "run.py").write_text(
        """from __future__ import annotations
import json
import os
from pathlib import Path
Path('progress.json').write_text(json.dumps({'stage':'done','progress':1.0})+'\\n', encoding='utf-8')
result = {'secret_visible': 'RESEARCH_WORKER_TOKEN' in os.environ, 'score': 0.81}
Path('result.json').write_text(json.dumps(result)+'\\n', encoding='utf-8')
Path('prediction.txt').write_text('prediction\\n', encoding='utf-8')
""",
        encoding="utf-8",
    )
    spec = JobSpec.new(
        work_session_id="WS-REMOTE",
        domain=Domain.RESEARCH,
        task_type="smoke_test",
        entrypoint="python run.py",
        resources=ResourceRequest(
            accelerator="cpu",
            cpu_cores=1,
            ram_gb=1,
            max_runtime_minutes=5,
            capabilities=("smoke_test",),
        ),
        outputs=("result.json", "prediction.txt"),
    )
    record = JobRecord(spec=spec)
    manager = PortableWorkerJobManager(data_dir=tmp_path / "worker")
    submitted = manager.submit(
        {
            "job": record.to_dict(),
            "source_bundle": build_source_bundle(source),
        }
    )
    assert submitted["status"] == "running"

    deadline = time.time() + 5
    status = submitted
    while time.time() < deadline and status["status"] == "running":
        status = manager.status(spec.job_id)
        time.sleep(0.02)
    assert status["status"] == "completed"
    collected = manager.collect(spec.job_id)
    assert collected["result"]["secret_visible"] is False
    paths = {item["path"] for item in collected["artifacts"]}
    assert {"result.json", "prediction.txt"} <= paths
    assert all(len(item["sha256"]) == 64 for item in collected["artifacts"])
