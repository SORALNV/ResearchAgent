from __future__ import annotations


def _app(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_AGENT_CORE_TOKEN", "core-test-token")
    monkeypatch.setenv("RESEARCH_AGENT_DATA_DIR", str(tmp_path / "runtime"))
    monkeypatch.setenv("RESEARCH_AGENT_DATABASE", "platform.sqlite3")
    monkeypatch.setenv("CODEX_COMMAND", "definitely-not-installed-codex")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)

    from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
    import harness.platform.api as api_module
    from harness.platform.api_extensions import register_portable_routes
    from harness.platform.application import build_application

    for name, value in {
        "Depends": Depends,
        "FastAPI": FastAPI,
        "Header": Header,
        "HTTPException": HTTPException,
        "WebSocket": WebSocket,
        "WebSocketDisconnect": WebSocketDisconnect,
    }.items():
        setattr(api_module, name, value)
    service = build_application(tmp_path)
    app = api_module.create_app(service)
    register_portable_routes(app, service)
    return app


def test_core_api_requires_token_and_initializes_kaggle_domain(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    app = _app(tmp_path, monkeypatch)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/projects").status_code == 401
        headers = {"Authorization": "Bearer core-test-token"}
        response = client.post(
            "/v1/projects",
            headers=headers,
            json={
                "domain": "kaggle",
                "title": "Example Kaggle",
                "description": "test",
                "metadata": {
                    "competition_url": "https://www.kaggle.com/competitions/example-slug"
                },
            },
        )
        assert response.status_code == 200, response.text
        project = response.json()
        project_id = project["project_id"]
        assert project["metadata"]["competition_slug"] == "example-slug"

        status = client.get(
            f"/v1/kaggle/projects/{project_id}", headers=headers
        )
        assert status.status_code == 200, status.text
        assert status.json()["competition"]["slug"] == "example-slug"

        rules = client.post(
            f"/v1/kaggle/projects/{project_id}/rules/acknowledge",
            headers=headers,
            json={"rules_text": "# Rules\nNo external data.", "actor": "sora"},
        )
        assert rules.status_code == 200
        assert rules.json()["rules_acknowledged"] is True
        assert len(rules.json()["rules_hash"]) == 64

        cv = client.post(
            f"/v1/kaggle/projects/{project_id}/cv",
            headers=headers,
            json={
                "strategy": "StratifiedKFold",
                "n_splits": 5,
                "metric": "roc_auc",
                "seed": 42,
                "shuffle": True,
                "group_column": None,
                "time_column": None,
                "stratify_column": "target",
            },
        )
        assert cv.status_code == 200, cv.text
        cv_id = cv.json()["cv_spec_id"]
        locked = client.post(
            f"/v1/kaggle/projects/{project_id}/cv/{cv_id}/lock",
            headers=headers,
            json={"actor": "sora"},
        )
        assert locked.status_code == 200
        assert locked.json()["locked"] is True

        session = client.post(
            "/v1/work-sessions",
            headers=headers,
            json={
                "project_id": project_id,
                "title": "Baseline",
                "objective": "Create minimal baseline and CV",
                "metadata": {},
            },
        )
        assert session.status_code == 200
        session_id = session.json()["session_id"]
        status = client.get(
            f"/v1/work-sessions/{session_id}", headers=headers
        )
        assert status.status_code == 200
        assert status.json()["work_session"]["project_id"] == project_id
