"""Audit logging + secret redaction tests (Phase 2). No real secrets used."""

import json

from app.security.audit import AuditLogger, sanitize


def test_sanitize_redacts_sensitive_keys():
    data = {
        "command": "dir",
        "api_key": "gsk_real-looking-key-1234567890",
        "nested": {"password": "hunter2", "ok": True},
        "Authorization": "Bearer abc.def.ghi",
    }
    out = sanitize(data)
    assert out["api_key"] == "***"
    assert out["nested"]["password"] == "***"
    assert out["Authorization"] == "***"
    assert out["command"] == "dir"
    assert out["nested"]["ok"] is True


def test_sanitize_redacts_tokens_inside_strings():
    assert sanitize("key=gsk_abcdefgh12345678") == "key=***"
    assert sanitize("Authorization: Bearer sometoken123") != "Authorization: Bearer sometoken123"
    assert sanitize(["a", {"token": "x"}]) == ["a", {"token": "***"}]
    assert sanitize(42) == 42


def test_audit_record_writes_jsonl(tmp_path):
    path = tmp_path / "tools.jsonl"
    audit = AuditLogger(path)
    event = audit.record(
        tool_name="system.info",
        arguments={"api_key": "should-never-persist"},
        risk_level="SAFE",
        permission="allowed",
        confirmation="not_required",
        status="completed",
        duration_ms=3,
    )
    assert event["arguments"] == {"api_key": "***"}
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    stored = json.loads(lines[0])
    assert stored["tool"] == "system.info"
    assert stored["status"] == "completed"
    assert stored["arguments"] == {"api_key": "***"}
    assert "timestamp" in stored
    for field in ("risk_level", "permission", "confirmation", "duration_ms"):
        assert field in stored
