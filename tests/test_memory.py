"""Persistent memory tests (Phase 3). Fully offline, tmp databases only."""

from datetime import datetime, timezone

import pytest

from app.core.exceptions import MemoryRejected
from app.memory.categories import normalize_category
from app.memory.database import create_memory_engine, init_db
from app.memory.manager import MemoryManager
from app.memory.models import Base, MemoryEntry
from app.memory.redaction import evaluate


@pytest.fixture
def manager(tmp_path):
    engine = create_memory_engine(str(tmp_path / "test.db"))
    return MemoryManager(engine)


def test_init_db_creates_tables(tmp_path):
    engine = create_memory_engine(str(tmp_path / "fresh.db"))
    init_db(engine)
    init_db(engine)  # idempotent, never destroys data
    assert "memories" in Base.metadata.tables


def test_remember_returns_record(manager):
    rec = manager.remember("hello world", conversation_id="c1")
    assert rec.id > 0
    assert rec.content == "hello world"
    assert rec.category == "conversation"
    assert rec.conversation_id == "c1"
    assert rec.created_at is not None
    assert rec.updated_at is not None
    assert rec.is_deleted is False


def test_remember_rejects_empty(manager):
    with pytest.raises(ValueError):
        manager.remember("   ")


def test_remember_rejects_unknown_category(manager):
    with pytest.raises(ValueError):
        manager.remember("x", category="dreams")


def test_normalize_category_is_strict():
    assert normalize_category(" Fact ") == "fact"
    with pytest.raises(ValueError):
        normalize_category("nope")


def test_tool_history_needs_explicit_flag(manager):
    with pytest.raises(ValueError):
        manager.remember("ran ls", category="tool_history", conversation_id="c1")
    rec = manager.remember(
        "ran ls", category="tool_history", conversation_id="c1",
        explicit_tool_history=True,
    )
    assert rec.category == "tool_history"


def test_get_returns_record_and_touches(manager):
    rec = manager.remember("find me", conversation_id="c1")
    fetched = manager.get(rec.id)
    assert fetched is not None and fetched.id == rec.id
    assert fetched.last_accessed_at is not None
    assert manager.get(999999) is None


def test_search_by_text(manager):
    manager.remember("the ev project uses matlab", conversation_id="c1")
    manager.remember("unrelated cooking note", conversation_id="c1")
    hits = manager.search("matlab")
    assert [h.content for h in hits] == ["the ev project uses matlab"]


def test_search_category_filter(manager):
    manager.remember("my favorite color is blue", category="preference")
    manager.remember("blue color conversation", category="conversation")
    hits = manager.search("blue", category="preference")
    assert all(h.category == "preference" for h in hits)
    assert len(hits) == 1


def test_search_conversation_filter(manager):
    manager.remember("shared keyword here", conversation_id="aaa")
    manager.remember("shared keyword here", conversation_id="bbb")
    assert len(manager.search("shared", conversation_id="aaa")) == 1


def test_search_empty_query_lists_by_importance(manager):
    manager.remember("low priority", importance=0.1)
    manager.remember("high priority", importance=0.9)
    hits = manager.search("")
    assert hits[0].content == "high priority"


def test_get_recent_orders_newest_first(manager):
    manager.remember("first", conversation_id="c1")
    manager.remember("second", conversation_id="c1")
    recent = manager.get_recent("c1", limit=2)
    assert [r.content for r in recent] == ["second", "first"]


def test_get_context_merges_turns_and_knowledge(manager):
    manager.remember("turn one", conversation_id="c1")
    manager.remember("my favorite color is blue", category="preference")
    ctx = manager.get_context("other-conversation", query="favorite color")
    assert any(r.content == "turn one" for r in ctx) is False
    assert any("blue" in r.content for r in ctx)
    ctx_own = manager.get_context("c1")
    assert any(r.content == "turn one" for r in ctx_own)


def test_get_context_is_deterministic(manager):
    for i in range(5):
        manager.remember(f"fact number {i}", category="fact")
    first = [r.id for r in manager.get_context("c1", query="fact")]
    second = [r.id for r in manager.get_context("c1", query="fact")]
    assert first == second


def test_format_context_bounds_output(manager):
    manager.remember("x" * 5000, category="fact")
    text = manager.format_context(manager.get_context("c1"))
    assert len(text) <= manager.context_max_chars + 200
    assert manager.format_context([]) == ""


def test_forget_soft_deletes(manager):
    rec = manager.remember("temporary", conversation_id="c1")
    assert manager.forget(rec.id) is True
    assert manager.get(rec.id) is None
    assert manager.search("temporary") == []
    assert manager.forget(rec.id) is False  # already gone
    assert manager.forget(424242) is False


def test_forget_scoped_to_conversation(manager):
    rec = manager.remember("shared", conversation_id="aaa")
    assert manager.forget(rec.id, conversation_id="bbb") is False
    assert manager.get(rec.id) is not None


def test_clear_conversation(manager):
    manager.remember("one", conversation_id="c1")
    manager.remember("two", conversation_id="c1")
    manager.remember("other", conversation_id="c2")
    assert manager.clear_conversation("c1") == 2
    assert manager.get_recent("c1") == []
    assert len(manager.get_recent("c2")) == 1


def test_timestamps_present_and_sane(manager):
    rec = manager.remember("timed", conversation_id="c1")
    now = datetime.now(timezone.utc)
    assert rec.created_at is not None and rec.updated_at is not None
    assert abs((now - rec.created_at).total_seconds()) < 60


def test_count_helper(manager):
    manager.remember("a", conversation_id="c1", category="fact")
    manager.remember("b", conversation_id="c1", category="fact")
    manager.remember("c", conversation_id="c2", category="fact")
    assert manager.count(conversation_id="c1") == 2
    assert manager.count(category="fact") == 3


def test_retention_purge(tmp_path):
    engine = create_memory_engine(str(tmp_path / "r.db"))
    manager = MemoryManager(engine, retention_days=30)
    assert manager.purge_expired() == 0
    ever = MemoryManager(create_memory_engine(str(tmp_path / "r2.db")), retention_days=0)
    assert ever.purge_expired() == 0


# -- redaction ---------------------------------------------------------------

SECRET_CASES = [
    "gsk_abcdefgh12345678",
    "sk-abcdefgh1234567890",
    "xoxb-1234567890-abcdefgh",
    "Authorization: Bearer sometoken123",
    "Bearer sometoken123",
    "api_key = gsk_abcdefgh12345678",
    "API_KEY=gsk_abcdefgh12345678",
    "password: hunter2x",
    '"password": "hunter2x"',
    "DB_PASSWORD=Sup3rSecret99",
    "-----BEGIN RSA PRIVATE KEY-----\nMIIE...fakekey...\n-----END RSA PRIVATE KEY-----",
    "API_KEY=gsk_1\nOTHER_SECRET=topsecretvalue1",
]


@pytest.mark.parametrize("secret", SECRET_CASES)
def test_redaction_rejects_secrets(manager, secret):
    assert evaluate(secret).decision == "reject"
    with pytest.raises(MemoryRejected):
        manager.remember(secret, conversation_id="c1")
    assert manager.count() == 0  # nothing stored


def test_redaction_handles_env_style_block(manager):
    blob = "APP_NAME=NEXUS\nAI_API_KEY=gsk_abcdefgh12345678\nTIMEOUT=30"
    result = evaluate(blob)
    assert result.decision == "redact"
    rec = manager.remember(blob)
    assert "gsk_abcdefgh12345678" not in rec.content
    assert "AI_API_KEY=***" in rec.content


def test_redaction_redacts_embedded_secrets(manager):
    text = "the deploy failed, see token gsk_abcdefgh12345678 in the logs"
    result = evaluate(text)
    assert result.decision == "redact"
    assert "gsk_abcdefgh12345678" not in result.text
    rec = manager.remember(text, conversation_id="c1")
    assert "gsk_abcdefgh12345678" not in rec.content


def test_redaction_ignores_placeholders(manager):
    for placeholder in [
        "my key is your-groq-api-key",
        "set AI_API_KEY=xxx and retry",
        "the password is test",
    ]:
        assert evaluate(placeholder).decision == "keep"
        manager.remember(placeholder, conversation_id="c1")
    assert manager.count() == 3


def test_rejected_memory_stores_nothing(manager):
    before = manager.count()
    with pytest.raises(MemoryRejected):
        manager.remember("gsk_abcdefgh12345678")
    assert manager.count() == before
