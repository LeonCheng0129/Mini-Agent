"""Tests for workspace-scoped session persistence."""

import tempfile
from pathlib import Path

from mini_agent.logger import AgentLogger
from mini_agent.schema import Message
from mini_agent.session_store import SessionStore


def test_lazy_init_does_not_create_directories():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        store = SessionStore(workspace)

        assert not store.session_dir.exists()
        assert not store.logs_root.exists()
        assert store.get_current_session_id() is None
        assert store.list_sessions() == []


def test_materialize_session_and_restore_messages():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        store = SessionStore(workspace)
        messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="hello world"),
        ]

        meta = store.create_session(messages=messages, title_hint="hello world")
        assert store.session_dir.exists()
        assert store.logs_root.exists()
        assert store.get_current_session_id() == meta.session_id
        assert meta.title == "hello world"

        restored = store.load_session_messages(meta.session_id)
        assert len(restored) == 2
        assert restored[0].role == "system"
        assert restored[1].content == "hello world"


def test_append_replace_and_log_tracking():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        store = SessionStore(workspace)
        meta = store.create_session(
            messages=[
                Message(role="system", content="sys"),
                Message(role="user", content="first"),
            ],
            title_hint="first",
        )

        store.append_messages(
            meta.session_id,
            [
                Message(role="assistant", content="ok"),
                Message(role="tool", content="done", tool_call_id="1", name="mock"),
            ],
        )
        restored = store.load_session_messages(meta.session_id)
        assert len(restored) == 4
        assert restored[-1].role == "tool"

        store.replace_messages(meta.session_id, [Message(role="system", content="sys")])
        restored = store.load_session_messages(meta.session_id)
        assert len(restored) == 1
        assert restored[0].role == "system"

        store.add_log_file(meta.session_id, "agent_run_1.log")
        updated_meta = store.get_session(meta.session_id)
        assert updated_meta is not None
        assert updated_meta.log_files == ["agent_run_1.log"]
        assert updated_meta.last_log_file == "agent_run_1.log"


def test_load_session_ignores_corrupt_tail_line():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        store = SessionStore(workspace)
        meta = store.create_session(
            messages=[
                Message(role="system", content="sys"),
                Message(role="user", content="first"),
            ],
            title_hint="first",
        )

        session_file = store.session_dir / f"{meta.session_id}.jsonl"
        with session_file.open("a", encoding="utf-8") as handle:
            handle.write("{bad-json")

        restored = store.load_session_messages(meta.session_id)
        assert len(restored) == 2
        assert restored[1].content == "first"


def test_google_thought_signature_round_trip():
    with tempfile.TemporaryDirectory() as tmpdir:
        workspace = Path(tmpdir)
        store = SessionStore(workspace)
        meta = store.create_session(
            messages=[
                Message(role="system", content="sys"),
                Message(
                    role="assistant",
                    content="tool call",
                    thought_signature=b"gemini-signature",
                ),
            ],
            title_hint="gemini",
        )

        restored = store.load_session_messages(meta.session_id)
        assert len(restored) == 2
        assert restored[1].thought_signature == b"gemini-signature"


def test_logger_lazy_directory_creation():
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = Path(tmpdir) / ".logs" / "sess-1"
        logger = AgentLogger(log_dir=log_dir)
        assert not log_dir.exists()

        logger.start_new_run()
        assert log_dir.exists()
        assert logger.get_log_file_path() is not None
