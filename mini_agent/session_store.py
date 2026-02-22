"""Session persistence for CLI conversation restore."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from secrets import token_hex

from mini_agent.schema import Message


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(path)


def _build_title(first_user_message: str) -> str:
    one_line = " ".join(first_user_message.split())
    if not one_line:
        return f"Session {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    return one_line[:60]


@dataclass
class SessionMeta:
    session_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int
    last_role: str
    workspace: str
    log_files: list[str]
    last_log_file: str | None

    @classmethod
    def from_dict(cls, data: dict) -> "SessionMeta":
        return cls(
            session_id=data["session_id"],
            title=data["title"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            message_count=int(data.get("message_count", 0)),
            last_role=data.get("last_role", "system"),
            workspace=data.get("workspace", ""),
            log_files=list(data.get("log_files", [])),
            last_log_file=data.get("last_log_file"),
        )

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": self.message_count,
            "last_role": self.last_role,
            "workspace": self.workspace,
            "log_files": self.log_files,
            "last_log_file": self.last_log_file,
        }


class SessionStore:
    """Workspace-scoped session storage with lazy materialization."""

    def __init__(self, workspace_dir: Path | str):
        self.workspace_dir = Path(workspace_dir).resolve()
        self.session_dir = self.workspace_dir / ".session"
        self.index_path = self.session_dir / "index.json"
        self.logs_root = self.workspace_dir / ".logs"

    def has_materialized_sessions(self) -> bool:
        return self.index_path.exists()

    def generate_session_id(self) -> str:
        return f"sess-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{token_hex(3)}"

    def get_session_log_dir(self, session_id: str) -> Path:
        return self.logs_root / session_id

    def get_current_session_id(self) -> str | None:
        index = self._load_index()
        return index.get("current_session_id")

    def set_current_session(self, session_id: str) -> None:
        index = self._load_index()
        index["current_session_id"] = session_id
        self._save_index(index)

    def list_sessions(self, limit: int = 20) -> list[SessionMeta]:
        index = self._load_index()
        metas = [SessionMeta.from_dict(item) for item in index.get("sessions", [])]
        metas.sort(key=lambda item: item.updated_at, reverse=True)
        return metas[:limit]

    def get_session(self, session_id: str) -> SessionMeta | None:
        for meta in self.list_sessions(limit=10_000):
            if meta.session_id == session_id:
                return meta
        return None

    def create_session(self, messages: list[Message], title_hint: str | None = None) -> SessionMeta:
        if not messages or messages[0].role != "system":
            raise ValueError("Session must include system message as first message.")

        session_id = self.generate_session_id()
        now = _now_iso()
        title = _build_title(title_hint or "")

        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.logs_root.mkdir(parents=True, exist_ok=True)
        self._write_session_messages(session_id, messages)

        meta = SessionMeta(
            session_id=session_id,
            title=title,
            created_at=now,
            updated_at=now,
            message_count=len(messages),
            last_role=messages[-1].role,
            workspace=str(self.workspace_dir),
            log_files=[],
            last_log_file=None,
        )

        index = self._load_index()
        index["current_session_id"] = session_id
        sessions = [SessionMeta.from_dict(item) for item in index.get("sessions", [])]
        sessions.append(meta)
        index["sessions"] = [item.to_dict() for item in sessions]
        self._save_index(index)
        return meta

    def append_messages(self, session_id: str, messages: list[Message]) -> None:
        if not messages:
            return

        session_file = self.session_dir / f"{session_id}.jsonl"
        session_file.parent.mkdir(parents=True, exist_ok=True)
        with session_file.open("a", encoding="utf-8") as handle:
            for message in messages:
                handle.write(json.dumps(self._message_to_record(message), ensure_ascii=False) + "\n")

        self._update_meta_after_change(session_id=session_id, messages=messages)

    def replace_messages(self, session_id: str, messages: list[Message]) -> None:
        if not messages or messages[0].role != "system":
            raise ValueError("Session replace requires a system message.")
        self._write_session_messages(session_id, messages)
        self._update_meta_after_replace(session_id=session_id, messages=messages)

    def load_session_messages(self, session_id: str) -> list[Message]:
        session_file = self.session_dir / f"{session_id}.jsonl"
        if not session_file.exists():
            raise FileNotFoundError(f"Session file not found: {session_file}")

        messages: list[Message] = []
        with session_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # Allow recovery when tail line is partially written.
                    continue
                messages.append(self._record_to_message(record))
        return messages

    def add_log_file(self, session_id: str, log_filename: str) -> None:
        index = self._load_index()
        sessions = [SessionMeta.from_dict(item) for item in index.get("sessions", [])]
        changed = False
        for meta in sessions:
            if meta.session_id != session_id:
                continue
            if log_filename not in meta.log_files:
                meta.log_files.append(log_filename)
            meta.last_log_file = log_filename
            meta.updated_at = _now_iso()
            changed = True
            break

        if changed:
            index["sessions"] = [item.to_dict() for item in sessions]
            self._save_index(index)

    def _write_session_messages(self, session_id: str, messages: list[Message]) -> None:
        session_file = self.session_dir / f"{session_id}.jsonl"
        lines = [json.dumps(self._message_to_record(message), ensure_ascii=False) for message in messages]
        _atomic_write_text(session_file, "\n".join(lines) + "\n")

    def _update_meta_after_change(self, session_id: str, messages: list[Message]) -> None:
        index = self._load_index()
        sessions = [SessionMeta.from_dict(item) for item in index.get("sessions", [])]
        now = _now_iso()
        for meta in sessions:
            if meta.session_id != session_id:
                continue
            meta.updated_at = now
            meta.message_count += len(messages)
            meta.last_role = messages[-1].role
            break
        index["current_session_id"] = session_id
        index["sessions"] = [item.to_dict() for item in sessions]
        self._save_index(index)

    def _update_meta_after_replace(self, session_id: str, messages: list[Message]) -> None:
        index = self._load_index()
        sessions = [SessionMeta.from_dict(item) for item in index.get("sessions", [])]
        now = _now_iso()
        for meta in sessions:
            if meta.session_id != session_id:
                continue
            meta.updated_at = now
            meta.message_count = len(messages)
            meta.last_role = messages[-1].role
            break
        index["current_session_id"] = session_id
        index["sessions"] = [item.to_dict() for item in sessions]
        self._save_index(index)

    def _load_index(self) -> dict:
        if not self.index_path.exists():
            return {"version": 1, "current_session_id": None, "sessions": []}
        try:
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            backup = self.index_path.with_suffix(f".corrupt.{datetime.now().strftime('%Y%m%d%H%M%S')}.json")
            self.index_path.replace(backup)
            return {"version": 1, "current_session_id": None, "sessions": []}

    def _save_index(self, data: dict) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(self.index_path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")

    @staticmethod
    def _message_to_record(message: Message) -> dict:
        data = message.model_dump()
        if message.thought_signature is not None:
            data["thought_signature"] = base64.b64encode(message.thought_signature).decode("ascii")
        return data

    @staticmethod
    def _record_to_message(record: dict) -> Message:
        if "thought_signature" in record and isinstance(record["thought_signature"], str):
            try:
                record["thought_signature"] = base64.b64decode(record["thought_signature"].encode("ascii"))
            except Exception:
                record["thought_signature"] = None
        return Message.model_validate(record)
