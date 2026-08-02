from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .errors import ConfigurationError


@dataclass(frozen=True)
class Settings:
    project_root: Path
    source_dir: Path
    data_dir: Path
    api_base: str
    api_key: str
    embedding_model: str
    chat_model: str
    embedding_batch_size: int = 32

    @property
    def qdrant_dir(self) -> Path:
        return self.data_dir / "qdrant"

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "manifest.sqlite3"

    @classmethod
    def from_mapping(cls, env: Mapping[str, str], project_root: Path) -> "Settings":
        root = Path(project_root).resolve()

        def path_value(name: str, default: str) -> Path:
            value = env.get(name, default).strip()
            path = Path(value).expanduser()
            return path if path.is_absolute() else root / path

        try:
            batch_size = int(env.get("MPP_EMBEDDING_BATCH_SIZE", "32").strip())
        except ValueError as exc:
            raise ConfigurationError(
                "embedding_batch_size_invalid",
                "MPP_EMBEDDING_BATCH_SIZE must be an integer",
                {"value": env.get("MPP_EMBEDDING_BATCH_SIZE")},
            ) from exc

        if batch_size < 1:
            raise ConfigurationError(
                "embedding_batch_size_invalid",
                "MPP_EMBEDDING_BATCH_SIZE must be positive",
                {"value": batch_size},
            )

        return cls(
            project_root=root,
            source_dir=path_value("MPP_KB_SOURCE", r"C:\Users\LTC\Desktop\MPP"),
            data_dir=path_value("MPP_KB_DATA", r".local\knowledge_base"),
            api_base=env.get("MPP_API_BASE", "").strip().rstrip("/"),
            api_key=env.get("MPP_API_KEY", "").strip(),
            embedding_model=env.get("MPP_EMBEDDING_MODEL", "").strip(),
            chat_model=env.get("MPP_CHAT_MODEL", "").strip(),
            embedding_batch_size=batch_size,
        )

    def require_embedding_access(self) -> None:
        missing = [
            name
            for name, value in (
                ("MPP_API_BASE", self.api_base),
                ("MPP_API_KEY", self.api_key),
                ("MPP_EMBEDDING_MODEL", self.embedding_model),
            )
            if not value
        ]
        if missing:
            raise ConfigurationError(
                "embedding_config_missing",
                "Embedding configuration is incomplete",
                {"missing": missing},
            )

    def require_chat_access(self) -> None:
        missing = [
            name
            for name, value in (
                ("MPP_API_BASE", self.api_base),
                ("MPP_API_KEY", self.api_key),
                ("MPP_CHAT_MODEL", self.chat_model),
            )
            if not value
        ]
        if missing:
            raise ConfigurationError(
                "model_config_missing",
                "Chat model configuration is incomplete",
                {"missing": missing},
            )
