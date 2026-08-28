from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .errors import ConfigurationError


@dataclass(frozen=True)
class Settings:
    project_root: Path
    source_dir: Path
    data_dir: Path
    api_base: str
    api_key: str = field(repr=False)
    embedding_provider: str
    embedding_model: str
    local_embedding_dir: Path
    chat_model: str
    embedding_batch_size: int = 16

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
            raw_value = env.get(name, "")
            value = raw_value.strip() or default
            path = Path(value).expanduser()
            return path if path.is_absolute() else root / path

        raw_batch_size = env.get("MPP_EMBEDDING_BATCH_SIZE", "16")
        try:
            batch_size = int(raw_batch_size.strip())
        except ValueError as exc:
            raise ConfigurationError(
                "invalid_embedding_batch_size",
                "MPP_EMBEDDING_BATCH_SIZE must be an integer",
                details={"value": raw_batch_size},
            ) from exc

        if batch_size < 1:
            raise ConfigurationError(
                "invalid_embedding_batch_size",
                "MPP_EMBEDDING_BATCH_SIZE must be positive",
                details={"value": raw_batch_size},
            )

        embedding_provider = env.get("MPP_EMBEDDING_PROVIDER", "local").strip().lower()
        if embedding_provider not in {"local", "remote"}:
            raise ConfigurationError(
                "invalid_embedding_provider",
                "MPP_EMBEDDING_PROVIDER must be local or remote",
                details={"value": embedding_provider},
            )

        return cls(
            project_root=root,
            source_dir=path_value("MPP_KB_SOURCE", r"C:\Users\LTC\Desktop\MPP"),
            data_dir=path_value("MPP_KB_DATA", ".local/knowledge_base"),
            api_base=env.get("MPP_API_BASE", "").strip().rstrip("/"),
            api_key=env.get("MPP_API_KEY", "").strip(),
            embedding_provider=embedding_provider,
            embedding_model=env.get(
                "MPP_EMBEDDING_MODEL", "intfloat/multilingual-e5-small"
            ).strip(),
            local_embedding_dir=path_value(
                "MPP_LOCAL_EMBEDDING_DIR",
                ".local/models/multilingual-e5-small/onnx",
            ),
            chat_model=env.get("MPP_CHAT_MODEL", "").strip(),
            embedding_batch_size=batch_size,
        )

    @property
    def local_embedding_model_path(self) -> Path:
        optimized = self.local_embedding_dir / "model_qint8_avx512_vnni.onnx"
        standard = self.local_embedding_dir / "model.onnx"
        return optimized if optimized.is_file() else standard

    @property
    def local_embedding_tokenizer_path(self) -> Path:
        return self.local_embedding_dir / "tokenizer.json"

    @property
    def local_embedding_ready(self) -> bool:
        return (
            self.local_embedding_model_path.is_file()
            and self.local_embedding_tokenizer_path.is_file()
        )

    def require_embedding_access(self) -> None:
        if self.embedding_provider == "local":
            missing = [
                str(path)
                for path in (
                    self.local_embedding_model_path,
                    self.local_embedding_tokenizer_path,
                )
                if not path.is_file()
            ]
            if missing:
                raise ConfigurationError(
                    "local_embedding_model_missing",
                    "Local embedding model files are missing",
                    details={"missing": missing},
                )
            return
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
                details={"missing": missing},
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
                details={"missing": missing},
            )
