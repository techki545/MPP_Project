from pathlib import Path

import pytest

from knowledge_base.config import Settings
from knowledge_base.errors import ConfigurationError


def test_settings_resolves_source_data_and_storage_paths(tmp_path: Path):
    settings = Settings.from_mapping(
        {
            "MPP_KB_SOURCE": "corpus",
            "MPP_KB_DATA": ".local/kb",
        },
        project_root=tmp_path,
    )

    assert settings.source_dir == tmp_path / "corpus"
    assert settings.data_dir == tmp_path / ".local" / "kb"
    assert settings.qdrant_dir == tmp_path / ".local" / "kb" / "qdrant"
    assert settings.sqlite_path == tmp_path / ".local" / "kb" / "manifest.sqlite3"


def test_require_chat_access_reports_exact_missing_fields():
    settings = Settings.from_mapping({}, project_root=Path.cwd())

    with pytest.raises(ConfigurationError) as exc_info:
        settings.require_chat_access()

    error = exc_info.value
    assert error.code == "model_config_missing"
    assert error.details == {
        "missing": ["MPP_API_BASE", "MPP_API_KEY", "MPP_CHAT_MODEL"]
    }


def test_embedding_access_does_not_require_chat_model():
    settings = Settings.from_mapping(
        {
            "MPP_API_BASE": "https://provider.example/v1",
            "MPP_API_KEY": "local-test-secret",
            "MPP_EMBEDDING_MODEL": "text-embedding-3-large",
        },
        project_root=Path.cwd(),
    )

    settings.require_embedding_access()
