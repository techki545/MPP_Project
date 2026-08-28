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


@pytest.mark.parametrize("variable", ["MPP_KB_SOURCE", "MPP_KB_DATA"])
def test_blank_path_values_use_documented_defaults(tmp_path: Path, variable: str):
    settings = Settings.from_mapping({variable: "   "}, project_root=tmp_path)
    defaults = Settings.from_mapping({}, project_root=tmp_path)

    assert getattr(settings, variable.removeprefix("MPP_KB_").lower() + "_dir") == getattr(
        defaults, variable.removeprefix("MPP_KB_").lower() + "_dir"
    )
    assert getattr(settings, variable.removeprefix("MPP_KB_").lower() + "_dir") != tmp_path


def test_default_data_path_is_project_local(tmp_path: Path):
    settings = Settings.from_mapping({}, project_root=tmp_path)

    assert settings.data_dir == tmp_path / ".local" / "knowledge_base"


def test_local_embedding_is_the_default_and_uses_project_local_model(tmp_path: Path):
    settings = Settings.from_mapping({}, project_root=tmp_path)

    assert settings.embedding_provider == "local"
    assert settings.embedding_model == "intfloat/multilingual-e5-small"
    assert settings.local_embedding_model_path == (
        tmp_path
        / ".local"
        / "models"
        / "multilingual-e5-small"
        / "onnx"
        / "model.onnx"
    )


def test_invalid_embedding_provider_has_stable_error(tmp_path: Path):
    with pytest.raises(ConfigurationError) as captured:
        Settings.from_mapping(
            {"MPP_EMBEDDING_PROVIDER": "unknown"}, project_root=tmp_path
        )

    assert captured.value.code == "invalid_embedding_provider"


def test_settings_repr_does_not_expose_api_key(tmp_path: Path):
    settings = Settings.from_mapping(
        {"MPP_API_KEY": "not-a-real-secret-for-tests"}, project_root=tmp_path
    )

    assert "not-a-real-secret-for-tests" not in repr(settings)


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
            "MPP_EMBEDDING_PROVIDER": "remote",
            "MPP_API_BASE": "https://provider.example/v1",
            "MPP_API_KEY": "local-test-secret",
        },
        project_root=Path.cwd(),
    )

    settings.require_embedding_access()


@pytest.mark.parametrize("raw_value", ["abc", "0", "-1"])
def test_invalid_embedding_batch_size_has_stable_error(
    raw_value: str, tmp_path: Path
):
    with pytest.raises(ConfigurationError) as exc_info:
        Settings.from_mapping(
            {"MPP_EMBEDDING_BATCH_SIZE": raw_value}, project_root=tmp_path
        )

    error = exc_info.value
    assert error.code == "invalid_embedding_batch_size"
    assert error.details == {"value": raw_value}


def test_positive_embedding_batch_size_is_accepted(tmp_path: Path):
    settings = Settings.from_mapping(
        {"MPP_EMBEDDING_BATCH_SIZE": "64"}, project_root=tmp_path
    )

    assert settings.embedding_batch_size == 64
