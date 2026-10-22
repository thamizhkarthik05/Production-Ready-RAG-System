
import json
from io import BytesIO
from urllib.error import URLError

import pytest

from app.llm_service import LLMServiceError, OllamaService


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def test_generate_returns_model_response(monkeypatch):
    def fake_urlopen(request, timeout):
        assert request.full_url == (
            "http://localhost:11434/api/generate"
        )
        assert timeout == 120.0

        payload = json.loads(request.data.decode("utf-8"))
        assert payload["model"] == "qwen2.5:3b"
        assert payload["prompt"] == "Explain RAG."
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0.1

        return FakeResponse({"response": "RAG retrieves relevant evidence."})

    monkeypatch.setattr("app.llm_service.urlopen", fake_urlopen)

    service = OllamaService()
    result = service.generate("Explain RAG.")

    assert result == "RAG retrieves relevant evidence."


def test_generate_strips_response_whitespace(monkeypatch):
    monkeypatch.setattr(
        "app.llm_service.urlopen",
        lambda request, timeout: FakeResponse(
            {"response": "  Grounded answer. \n"}
        ),
    )

    assert OllamaService().generate("Question?") == "Grounded answer."


@pytest.mark.parametrize("prompt", ["", "   ", "\n"])
def test_generate_rejects_empty_prompt(prompt):
    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        OllamaService().generate(prompt)


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_generate_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="Temperature"):
        OllamaService().generate("Question?", temperature=temperature)


def test_generate_rejects_empty_model_response(monkeypatch):
    monkeypatch.setattr(
        "app.llm_service.urlopen",
        lambda request, timeout: FakeResponse({"response": "  "}),
    )

    with pytest.raises(LLMServiceError, match="empty generated response"):
        OllamaService().generate("Question?")


def test_generate_handles_connection_error(monkeypatch):
    def fake_urlopen(request, timeout):
        raise URLError("Connection refused")

    monkeypatch.setattr("app.llm_service.urlopen", fake_urlopen)

    with pytest.raises(LLMServiceError, match="Cannot connect to Ollama"):
        OllamaService().generate("Question?")


def test_generate_handles_invalid_json(monkeypatch):
    class InvalidJSONResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def read(self):
            return b"not valid JSON"

    monkeypatch.setattr(
        "app.llm_service.urlopen",
        lambda request, timeout: InvalidJSONResponse(),
    )

    with pytest.raises(LLMServiceError, match="invalid JSON"):
        OllamaService().generate("Question?")


def test_service_reads_configuration_from_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:3b")

    service = OllamaService()

    assert service.base_url == "http://127.0.0.1:11434"
    assert service.model == "qwen2.5:3b"
