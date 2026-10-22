
"""Local language model integration using Ollama."""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class LLMServiceError(RuntimeError):
    """Raised when the local language model cannot generate a response."""


class OllamaService:
    """Generate text using a locally running Ollama model."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = (
            base_url or os.getenv(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            )
        ).rstrip("/")
        self.model = model or os.getenv(
            "OLLAMA_MODEL", "qwen2.5:3b"
        )
        self.timeout = timeout

        if not self.model.strip():
            raise ValueError("Ollama model name cannot be empty.")

        if self.timeout <= 0:
            raise ValueError("Timeout must be greater than zero.")

    def generate(
        self,
        prompt: str,
        temperature: float = 0.1,
    ) -> str:
        """Generate a non-streaming response from Ollama."""

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        if not 0.0 <= temperature <= 2.0:
            raise ValueError("Temperature must be between 0 and 2.")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }

        request = Request(
            url=f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))

        except HTTPError as exc:
            raise LLMServiceError(
                f"Ollama returned HTTP {exc.code}."
            ) from exc

        except (URLError, TimeoutError, OSError) as exc:
            raise LLMServiceError(
                "Cannot connect to Ollama. Ensure it is running "
                "and the configured URL is correct."
            ) from exc

        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise LLMServiceError(
                "Ollama returned an invalid JSON response."
            ) from exc

        generated_text = data.get("response")

        if not isinstance(generated_text, str) or not generated_text.strip():
            raise LLMServiceError(
                "Ollama returned an empty generated response."
            )

        return generated_text.strip()
