import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from detector.utils import model
from detector.utils.model_profiles import get_model_profile


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=7,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=3),
        )
        choice = SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content="answer"),
        )
        return SimpleNamespace(choices=[choice], usage=usage)


class _FakeOpenAI:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.chat = SimpleNamespace(completions=_FakeCompletions())
        self.closed = False

    def close(self):
        self.closed = True


class _FakeHTTPClient:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.closed = False

    def close(self):
        self.closed = True


class ModelAccessTests(unittest.TestCase):
    def test_normalize_chat_completions_endpoint(self):
        self.assertEqual(
            model._normalize_base_url(
                " \nhttp://localhost:8000/v1/chat/completions/\t"
            ),
            "http://localhost:8000/v1",
        )
        self.assertEqual(
            model._normalize_base_url("https://api.openai.com/v1/"),
            "https://api.openai.com/v1",
        )

    def test_base_url_is_required(self):
        with self.assertRaisesRegex(ValueError, "base_url"):
            model._normalize_base_url("")

    def test_profiles_expand_or_clear_environment_placeholders(self):
        with patch.dict(
            "os.environ",
            {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_MODEL": "gpt-test",
            },
            clear=True,
        ):
            openai_profile = get_model_profile("openai")
            self.assertEqual(openai_profile["api_key"], "test-key")
            self.assertEqual(openai_profile["model"], "gpt-test")
            self.assertEqual(
                openai_profile["base_url"], "https://api.openai.com/v1"
            )

            self_hosted = get_model_profile("self_hosted")
            self.assertIsNone(self_hosted["base_url"])
            self.assertIsNone(self_hosted["model"])
            self.assertEqual(self_hosted["api_key"], "EMPTY")

    def test_client_uses_explicit_normalized_configuration_and_keeps_usage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = str(Path(temp_dir) / "model.cache")
            with (
                patch.object(model, "OpenAI", _FakeOpenAI),
                patch.object(model.httpx, "Client", _FakeHTTPClient),
            ):
                client = model.APIModel(
                    cache_path,
                    "http://localhost:8000/v1/chat/completions",
                    "local-model",
                    "EMPTY",
                    extra_params={"seed": 4},
                )
                self.assertEqual(client.base_url, "http://localhost:8000/v1")
                self.assertEqual(client.client.init_kwargs["api_key"], "EMPTY")
                self.assertEqual(
                    client.client.init_kwargs["base_url"],
                    "http://localhost:8000/v1",
                )

                response, usage = client.generate("hello", return_usage=True)
                self.assertEqual(response, "answer")
                self.assertEqual(
                    usage,
                    {
                        "input_tokens": 11,
                        "reasoning_tokens": 3,
                        "output_tokens": 7,
                    },
                )
                call = client.client.chat.completions.calls[0]
                self.assertEqual(call["model"], "local-model")
                self.assertEqual(call["seed"], 4)

                cached, cached_usage = client.generate("hello", return_usage=True)
                self.assertEqual(cached, "answer")
                self.assertIsNone(cached_usage)
                self.assertEqual(len(client.client.chat.completions.calls), 1)
                client.close()


if __name__ == "__main__":
    unittest.main()
