import json

import numpy as np
import pytest
from pydantic import ValidationError

from proimages.core.lut_gen import llm
from proimages.core.lut_gen.llm import _encode_reference, _parse_params

VALID_PAYLOAD = {
    "name": "Summer Film",
    "description": "warm and slightly faded",
    "tone": 0.1,
    "saturation": -0.2,
    "temperature": 0.5,
    "tint": 0.0,
    "contrast": 0.15,
    "lift": [0.0, 0.02, 0.08],
    "gamma": [0.0, 0.0, 0.0],
    "gain": [0.1, 0.05, -0.05],
}


def test_parse_params_accepts_a_well_formed_response():
    params = _parse_params(json.dumps(VALID_PAYLOAD))

    assert params.name == "Summer Film"
    assert params.temperature == 0.5
    assert params.lift == (0.0, 0.02, 0.08)


def test_parse_params_rejects_out_of_range_values():
    payload = VALID_PAYLOAD | {"temperature": 4.2}

    with pytest.raises(ValidationError):
        _parse_params(json.dumps(payload))


def test_parse_params_rejects_out_of_range_triplet_element():
    payload = VALID_PAYLOAD | {"lift": [0.0, 0.0, 9.9]}

    with pytest.raises(ValidationError):
        _parse_params(json.dumps(payload))


def test_parse_params_fills_defaults_for_missing_fields():
    params = _parse_params(json.dumps({"name": "Sparse"}))

    assert params.name == "Sparse"
    assert params.tone == 0.0
    assert params.schema_version == 1


def test_parse_params_ignores_model_supplied_schema_version():
    params = _parse_params(json.dumps(VALID_PAYLOAD | {"schema_version": 99}))

    assert params.schema_version == 1


def test_encode_reference_downscales_large_images():
    image = np.random.default_rng(0).random((2000, 1000, 3)).astype(np.float32)

    encoded = _encode_reference(image)

    assert isinstance(encoded, str) and len(encoded) > 0


def test_params_from_description_sends_json_mode_request(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = type("Message", (), {"content": json.dumps(VALID_PAYLOAD)})
            choice = type("Choice", (), {"message": message})
            return type("Response", (), {"choices": [choice]})

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})

    monkeypatch.setattr(llm, "_client", lambda api_key, base_url: FakeClient())

    params = llm.params_from_description("warm summer film", "gpt-4o", "sk-test")

    assert params.name == "Summer Film"
    assert captured["model"] == "gpt-4o"
    assert captured["response_format"] == {"type": "json_object"}
    assert "warm summer film" in captured["messages"][1]["content"]


def test_params_from_reference_image_includes_stats_and_image(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = type("Message", (), {"content": json.dumps(VALID_PAYLOAD)})
            choice = type("Choice", (), {"message": message})
            return type("Response", (), {"choices": [choice]})

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})

    monkeypatch.setattr(llm, "_client", lambda api_key, base_url: FakeClient())
    image = np.full((32, 32, 3), 0.5, dtype=np.float32)

    params = llm.params_from_reference_image(image, "gpt-4o", "sk-test")

    assert params.name == "Summer Film"
    content = captured["messages"][1]["content"]
    assert "warm_cool_balance" in content[0]["text"]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
