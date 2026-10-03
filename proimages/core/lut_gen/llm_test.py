import json

import numpy as np
import pytest
from pydantic import ValidationError

from proimages.core.lut_gen import llm
from proimages.core.lut_gen.llm import _encode_reference, _parse_params
from proimages.core.lut_gen.params import GradeParams

VALID_PAYLOAD = {
    "name": "Summer Film",
    "description": "warm and slightly faded",
    "tone": 0.1,
    "saturation": 0.8,
    "temperature": 0.5,
    "tint": 0.0,
    "contrast": 1.15,
    "lift": [0.0, 0.02, 0.08],
    "gamma": [1.0, 1.05, 1.1],
    "gain": [1.1, 1.05, 0.95],
}


def _scripted_client(monkeypatch, replies: list[str]):
    """Patch llm._client with a fake that answers with each reply in turn. Returns the
    recorded create() calls and the recorded _client() calls."""
    create_calls: list[dict] = []
    client_calls: list[tuple] = []

    class FakeCompletions:
        def create(self, **kwargs):
            create_calls.append(kwargs)
            message = type("Message", (), {"content": replies[len(create_calls) - 1]})
            choice = type("Choice", (), {"message": message})
            return type("Response", (), {"choices": [choice]})

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})

    def fake_client(*args):
        client_calls.append(args)
        return FakeClient()

    monkeypatch.setattr(llm, "_client", fake_client)
    return create_calls, client_calls


def test_the_baseline_payload_is_itself_a_valid_v2_grade():
    params = GradeParams.model_validate(VALID_PAYLOAD)

    assert params.name == "Summer Film"


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


def test_parse_params_rejects_a_gain_below_lift():
    payload = VALID_PAYLOAD | {"lift": [0.3, 0.0, 0.0], "gain": [0.2, 1.0, 1.0]}

    with pytest.raises(ValidationError):
        _parse_params(json.dumps(payload))


def test_parse_params_rejects_a_non_object_reply():
    with pytest.raises(ValidationError):
        _parse_params(json.dumps([1, 2, 3]))


def test_parse_params_fills_defaults_for_missing_fields():
    params = _parse_params(json.dumps({"name": "Sparse"}))

    assert params.name == "Sparse"
    assert params.tone == 0.0
    assert params.gamma == (1.0, 1.0, 1.0)
    assert params.schema_version == 2


def test_parse_params_ignores_model_supplied_schema_version():
    params = _parse_params(json.dumps(VALID_PAYLOAD | {"schema_version": 99}))

    assert params.schema_version == 2


def test_encode_reference_downscales_large_images():
    image = np.random.default_rng(0).random((2000, 1000, 3)).astype(np.float32)

    encoded = _encode_reference(image)

    assert isinstance(encoded, str) and len(encoded) > 0


def test_params_from_description_sends_json_mode_request(monkeypatch):
    create_calls, client_calls = _scripted_client(monkeypatch, [json.dumps(VALID_PAYLOAD)])

    params = llm.params_from_description("warm summer film", "gpt-4o", "sk-test")

    assert params.name == "Summer Film"
    assert client_calls == [("sk-test", None)]
    assert len(create_calls) == 1
    assert create_calls[0]["model"] == "gpt-4o"
    assert create_calls[0]["response_format"] == {"type": "json_object"}
    messages = create_calls[0]["messages"]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "warm summer film" in messages[1]["content"]


def test_params_from_reference_image_includes_stats_and_image(monkeypatch):
    create_calls, _ = _scripted_client(monkeypatch, [json.dumps(VALID_PAYLOAD)])
    image = np.full((32, 32, 3), 0.5, dtype=np.float32)

    params = llm.params_from_reference_image(image, "gpt-4o", "sk-test")

    assert params.name == "Summer Film"
    content = create_calls[0]["messages"][1]["content"]
    assert "warm_cool_balance" in content[0]["text"]
    assert "cdl_fit" in content[0]["text"]
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


@pytest.mark.parametrize(
    "bad_reply, complaint",
    [
        ("this is not json", "Expecting value"),
        (json.dumps(VALID_PAYLOAD | {"temperature": 4.2}), "temperature"),
        (json.dumps(VALID_PAYLOAD | {"notes": "extra key"}), "notes"),
    ],
)
def test_an_invalid_reply_is_retried_once_with_the_error_shown_to_the_model(monkeypatch, bad_reply, complaint):
    create_calls, client_calls = _scripted_client(monkeypatch, [bad_reply, json.dumps(VALID_PAYLOAD)])

    params = llm.params_from_description("warm summer film", "gpt-4o", "sk-secret")

    assert params.name == "Summer Film"
    assert client_calls == [("sk-secret", None)]
    assert len(create_calls) == 2

    first, second = (call["messages"] for call in create_calls)
    assert [message["role"] for message in first] == ["system", "user"]
    assert second[:2] == first
    assert second[2] == {"role": "assistant", "content": bad_reply}
    assert second[3]["role"] == "user"
    assert "That JSON was invalid" in second[3]["content"]
    assert complaint in second[3]["content"]
    assert "sk-secret" not in json.dumps(second)
    assert create_calls[1]["response_format"] == {"type": "json_object"}


def test_a_valid_reply_is_not_retried(monkeypatch):
    create_calls, _ = _scripted_client(monkeypatch, [json.dumps(VALID_PAYLOAD)])

    llm.params_from_description("warm summer film", "gpt-4o", "sk-test")

    assert len(create_calls) == 1


def test_two_invalid_replies_raise_after_exactly_two_requests(monkeypatch):
    bad = json.dumps(VALID_PAYLOAD | {"temperature": 4.2})
    create_calls, client_calls = _scripted_client(monkeypatch, [bad, bad, json.dumps(VALID_PAYLOAD)])

    with pytest.raises(ValidationError):
        llm.params_from_description("warm summer film", "gpt-4o", "sk-test")

    assert len(create_calls) == 2
    assert len(client_calls) == 1


def test_a_reply_that_stays_unparseable_raises_json_error(monkeypatch):
    create_calls, _ = _scripted_client(monkeypatch, ["nope", "still nope"])

    with pytest.raises(json.JSONDecodeError):
        llm.params_from_description("warm summer film", "gpt-4o", "sk-test")

    assert len(create_calls) == 2


def test_reference_image_requests_are_retried_too(monkeypatch):
    create_calls, client_calls = _scripted_client(monkeypatch, ["nope", json.dumps(VALID_PAYLOAD)])
    image = np.full((32, 32, 3), 0.5, dtype=np.float32)

    params = llm.params_from_reference_image(image, "gpt-4o", "sk-test")

    assert params.name == "Summer Film"
    assert len(create_calls) == 2
    assert len(client_calls) == 1
    assert create_calls[1]["messages"][1] == create_calls[0]["messages"][1]
