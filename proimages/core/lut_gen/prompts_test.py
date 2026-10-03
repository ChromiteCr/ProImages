import json

from proimages.core.lut_gen.params import GradeParams
from proimages.core.lut_gen.prompts import DESCRIPTION_PROMPT, REFERENCE_PROMPT, SYSTEM_PROMPT

MODEL_FIELDS = set(GradeParams.model_fields) - {"schema_version"}


def _json_examples() -> list[dict]:
    return [json.loads(line) for line in SYSTEM_PROMPT.splitlines() if line.startswith('{"name"')]


def test_the_templates_that_go_through_str_format_contain_only_their_placeholders():
    assert DESCRIPTION_PROMPT.format(description="warm and soft").endswith("warm and soft")
    assert '{"stats": 1}' in REFERENCE_PROMPT.format(stats='{"stats": 1}')


def test_system_prompt_is_used_verbatim_so_it_may_hold_json_braces():
    assert "{" in SYSTEM_PROMPT and "}" in SYSTEM_PROMPT


def test_system_prompt_names_every_parameter_the_model_must_return():
    for field in MODEL_FIELDS:
        assert f'"{field}"' in SYSTEM_PROMPT
    assert "schema_version" not in SYSTEM_PROMPT


def test_system_prompt_examples_are_complete_valid_grades():
    examples = _json_examples()

    assert len(examples) >= 1
    for example in examples:
        assert set(example) == MODEL_FIELDS
        GradeParams.model_validate(example)


def test_system_prompt_states_the_formula_neutrals_and_the_gamma_direction():
    assert "((gain - lift) * x + lift) ^ (1 / gamma)" in SYSTEM_PROMPT
    assert "lift 0, gamma 1 and gain 1" in SYSTEM_PROMPT
    assert "opposite direction" in SYSTEM_PROMPT and "power" in SYSTEM_PROMPT
    assert "brighter" in SYSTEM_PROMPT


def test_reference_prompt_says_the_stats_are_display_encoded_and_cdl_fit_is_rough():
    assert "display-encoded" in REFERENCE_PROMPT
    assert "cdl_fit" in REFERENCE_PROMPT and "rough starting point" in REFERENCE_PROMPT
    assert "neutral" in REFERENCE_PROMPT
