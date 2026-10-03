import argparse
import json
import os
import sys
from pathlib import Path

from pydantic import ValidationError

from proimages.core.lut_gen.bake import DEFAULT_LUT_SIZE, MAX_LUT_SIZE, MIN_LUT_SIZE, bake_cube_file
from proimages.core.lut_gen.cdl import cc_id_from_name, to_cc_xml, to_cdl
from proimages.core.lut_gen.params import SCHEMA_VERSION, GradeParams, format_validation_error
from proimages.core.system.io import load_image


def _load_params_file(path: str) -> GradeParams:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        sys.exit(f"cannot read --params file {path}: {exc}")

    if not isinstance(payload, dict) or "schema_version" not in payload:
        sys.exit(
            f'--params file {path} must be a JSON object containing "schema_version": {SCHEMA_VERSION} '
            "(lift/gamma/gain now use standard neutral values 0/1/1)"
        )

    try:
        return GradeParams.model_validate(payload)
    except ValidationError as exc:
        sys.exit(f"invalid --params file {path}: {format_validation_error(exc)}")


def _resolve_params(args: argparse.Namespace) -> GradeParams:
    if args.params:
        return _load_params_file(args.params)

    api_key = args.api_key or os.environ.get("PROIMAGES_API_KEY")
    if not api_key:
        sys.exit("--api-key is required (or set PROIMAGES_API_KEY) when using --describe or --reference")
    if not args.model:
        sys.exit("--model is required when using --describe or --reference")

    from proimages.core.lut_gen.llm import params_from_description, params_from_reference_image

    if args.describe:
        return params_from_description(args.describe, args.model, api_key, args.base_url)
    return params_from_reference_image(load_image(args.reference), args.model, api_key, args.base_url)


def main() -> None:
    parser = argparse.ArgumentParser(prog="proimages-lut", description="Generate a .cube LUT")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--describe", help="Describe the look you want in words")
    source.add_argument("--reference", help="Path to a reference photo to match")
    source.add_argument("--params", help="Path to a GradeParams JSON file (bakes directly, no model call)")

    parser.add_argument("-o", "--output", required=True, help="Path to write the .cube file")
    parser.add_argument("--cdl-out", help="Also write the grade as an ASC CDL .cc file here (exact, unlike the LUT)")
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_LUT_SIZE,
        help=f"LUT cube size per edge, {MIN_LUT_SIZE} to {MAX_LUT_SIZE}",
    )
    parser.add_argument("--model", help="Model name, e.g. gpt-4o")
    parser.add_argument("--api-key", help="API key; falls back to PROIMAGES_API_KEY")
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint, e.g. https://api.deepseek.com/v1")
    parser.add_argument("--save-params", help="Also write the resolved parameters as JSON here")
    args = parser.parse_args()

    if not MIN_LUT_SIZE <= args.size <= MAX_LUT_SIZE:
        parser.error(f"--size must be between {MIN_LUT_SIZE} and {MAX_LUT_SIZE}")

    params = _resolve_params(args)
    bake_cube_file(params, args.output, size=args.size)
    print(f"wrote {args.output}: {params.name}")

    if args.cdl_out:
        cc_xml = to_cc_xml(to_cdl(params), cc_id_from_name(params.name), params.description)
        Path(args.cdl_out).write_text(cc_xml, encoding="utf-8")
        print(f"wrote {args.cdl_out}")

    if args.save_params:
        Path(args.save_params).write_text(json.dumps(params.model_dump(), indent=2))
        print(f"wrote {args.save_params}")


if __name__ == "__main__":
    main()
