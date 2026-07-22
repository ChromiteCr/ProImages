import argparse
import json
import os
import sys
from pathlib import Path

from proimages.core.lut_gen.bake import DEFAULT_LUT_SIZE, bake_cube_file
from proimages.core.lut_gen.params import GradeParams
from proimages.core.system.io import load_image


def _resolve_params(args: argparse.Namespace) -> GradeParams:
    if args.params:
        return GradeParams.model_validate_json(Path(args.params).read_text())

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
    parser.add_argument("--size", type=int, default=DEFAULT_LUT_SIZE, help="LUT cube size per edge")
    parser.add_argument("--model", help="Model name, e.g. gpt-4o")
    parser.add_argument("--api-key", help="API key; falls back to PROIMAGES_API_KEY")
    parser.add_argument("--base-url", help="OpenAI-compatible endpoint, e.g. https://api.deepseek.com/v1")
    parser.add_argument("--save-params", help="Also write the resolved parameters as JSON here")
    args = parser.parse_args()

    params = _resolve_params(args)
    bake_cube_file(params, args.output, size=args.size)
    print(f"wrote {args.output}: {params.name}")

    if args.save_params:
        Path(args.save_params).write_text(json.dumps(params.model_dump(), indent=2))
        print(f"wrote {args.save_params}")


if __name__ == "__main__":
    main()
