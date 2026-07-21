import argparse

from proimages.cli_args import add_device_argument
from proimages.core.pipeline import process_image
from proimages.core.system.io import load_image, save_image
from proimages.gpu_config import detect_device


def main() -> None:
    parser = argparse.ArgumentParser(prog="proimages", description="Process a photo through the ProImages pipeline")
    parser.add_argument("input", help="Path to the input photo (RAW or JPEG/PNG/HEIC)")
    parser.add_argument("output", help="Path to write the processed photo")
    parser.add_argument("--lut", default=None, help="Path to a .cube LUT file to apply")
    add_device_argument(parser)
    args = parser.parse_args()

    device = detect_device(override=args.device)
    print(f"using device: {device}")

    image = load_image(args.input)
    result = process_image(image, lut_path=args.lut)
    save_image(result, args.output)


if __name__ == "__main__":
    main()
