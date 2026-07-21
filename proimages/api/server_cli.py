import argparse
import os

import uvicorn

from proimages.cli_args import add_device_argument


def main() -> None:
    parser = argparse.ArgumentParser(prog="proimages-api", description="Run the ProImages FastAPI server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    add_device_argument(parser)
    args = parser.parse_args()

    if args.device is not None:
        os.environ["PROIMAGES_DEVICE"] = args.device

    uvicorn.run("proimages.api.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
