"""Command line entry point: ``python -m scenario_search <command>``."""

from __future__ import annotations

import argparse
import json
import logging

from . import config


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="scenario_search")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build index artifacts from the nuScenes download")
    b.add_argument(
        "--steps",
        default="logs,measures,tags,captions,clip",
        help="comma-separated subset of logs,measures,tags,captions,clip",
    )
    s = sub.add_parser("search", help="run one query from the command line")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=10)
    s.add_argument("--mode", default="hybrid")
    s.add_argument("--no-clip", action="store_true")
    v = sub.add_parser("serve", help="start the API + demo UI")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    if args.cmd == "build":
        steps = set(args.steps.split(","))
        if "logs" in steps:
            from .nuscenes_io import build_logs, save_logs

            save_logs(build_logs(config.DATA_DIR, config.VERSION), config.ARTIFACTS_DIR / config.LOGS_FILE)
        if "measures" in steps:
            from .build_measures import build

            build()
        from . import index

        if "tags" in steps:
            index.build_tags()
        if "captions" in steps:
            index.build_caption_embeddings()
        if "clip" in steps:
            index.build_clip_index()
    elif args.cmd == "search":
        from .search import SearchEngine

        eng = SearchEngine(use_clip=not args.no_clip)
        print(json.dumps(eng.search(args.query, k=args.k, mode=args.mode), indent=1))
    elif args.cmd == "serve":
        import uvicorn

        uvicorn.run("scenario_search.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
