"""CLI: ``vidgrep-ml demo|serve``."""

from __future__ import annotations

import argparse
import sys

from .demo import demo_video
from .service import IndexService


def _demo(queries: list[str]) -> int:
    svc = IndexService()
    frames = demo_video("demo")
    svc.index_frames(frames)
    print("=" * 70)
    print(f"vidgrep-ml demo — embedder={svc.embedder.name}  indexed {len(frames)} frames")
    print("=" * 70)
    if not queries:
        queries = [
            "person in a red jacket",
            "someone with a backpack",
            "a car turning left",
            "night time traffic",
        ]
    for q in queries:
        print(f"\n  query: \"{q}\"")
        for r in svc.search("demo", q, top_k=3):
            ts = r.timestamp_ms / 1000
            print(f"    {ts:5.1f}s  score={r.score:.3f}  labels={r.labels}  box={r.box}")
    return 0


def _serve(host: str, port: int) -> int:
    try:
        import uvicorn  # type: ignore
    except ImportError:
        print("Install server extras: pip install 'vidgrep-ml[server]'", file=sys.stderr)
        return 1
    uvicorn.run("vidgrep_ml.server:app", host=host, port=port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vidgrep-ml")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="index a synthetic video and run text searches offline")
    d.add_argument("queries", nargs="*")
    s = sub.add_parser("serve", help="run the FastAPI service")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    args = p.parse_args(argv)
    if args.cmd == "demo":
        return _demo(args.queries)
    return _serve(args.host, args.port)


if __name__ == "__main__":
    raise SystemExit(main())
