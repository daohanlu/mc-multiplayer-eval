#!/usr/bin/env python3
"""Serve the human-eval bundle and persist annotator progress to disk.

Standard library only — no pip install, no virtualenv.

    python human-eval/serve.py                 # http://localhost:8080
    python human-eval/serve.py --port 9000 --host 0.0.0.0

Static files are served from this directory. On top of that:

    GET  /api/progress/<task>/<name>   -> saved answers for that annotator
    POST /api/progress/<task>/<name>   <- full answer set (client autosaves)
    GET  /api/annotators               -> who has started what, and how far

Answers land in ``responses/<task>__<name>.json``. Every POST rewrites that
file atomically, so a crashed browser, a closed tab, or a different machine all
resume from the same place — the annotator just re-enters the same name.

The pages also mirror answers into ``localStorage``, so the bundle still works
if you serve it with ``python -m http.server`` (no saving, but no data loss
within a browser either).

``--serve-key`` is off by default: the answer-key JSONs stay unreadable over
HTTP so a curious annotator cannot look up which model they are grading.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent
RESPONSES = HERE / "responses"

VALID_TASK = {"consistency", "artifacts"}
# Annotator names become filenames; keep them boring.
NAME_RE = re.compile(r"^[A-Za-z0-9 _.\-]{1,64}$")
MAX_BODY = 8 * 1024 * 1024

SERVE_KEY = False


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("_").lower()


def response_path(task: str, name: str) -> Path:
    return RESPONSES / f"{task}__{slugify(name)}.json"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    # --- helpers ----------------------------------------------------------

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _parse_api_path(self) -> tuple[str, str] | None:
        parts = [p for p in self.path.split("?")[0].split("/") if p]
        if len(parts) != 4 or parts[0] != "api" or parts[1] != "progress":
            return None
        task, name = parts[2], unquote(parts[3])
        if task not in VALID_TASK or not NAME_RE.match(name):
            return None
        return task, name

    # --- routes -----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        clean = self.path.split("?")[0]

        if clean == "/api/annotators":
            return self._send_json({"annotators": self._list_annotators()})

        parsed = self._parse_api_path()
        if parsed:
            task, name = parsed
            path = response_path(task, name)
            if not path.exists():
                return self._send_json({"answers": {}, "found": False})
            try:
                return self._send_json({**json.loads(path.read_text()),
                                        "found": True})
            except json.JSONDecodeError:
                return self._send_json({"error": "corrupt save file"},
                                       HTTPStatus.INTERNAL_SERVER_ERROR)

        if clean.startswith("/api/"):
            return self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        # Don't hand the answer key to the browser.
        if not SERVE_KEY and clean.endswith("_key.json"):
            return self._send_json(
                {"error": "answer key is not served; run with --serve-key "
                          "if you really want this"},
                HTTPStatus.FORBIDDEN,
            )

        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = self._parse_api_path()
        if not parsed:
            return self._send_json({"error": "bad path"}, HTTPStatus.NOT_FOUND)
        task, name = parsed

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._send_json({"error": "bad length"},
                                   HTTPStatus.BAD_REQUEST)
        if length <= 0 or length > MAX_BODY:
            return self._send_json({"error": "bad length"},
                                   HTTPStatus.BAD_REQUEST)

        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            return self._send_json({"error": "bad json"},
                                   HTTPStatus.BAD_REQUEST)
        if not isinstance(payload.get("answers"), dict):
            return self._send_json({"error": "missing answers"},
                                   HTTPStatus.BAD_REQUEST)

        payload["task"] = task
        payload["annotator"] = name
        self._atomic_write(response_path(task, name), payload)
        return self._send_json({"ok": True,
                                "answered": len(payload["answers"])})

    # --- internals --------------------------------------------------------

    @staticmethod
    def _atomic_write(path: Path, payload: dict) -> None:
        RESPONSES.mkdir(exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(RESPONSES), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    @staticmethod
    def _list_annotators() -> list[dict]:
        out = []
        for path in sorted(RESPONSES.glob("*.json")):
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            out.append({
                "task": data.get("task"),
                "annotator": data.get("annotator"),
                "answered": len(data.get("answers", {})),
                "total": data.get("total"),
                "updated": data.get("updated"),
            })
        return out

    def log_message(self, fmt: str, *args) -> None:
        # Keep the console readable: only surface API traffic and errors.
        if "/api/" in self.path or not args or str(args[1]).startswith(("4", "5")):
            super().log_message(fmt, *args)


def main() -> None:
    global SERVE_KEY
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1",
                        help="Use 0.0.0.0 to accept connections from other "
                             "machines. Default: localhost only.")
    parser.add_argument("--serve-key", action="store_true",
                        help="Also serve data/*_key.json over HTTP (unblinds "
                             "the task; off by default).")
    args = parser.parse_args()
    SERVE_KEY = args.serve_key

    if not (HERE / "data" / "consistency_items.json").exists():
        raise SystemExit("data/ is empty — run: python human-eval/build_human_eval.py")

    RESPONSES.mkdir(exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"human-eval serving at http://{args.host}:{args.port}")
    print(f"saving answers to {RESPONSES}")
    if args.serve_key:
        print("WARNING: --serve-key is on; the task is no longer blind")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
