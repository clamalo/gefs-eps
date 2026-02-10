import asyncio
import os
import re
import sys
import tempfile

import aiohttp
from aiohttp import web

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from src.ingest.ingest import download_records_concat


_RANGE_RE = re.compile(r"^bytes=(\d+)-(\d+)?$")


async def _range_handler(request: web.Request) -> web.Response:
    blob: bytes = request.app["blob"]
    delay_by_start = request.app["delay_by_start"]

    range_header = request.headers.get("Range")
    if not range_header:
        return web.Response(body=blob, status=200)

    match = _RANGE_RE.match(range_header)
    if not match:
        return web.Response(text="bad range", status=400)

    start = int(match.group(1))
    end_str = match.group(2)
    end = int(end_str) if end_str is not None else (len(blob) - 1)

    # Force out-of-order completion.
    await asyncio.sleep(delay_by_start.get(start, 0.0))

    if start < 0 or start >= len(blob) or end < start:
        return web.Response(status=416)

    end = min(end, len(blob) - 1)
    body = blob[start : end + 1]
    headers = {"Content-Range": f"bytes {start}-{end}/{len(blob)}"}
    return web.Response(body=body, status=206, headers=headers)


async def main() -> None:
    # Three chunks; the server will delay them so they finish out of order.
    blob = b"AAA" + b"BBBB" + b"CC" + b"DDDDDD"
    delays = {0: 0.3, 3: 0.1, 7: 0.2}
    records = [
        {"url": "", "start": 0, "end": 2},
        {"url": "", "start": 3, "end": 6},
        {"url": "", "start": 7, "end": 8},
    ]

    app = web.Application()
    app["blob"] = blob
    app["delay_by_start"] = delays
    app.router.add_get("/blob", _range_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()

    try:
        port = site._server.sockets[0].getsockname()[1]
        url = f"http://127.0.0.1:{port}/blob"
        for r in records:
            r["url"] = url

        with tempfile.TemporaryDirectory() as td:
            out_path = os.path.join(td, "out.bin")
            sem = asyncio.Semaphore(3)
            async with aiohttp.ClientSession() as session:
                await download_records_concat(
                    session,
                    records,
                    out_path,
                    sem,
                    description="selftest",
                    max_concurrency=3,
                )

            with open(out_path, "rb") as f:
                got = f.read()

        expected = b"AAABBBBCC"
        if got != expected:
            raise SystemExit(f"selftest failed: expected {expected!r}, got {got!r}")
    finally:
        await runner.cleanup()

    print("selftest passed")


if __name__ == "__main__":
    asyncio.run(main())
