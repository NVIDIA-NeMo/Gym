#!/usr/bin/env python3
"""Create, probe, screenshot, and delete cell-2 OSWorld sandboxes."""

import argparse
import asyncio
import io
import os
import time
from pathlib import Path

import aiohttp
from PIL import Image

from nemo_gym.sandbox import AsyncSandbox, SandboxSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of sandboxes to validate concurrently (cell-2 currently supports 4)",
    )
    parser.add_argument(
        "--screenshot-dir",
        type=Path,
        default=None,
        help="Optionally save each screenshot for visual inspection",
    )
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1")
    return args


async def wait_for_guest_ready(
    session: aiohttp.ClientSession,
    base_url: str,
    timeout_s: float = 600,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            async with session.get(
                f"{base_url}/platform",
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status == 200:
                    return
                last_error = RuntimeError(
                    f"guest readiness returned HTTP {response.status}: "
                    f"{(await response.text())[:200]}"
                )
        except (aiohttp.ClientError, asyncio.TimeoutError) as error:
            last_error = error
        await asyncio.sleep(5)
    raise TimeoutError(f"guest did not become ready within {timeout_s}s") from last_error


async def wait_for_desktop_ready(
    session: aiohttp.ClientSession,
    base_url: str,
    timeout_s: float = 600,
    min_nonblack_ratio: float = 0.02,
) -> tuple[bytes, float, tuple[int, int]]:
    deadline = time.monotonic() + timeout_s
    last_ratio = 0.0
    while time.monotonic() < deadline:
        try:
            async with session.get(
                f"{base_url}/screenshot",
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status != 200:
                    await response.read()
                else:
                    screenshot = await response.read()
                    if screenshot.startswith(b"\x89PNG\r\n\x1a\n"):
                        with Image.open(io.BytesIO(screenshot)) as image:
                            grayscale = image.convert("L")
                            histogram = grayscale.histogram()
                            total_pixels = grayscale.width * grayscale.height
                            last_ratio = 1.0 - sum(histogram[:8]) / total_pixels
                            size = image.size
                        if last_ratio >= min_nonblack_ratio:
                            return screenshot, last_ratio, size
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
            pass
        await asyncio.sleep(5)
    raise TimeoutError(
        f"desktop did not become non-black within {timeout_s}s; "
        f"last_nonblack_ratio={last_ratio:.6f}"
    )


async def smoke_one(
    index: int,
    provider_config: dict,
    pool_ref: str,
    screenshot_dir: Path | None,
) -> None:
    sandbox = AsyncSandbox(
        provider_config,
        SandboxSpec(
            ttl_s=900,
            metadata={
                "purpose": "osworld-cell2-smoke",
                "smoke_index": str(index),
            },
            provider_options={"extensions": {"poolRef": pool_ref}},
        ),
    )

    await sandbox.start()
    sandbox_id = sandbox._require_handle().sandbox_id
    print(f"[{index}] created sandbox={sandbox_id}")
    try:
        endpoint = await sandbox.endpoint(5000)
        headers = dict(endpoint.headers)
        async with aiohttp.ClientSession(headers=headers) as session:
            base_url = endpoint.url.rstrip("/")
            await wait_for_guest_ready(session, base_url)
            print(f"[{index}] guest HTTP ready")
            async with session.post(
                f"{base_url}/execute",
                json={
                    "command": ["bash", "-lc", f"echo cell2-ok-{index}"],
                    "shell": False,
                },
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                response.raise_for_status()
                payload = await response.json()
                print(
                    f"[{index}] execute returncode={payload.get('returncode')} "
                    f"output={payload.get('output', '').strip()!r}"
                )

            screenshot, nonblack_ratio, size = await wait_for_desktop_ready(
                session, base_url
            )
            print(
                f"[{index}] desktop ready size={size[0]}x{size[1]} "
                f"nonblack_ratio={nonblack_ratio:.4f}"
            )
            if screenshot_dir is not None:
                screenshot_path = screenshot_dir / f"cell2-smoke-{index}.png"
                screenshot_path.write_bytes(screenshot)
                print(f"[{index}] saved screenshot={screenshot_path}")
            print(f"[{index}] screenshot bytes={len(screenshot)} content_type=image/png")
    finally:
        await sandbox.stop()
        print(f"[{index}] deleted sandbox={sandbox_id}")


async def main() -> None:
    args = parse_args()
    if args.screenshot_dir is not None:
        args.screenshot_dir.mkdir(parents=True, exist_ok=True)
    domain = os.environ["OPENSANDBOX_DOMAIN"]
    api_key = os.environ.get("OPENSANDBOX_API_KEY") or os.environ.get(
        "OPEN_SANDBOX_API_KEY"
    )
    if not api_key:
        raise RuntimeError(
            "Set OPENSANDBOX_API_KEY (or OPEN_SANDBOX_API_KEY) before calling cell-2"
        )
    pool_ref = os.environ.get("OSWORLD_POOL_REF", "osworld-kvm")
    create_timeout_s = float(os.environ.get("OPENSANDBOX_CREATE_TIMEOUT_S", "300"))
    create_retries = int(os.environ.get("OPENSANDBOX_CREATE_RETRIES", "1"))
    provider_config = {
        "opensandbox": {
            "connection": {
                "domain": domain,
                "api_key": api_key,
                "protocol": "http",
                "request_timeout_s": 300,
                "use_server_proxy": True,
            },
            "create": {
                "request_timeout_s": create_timeout_s,
                "timeout_s": create_timeout_s,
                "skip_health_check": True,
                "retries": create_retries,
            },
            "probe": {"command": None},
        }
    }
    await asyncio.gather(
        *(
            smoke_one(index, provider_config, pool_ref, args.screenshot_dir)
            for index in range(args.concurrency)
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
