# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Dev tool: materialize the GDPVal ``images_and_text`` judge conversion to disk.

Runs the SAME code path an image-only local VLM judge (e.g. self-hosted
MiniMax-M3) receives — :func:`resources_servers.gdpval.comparison.build_file_section`
with ``media_mode="images_and_text"`` — and writes every produced content block
out as a file so you can compare the ORIGINAL deliverable (PDF / docx / xlsx /
csv / image) against the rasterized page PNGs and the extracted text the judge
actually sees.

Usage (from the Gym repo root, with a venv that has PyMuPDF + pdfminer.six +
openai — e.g. the resources_servers/gdpval venv or the root .venv):

    python -m resources_servers.gdpval.inspect_conversion \
        --input /path/to/task_dir_or_single_file \
        --out   ./conversion_preview \
        --dpi 144 --max-pages 30

For a directory it converts every deliverable (mirrors a real judge request);
for a single file it converts just that file. Office docs (.docx/.pptx/.xlsx)
are only rasterized if a sibling ``<stem>.pdf`` already exists (the resources
server pre-converts those via LibreOffice at eval time); otherwise they fall
back to the ``[no PDF render available]`` marker, which this tool reports.

Output layout (per deliverable ``<name>``)::

    <out>/<name>/text_01.txt        # extracted-text / marker blocks, in order
    <out>/<name>/page_001.png       # rasterized page images, in order
    <out>/<name>/MANIFEST.txt       # ordered block list + byte sizes
    <out>/SUMMARY.txt               # one line per deliverable
"""

from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path


def _decode_data_url(url: str) -> tuple[str, bytes] | None:
    """Return ``(mime, raw_bytes)`` for a ``data:<mime>;base64,<...>`` URL, else None."""
    if not url.startswith("data:"):
        return None
    try:
        header, b64 = url.split(",", 1)
    except ValueError:
        return None
    mime = header[len("data:") :].split(";", 1)[0] or "application/octet-stream"
    try:
        return mime, base64.b64decode(b64)
    except Exception:
        return None


_MIME_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "application/pdf": ".pdf",
    "audio/wav": ".wav",
    "audio/mp3": ".mp3",
    "video/mp4": ".mp4",
}


def _write_blocks(name: str, blocks: list[dict], out_dir: Path) -> dict:
    """Write one deliverable's content blocks to ``out_dir/<name>/``. Returns stats."""
    dest = out_dir / _safe(name)
    dest.mkdir(parents=True, exist_ok=True)

    manifest: list[str] = [f"deliverable: {name}", f"total_blocks: {len(blocks)}", ""]
    n_text = n_img = n_media = 0
    text_chars = 0

    for i, block in enumerate(blocks):
        btype = block.get("type")
        if btype == "text":
            n_text += 1
            text = block.get("text", "")
            text_chars += len(text)
            fname = f"text_{n_text:02d}.txt"
            (dest / fname).write_text(text, encoding="utf-8")
            preview = text.replace("\n", " ")[:80]
            manifest.append(f"[{i:03d}] text  -> {fname}  ({len(text)} chars)  {preview!r}")
        elif btype == "image_url":
            url = (block.get("image_url") or {}).get("url", "")
            decoded = _decode_data_url(url)
            if decoded is None:
                manifest.append(f"[{i:03d}] image_url -> (non-data URL or undecodable): {url[:60]}")
                continue
            mime, raw = decoded
            ext = _MIME_EXT.get(mime, ".bin")
            if mime == "image/png":
                n_img += 1
                fname = f"page_{n_img:03d}{ext}"
            else:
                n_media += 1
                fname = f"media_{n_media:02d}{ext}"
            (dest / fname).write_bytes(raw)
            manifest.append(f"[{i:03d}] image_url ({mime}) -> {fname}  ({len(raw):,} bytes)")
        else:
            manifest.append(f"[{i:03d}] {btype!r} (unhandled)")

    (dest / "MANIFEST.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    return {
        "name": name,
        "dir": dest,
        "blocks": len(blocks),
        "text_blocks": n_text,
        "page_images": n_img,
        "media_blocks": n_media,
        "text_chars": text_chars,
    }


def _safe(name: str) -> str:
    """Directory-safe deliverable name: drop the file extension and turn spaces
    (and any other filesystem-unfriendly character) into underscores.

    e.g. ``"Q3 Sales Report.pdf" -> "Q3_Sales_Report"``.
    """
    stem = Path(name).stem  # strips the trailing extension: "a.pdf" -> "a"
    cleaned = "".join(c if (c.isalnum() or c in "._-") else "_" for c in stem)
    return cleaned or "deliverable"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="A deliverable file OR a task directory of deliverables.")
    ap.add_argument("--out", default="./conversion_preview", help="Output directory (default ./conversion_preview).")
    ap.add_argument("--dpi", type=int, default=144, help="PDF rasterization DPI (default 144).")
    ap.add_argument("--max-pages", type=int, default=30, help="Max pages per PDF to render (default 30).")
    ap.add_argument("--no-text", action="store_true", help="Skip the extracted-text block (images only).")
    ap.add_argument(
        "--av-capable",
        action="store_true",
        help="Shorthand: judge reads BOTH audio and video (pass both through as native media blocks).",
    )
    ap.add_argument(
        "--audio-capable",
        action="store_true",
        help="Judge reads audio (pass audio through). MiniMax-M3 does NOT — leave off to see audio stubbed.",
    )
    ap.add_argument(
        "--video-capable",
        action="store_true",
        help="Judge reads video (pass video through). MiniMax-M3 DOES read video.",
    )
    args = ap.parse_args()
    audio_capable = args.av_capable or args.audio_capable
    video_capable = args.av_capable or args.video_capable

    # Import the REAL judge conversion code (needs the Gym repo root on sys.path).
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from resources_servers.gdpval.comparison import (  # noqa: E402
        build_file_section,
        get_file_image_text_blocks,
        IGNORE_FILES,
    )

    in_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"ERROR: input does not exist: {in_path}", file=sys.stderr)
        return 2

    include_text = not args.no_text
    results: list[dict] = []

    if in_path.is_dir():
        # Directory → exactly what a judge request builds for this task's files.
        section = build_file_section(
            str(in_path),
            media_mode="images_and_text",
            render_dpi=args.dpi,
            max_pages=args.max_pages,
            include_text=include_text,
            audio_capable=audio_capable,
            video_capable=video_capable,
        )
        # build_file_section prefixes each file with a "\n<name>:\n" text marker;
        # split the flat block list back into per-deliverable groups on those.
        groups: list[tuple[str, list[dict]]] = []
        current_name = "_preamble"
        current: list[dict] = []
        for block in section:
            text = block.get("text", "") if block.get("type") == "text" else ""
            stripped = text.strip()
            if block.get("type") == "text" and stripped.endswith(":") and "\n" in text:
                if current:
                    groups.append((current_name, current))
                current_name = stripped.rstrip(":").strip()
                current = []
            else:
                current.append(block)
        if current:
            groups.append((current_name, current))

        for name, blocks in groups:
            if not blocks:
                continue
            results.append(_write_blocks(name, blocks, out_dir))
    else:
        # Single file.
        if in_path.name in IGNORE_FILES:
            print(f"NOTE: {in_path.name} is in IGNORE_FILES and would be skipped by the judge.")
        blocks = get_file_image_text_blocks(
            str(in_path.parent),
            in_path.name,
            render_dpi=args.dpi,
            max_pages=args.max_pages,
            include_text=include_text,
            audio_capable=audio_capable,
            video_capable=video_capable,
        )
        results.append(_write_blocks(in_path.name, blocks, out_dir))

    # Summary.
    summary_lines = [
        f"input:  {in_path}",
        f"output: {out_dir}",
        f"dpi={args.dpi}  max_pages={args.max_pages}  include_text={include_text}  "
        f"audio_capable={audio_capable}  video_capable={video_capable}",
        "",
        f"{'deliverable':50s} {'blocks':>6} {'pages':>6} {'txtblk':>6} {'media':>6} {'txtchars':>9}",
        "-" * 90,
    ]
    for r in results:
        summary_lines.append(
            f"{r['name'][:50]:50s} {r['blocks']:6d} {r['page_images']:6d} "
            f"{r['text_blocks']:6d} {r['media_blocks']:6d} {r['text_chars']:9d}"
        )
    summary = "\n".join(summary_lines) + "\n"
    (out_dir / "SUMMARY.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Wrote per-deliverable output under: {out_dir}")
    print("Open the original file alongside <out>/<name>/page_*.png and text_*.txt to compare.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
