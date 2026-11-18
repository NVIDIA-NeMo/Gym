# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2026 Vision2Code Authors
# SPDX-License-Identifier: Apache-2.0
# Save handling adapted from vision2code/rendering/render_python.py (Apache-2.0):
# https://github.com/image2code/vision2code/blob/000f2e162dd62fac5637c718e018d7123b3aff1e/vision2code/rendering/render_python.py
"""Trusted entrypoint inside the renderer; never run submitted code on the host."""

import os
import runpy


def _configure_matplotlib_output(*, output_path: str, width: int, height: int) -> None:
    """Route pyplot and Figure saves through the benchmark's fixed-size image output."""
    from matplotlib.figure import Figure

    original_savefig = Figure.savefig
    dpi = 100

    def savefig(figure: Figure, *args: object, **kwargs: object) -> None:
        figure.set_size_inches(width / dpi, height / dpi, forward=True)
        for name in ("fname", "filename", "filename_or_obj"):
            kwargs.pop(name, None)
        kwargs.update(dpi=dpi, bbox_inches=None, pad_inches=0)
        original_savefig(figure, output_path, *args[1:], **kwargs)

    # pyplot.savefig delegates to Figure.savefig, so both public entrypoints use this handler.
    Figure.savefig = savefig


if __name__ == "__main__":
    # Keep program diagnostics on stdout, away from Enroot's stderr classifier.
    # Redirect the descriptor too, so native extensions and children follow it.
    # Enroot and unshare retain their separate transport stderr outside this process.
    os.dup2(1, 2)
    os.makedirs("/workspace/.tmp", exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    os.chdir("/workspace")
    width = int(os.environ["TARGET_WIDTH"])
    height = int(os.environ["TARGET_HEIGHT"])
    _configure_matplotlib_output(output_path=os.environ["OUTPUT_PATH"], width=width, height=height)
    runpy.run_path(
        "/workspace/code.py",
        run_name="__main__",
        init_globals={
            "OUTPUT_PATH": os.environ["OUTPUT_PATH"],
            "TARGET_WIDTH": width,
            "TARGET_HEIGHT": height,
        },
    )
