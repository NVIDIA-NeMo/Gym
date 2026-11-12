# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from mmrl/render.py in NVIDIA's visual-coding RL research code. Candidate execution is remote-only.
"""Run only INSIDE a disposable execution sandbox, never on the trainer host."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


LATEX_PACKAGES = (
    "amsmath",
    "amssymb",
    "amsfonts",
    "graphicx",
    "xcolor",
    "algorithm",
    "listings",
    "stfloats",
    "epstopdf",
    "pgfplots",
    "tikz",
    "tikz-cd",
    "tikz-qtree",
    "tikz-dependency",
    "tikz-3dplot",
    "tikz-network",
    "threeparttable",
    "adjustbox",
    "booktabs",
    "multirow",
    "array",
    "tcolorbox",
)
LATEX_ALGORITHM_PACKAGES = ("algorithmic", "algpseudocode")


def latex_document(body):
    """Supply the promised body preamble without repairing authored LaTeX."""
    # Preserve explicit documents, including malformed ones, for the compiler.
    active = re.sub(r"(?<!\\)%[^\n]*", "", body)
    if re.search(r"\\documentclass\b|\\(?:begin|end)\s*\{document\}", active):
        return body
    # These are two established package dialects, not spelling corrections.
    algorithm_package = (
        "algorithmic"
        if re.search(r"\\(?:STATE|IF|FOR|WHILE|REQUIRE|ENSURE|RETURN|REPEAT|LOOP)\b", active)
        else "algpseudocode"
    )
    imports = "\n".join(r"\usepackage{" + package + "}" for package in (*LATEX_PACKAGES, algorithm_package))
    return "\\documentclass{article}\n" + imports + "\n\\begin{document}\n" + body + "\n\\end{document}\n"


def render_chart(source, output):
    """Preserve explicitly saved figures, including saves followed by close/clear."""
    import io
    import runpy

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib._pylab_helpers import Gcf
    from matplotlib.figure import Figure

    created = []
    last_saved = None
    original_init, original_savefig, original_show = Figure.__init__, Figure.savefig, plt.show

    def track_figure(figure, *args, **kwargs):
        original_init(figure, *args, **kwargs)
        created.append(figure)

    def capture_save(figure, *args, **kwargs):
        nonlocal last_saved
        result = original_savefig(figure, *args, **kwargs)
        image = io.BytesIO()
        original_savefig(figure, image, format="png", dpi=120, bbox_inches="tight")
        last_saved = image.getvalue()
        return result

    Figure.__init__, Figure.savefig = track_figure, capture_save
    plt.show = lambda *args, **kwargs: None
    try:
        runpy.run_path(str(source), run_name="__main__")
    finally:
        Figure.__init__, Figure.savefig, plt.show = original_init, original_savefig, original_show

    if last_saved is not None:
        Path(output).write_bytes(last_saved)
    else:
        manager = Gcf.get_active()
        figure = manager.canvas.figure if manager is not None else (created[-1] if created else None)
        if figure is None:
            raise ValueError("Chart code produced no matplotlib figure")
        original_savefig(figure, output, format="png", dpi=120, bbox_inches="tight")


def check_runtime():
    import cairosvg  # noqa: F401
    import matplotlib  # noqa: F401
    import PIL  # noqa: F401
    from playwright.sync_api import sync_playwright

    for name in ("pdflatex", "pdftoppm", "kpsewhich"):
        if not shutil.which(name):
            raise RuntimeError(f"Renderer dependency missing: {name}")
    for package in (*LATEX_PACKAGES, *LATEX_ALGORITHM_PACKAGES):
        found = subprocess.run(
            ["kpsewhich", package + ".sty"], capture_output=True, text=True, check=False, timeout=10
        )
        if found.returncode or not found.stdout.strip():
            raise RuntimeError(f"Renderer TeX dependency missing: {package}")
    with sync_playwright() as pw:
        if not Path(pw.chromium.executable_path).is_file():
            raise RuntimeError("Chromium is not installed")


def render(benchmark, source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    env = os.environ | {"MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1"}
    if benchmark == "chart2code":
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--chart", str(source), str(output)],
            cwd=source.parent,
            env=env,
            check=True,
            timeout=120,
        )
    elif benchmark in {"mmsvg_img2svg", "mmsvg_text2svg"}:
        import cairosvg

        cairosvg.svg2png(url=str(source), write_to=str(output), unsafe=False)
    elif benchmark == "design2code_full":
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page(viewport={"width": 1280, "height": 960}, device_scale_factor=1)
            # Reproducible previews: no remote assets, beacons or downloads.
            page.route(
                "**/*",
                lambda route: (
                    route.continue_() if route.request.url.startswith(("file:", "data:", "blob:")) else route.abort()
                ),
            )
            page.goto(source.as_uri(), wait_until="load", timeout=30000)
            page.screenshot(path=str(output), full_page=True, timeout=30000)
            browser.close()
    elif benchmark == "image2struct_latex":
        body = source.read_text()
        document = latex_document(body)
        if document != body:
            source = source.with_name("mmrl-document.tex")
            source.write_text(document)
        subprocess.run(
            [
                "pdflatex",
                "-no-shell-escape",
                "-interaction=nonstopmode",
                "-halt-on-error",
                str(source),
            ],
            cwd=source.parent,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=120,
        )
        subprocess.run(
            [
                "pdftoppm",
                "-f",
                "1",
                "-singlefile",
                "-png",
                "-r",
                "120",
                str(source.with_suffix(".pdf")),
                str(output.with_suffix("")),
            ],
            cwd=source.parent,
            check=True,
            timeout=60,
        )
    else:
        raise ValueError("Unsupported benchmark")
    from PIL import Image

    with Image.open(output) as im:
        if im.width < 1 or im.height < 1:
            raise ValueError("Empty render")
        im.verify()


if __name__ == "__main__":
    if sys.argv[1:] == ["--check"]:
        check_runtime()
    elif sys.argv[1:2] == ["--chart"]:
        render_chart(*sys.argv[2:])
    else:
        render(*sys.argv[1:])
