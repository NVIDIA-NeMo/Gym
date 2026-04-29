# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare Answer-Judge benchmark data for NeMo Gym."""

from __future__ import annotations

import json
from pathlib import Path


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "answer-judge_benchmark.jsonl"

ROWS = [
    {
        "problem": "A parabola with equation $y=ax^2+bx+c$ has a vertical line of symmetry at $x=1$ and goes through the two points $(-1,3)$ and $(2,-2)$. The quadratic $ax^2 + bx +c$  has two real roots. The greater root is $\\sqrt{n}+1$. What is $n$?",
        "expected_answer": "2.2",
        "predicted_answer": "\\frac{11}{5}",
        "expected_judgement": "Judgement: Yes",
        "comment": "Exactly matching, but different notation",
    },
    {
        "problem": "Write $\\cfrac{\\cfrac{3}{8}+\\cfrac{7}{8}}{\\cfrac{4}{5}}$ as a simplified fraction.",
        "expected_answer": "\\cfrac{25}{16}",
        "predicted_answer": "\\frac{25}{16}",
        "expected_judgement": "Judgement: Yes",
        "comment": "Exactly matching, but different notation",
    },
    {
        "problem": "Paula invests $\\$10,\\!000$ at the start of a 5 year period at an interest rate of $10\\%$. At the end of those 5 years, how much is her investment worth if the interest is compounded quarterly?  Express your answer rounded to the nearest cent.",
        "expected_answer": "16,386.16",
        "predicted_answer": "16384.60",
        "expected_judgement": "Judgement: No",
        "comment": "Different values",
    },
    {
        "problem": "Find the domain of the real-valued function \\[f(x)=\\sqrt{-6x^2+11x-4}.\\] Give the endpoints in your answer as common fractions (not mixed numbers or decimals).",
        "expected_answer": "[\\frac{1}{2},\\frac{4}{3}]",
        "predicted_answer": "\\left[\\frac{4}{3},\\frac{1}{2}\\right]",
        "expected_judgement": "Judgement: No",
        "comment": "Order matters, so not equivalent.",
    },
    {
        "problem": "A line segment of length $5$ has one endpoint at $(1, 2)$ and the other endpoint at $(4, b)$. Find all possible values of $b$, separated by commas.",
        "expected_answer": "6,-2",
        "predicted_answer": "-2, 6",
        "expected_judgement": "Judgement: Yes",
        "comment": "Order does not matter, so equivalent.",
    },
    {
        "problem": "For how many integers $n$ does the expression\\[\\sqrt{\\frac{\\log (n^2) - (\\log n)^2}{\\log n - 3}}\\]represent a real number, where log denotes the base $10$ logarithm?",
        "expected_answer": "901.0",
        "predicted_answer": "902",
        "expected_judgement": "Judgement: No",
        "comment": "Different values",
    },
    {
        "problem": '$ABCDEFGH$ is a regular octagon of side 12cm. Find the area in square centimeters of trapezoid $BCDE$. Express your answer in simplest radical form.\n\n[asy] real x = 22.5; draw(dir(0+x)--dir(45+x)--dir(90+x)--dir(90+45+x)-- dir(90+2*45+x)--dir(90+3*45+x)-- dir(90+4*45+x)-- dir(90+5*45+x)--dir(90+6*45+x));\n\nlabel("$A$", dir(90+45+x), W); label("$B$", dir(90+x), NW);label("$C$", dir(45+x), NE); label("$D$", dir(x), E);label("$E$", dir(90+5*45+x), E);label("$F$", dir(90+4*45+x), SE); label("$G$", dir(90+3*45+x), SW);label("$H$", dir(90+2*45+x), W);\ndraw( dir(90+x)--dir(90+5*45+x) );\n[/asy]',
        "expected_answer": "72+72\\sqrt{2}",
        "predicted_answer": "144(1 + \\sqrt{2})",
        "expected_judgement": "Judgement: No",
        "comment": "Different values",
    },
    {
        "problem": "Find all solutions to\n\\[\\sin \\left( \\tan^{-1} (x) + \\cot^{-1} \\left( \\frac{1}{x} \\right) \\right) = \\frac{1}{3}.\\]Enter all the solutions, separated by commas.",
        "expected_answer": "3\\pm2\\sqrt{2}",
        "predicted_answer": "3 + 2\\sqrt{2},\\ 3 - 2\\sqrt{2},\\ -3 + 2\\sqrt{2},\\ -3 - 2\\sqrt{2}",
        "expected_judgement": "Judgement: No",
        "comment": "Expected answer has 2 values, while predicted has 4",
    },
    {
        "problem": "A line is parameterized by a parameter $t,$ so that the vector on the line at $t = -2$ is $\\begin{pmatrix} 2 \\\\ -4 \\end{pmatrix},$ and the vector on the line at $t = 3$ is $\\begin{pmatrix} 1 \\\\ 7 \\end{pmatrix}.$  Find the vector on the line at $t = 5.$",
        "expected_answer": "\\begin{pmatrix}3/5\\\\57/5\\end{pmatrix}",
        "predicted_answer": "\\begin{pmatrix} \\frac{3}{5} \\\\ \\frac{57}{5} \\end{pmatrix}",
        "expected_judgement": "Judgement: Yes",
        "comment": "Same values, different notation",
    },
    {
        "problem": 'Let $\\mathbf{a} = \\begin{pmatrix} 7 \\\\ - 1 \\\\ 4 \\end{pmatrix}$ and $\\mathbf{b} = \\begin{pmatrix} 3 \\\\ 1 \\\\ 2 \\end{pmatrix}.$  Find the vector $\\mathbf{c}$ so that $\\mathbf{a},$ $\\mathbf{b},$ and $\\mathbf{c}$ are collinear, and $\\mathbf{b}$ bisects the angle between $\\mathbf{a}$ and $\\mathbf{c}.$\n\n[asy]\nunitsize(0.5 cm);\n\npair A, B, C, O;\n\nA = (-2,5);\nB = (1,3);\nO = (0,0);\nC = extension(O, reflect(O,B)*(A), A, B);\n\ndraw(O--A,Arrow(6));\ndraw(O--B,Arrow(6));\ndraw(O--C,Arrow(6));\ndraw(interp(A,C,-0.1)--interp(A,C,1.1),dashed);\n\nlabel("$\\\\mathbf{a}$", A, NE);\nlabel("$\\\\mathbf{b}$", B, NE);\nlabel("$\\\\mathbf{c}$", C, NE);\n[/asy]',
        "expected_answer": "\\begin{pmatrix}5/3\\\\5/3\\\\4/3\\end{pmatrix}",
        "predicted_answer": "\\begin{pmatrix} 5 \\\\ 5 \\\\ 4 \\end{pmatrix}",
        "expected_judgement": "Judgement: Yes",
        "comment": "Only the direction of the vector matters, so these two answers are equivalent.",
    },
    {
        "problem": "Find all positive integer $ m$ if there exists prime number $ p$ such that $ n^m\\minus{}m$ can not be divided by $ p$ for any integer $ n$.",
        "expected_answer": "m \\neq 1",
        "predicted_answer": "m \\geq 2",
        "expected_judgement": "Judgement: Yes",
        "comment": "Since problem asks for positive integers, these two equations are equivalent",
    },
    {
        "problem": "Find all triplets of integers $(a,b,c)$ such that the number\n\\[N = \\frac{(a-b)(b-c)(c-a)}{2} + 2\\]\nis a power of $2016$ .\n(A power of $2016$ is an integer of form $2016^n$ ,where $n$ is a non-negative integer.)",
        "expected_answer": "\\[\n(a, b, c) = (k, k+1, k+2) \\quad \\text{and all cyclic permutations, with } k \\in \\mathbb{Z}\n\\]",
        "predicted_answer": "(a, b, c) \\text{ such that } (a - b, b - c, c - a) \\text{ is } (1, 1, -2), (-2, 1, 1), \\text{ or } (1, -2, 1)",
        "expected_judgement": "Judgement: Yes",
        "comment": "The predicted answer lists the differences between the integers, while the expected answer provides a specific form for the integers. The differences correspond to the cyclic permutations of (k, k+1, k+2).",
    },
]


def prepare() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FPATH.open("w", encoding="utf-8") as fout:
        for row in ROWS:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(ROWS)} problems to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
