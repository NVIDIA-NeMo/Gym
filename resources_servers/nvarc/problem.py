# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Trimmed copy of progressive_learning/arc_agi/problem.py
# Self-contained: only depends on pydantic + stdlib
# Removed: beam.resource, matplotlib, metrics imports, from_file, plot, evaluate, visualize

import random
import re
from typing import Optional

from pydantic import BaseModel, Field


arc_colors = [
    "#000000",  # 0 = black
    "#0074D9",  # 1 = blue
    "#FF4136",  # 2 = red
    "#2ECC40",  # 3 = green
    "#FFDC00",  # 4 = yellow
    "#AAAAAA",  # 5 = gray
    "#F012BE",  # 6 = magenta
    "#FF851B",  # 7 = orange
    "#7FDBFF",  # 8 = cyan
    "#870C25",  # 9 = brown
]


class ColorPalette(BaseModel):
    colors: list[str] = Field(
        default_factory=lambda: [
            "black",
            "blue",
            "red",
            "green",
            "yellow",
            "gray",
            "magenta",
            "orange",
            "cyan",
            "brown",
        ]
    )

    @property
    def color_map(self):
        return {i: self.colors[i] for i in range(len(self.colors))}

    @property
    def max_width(self):
        return max(len(color) for color in self.colors)

    def permute(self, permutation: list[int] = None):
        if permutation is None:
            permutation = list(range(len(self.colors)))
            random.shuffle(permutation)
        return ColorPalette(colors=[self.colors[i] for i in permutation])

    @classmethod
    def abbreviated(cls):
        return ColorPalette(colors=["k", "b", "r", "g", "y", "e", "m", "o", "c", "n"])

    @classmethod
    def integers(cls):
        return ColorPalette(colors=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])


class Board(BaseModel):
    board: list[list[int]]

    @property
    def shape(self):
        return len(self.board), len(self.board[0])

    @property
    def is_valid(self) -> bool:
        try:
            rows = len(self.board)
            if rows == 0:
                return False
            cols = len(self.board[0])
            if cols == 0:
                return False
            return all(len(row) == cols for row in self.board)
        except Exception:
            return False

    @classmethod
    def from_text(
        cls, text: str, color_palette: ColorPalette = None, row_separator: str = "\n", column_separator: str = " "
    ):
        if color_palette is None:
            color_palette = ColorPalette()

        boxed_match = re.search(r"\\boxed\{(.+)\}", text, re.DOTALL)
        if boxed_match:
            text = boxed_match.group(1)

        if row_separator != "\n":
            text = text.replace(row_separator, "\n")
        if column_separator not in (" ", ""):
            text = text.replace(column_separator, " ")

        text = re.sub(r"[^\s\w]", "", text)
        text = re.sub(r"\b\w+\b", lambda x: x.group(0) if x.group(0) in color_palette.colors else "", text)

        board = []
        for line in text.split("\n"):
            if line.strip():
                board.append([color_palette.colors.index(color) for color in line.split()])

        if board and len(board) > 0 and len(board[0]) > 0:
            expected_cols = len(board[0])
            for i, row in enumerate(board):
                if len(row) != expected_cols:
                    raise ValueError(f"Invalid grid: row {i + 1} has {len(row)} columns, expected {expected_cols}")

        return cls(board=board)

    def description(
        self,
        color_palette: ColorPalette = None,
        add_border: bool = True,
        border_horizontal: str = "-",
        border_vertical: str = "|",
        border_corner: str = "+",
        add_rows: bool = False,
        add_columns: bool = False,
        external_border_only: bool = True,
        column_separator: str = " ",
        row_separator: str = "\n",
    ):
        if color_palette is None:
            color_palette = ColorPalette()

        max_col_num_width = len(str(self.shape[1])) if add_columns else 0
        max_row_num_width = len(str(self.shape[0])) if add_rows else 0
        effective_col_width = max(color_palette.max_width, max_col_num_width)

        d = ""
        if add_border:
            d += border_corner
            border_width = effective_col_width * self.shape[1] + len(column_separator) * (self.shape[1] - 1)
            if add_rows:
                row_header_width = max(max_row_num_width, effective_col_width) + len(column_separator)
                border_width += row_header_width
            if not external_border_only:
                border_width += self.shape[1] - 1
                if add_rows:
                    border_width += 1
            d += border_horizontal * border_width
            d += border_corner
            d += row_separator

        if add_columns:
            if add_border:
                d += border_vertical
            if add_rows:
                row_header_width = max(max_row_num_width, effective_col_width)
                d += f"{'':<{row_header_width}}"
                if add_border:
                    d += border_vertical
            for j in range(self.shape[1]):
                d += f"{j + 1:^{effective_col_width}}"
                if j < self.shape[1] - 1:
                    if add_border and not external_border_only:
                        d += border_vertical
                    else:
                        d += column_separator
            if add_border:
                d += border_vertical
            d += row_separator

            if add_border and not external_border_only:
                d += border_corner
                if add_rows:
                    row_header_width = max(max_row_num_width, effective_col_width) + len(column_separator)
                    d += border_horizontal * row_header_width
                    d += "+"
                for j in range(self.shape[1]):
                    d += border_horizontal * (effective_col_width + len(column_separator))
                    if j < self.shape[1] - 1:
                        d += "+"
                d += border_corner
                d += row_separator

        for i, bi in enumerate(self.board):
            if add_border:
                d += border_vertical
            if add_rows:
                row_num_width = max(max_row_num_width, effective_col_width)
                d += f"{i + 1:^{row_num_width}}"
                if add_border:
                    d += border_vertical
            for j, bj in enumerate(bi):
                color_name = color_palette.color_map[bj]
                d += f"{color_name:^{effective_col_width}}"
                if j < len(bi) - 1:
                    if add_border and not external_border_only:
                        d += border_vertical
                    else:
                        d += column_separator
            if add_border:
                d += border_vertical
            d += row_separator

            if add_border and not external_border_only and i < len(self.board) - 1:
                d += border_vertical
                if add_rows:
                    d += border_horizontal * (effective_col_width + len(column_separator))
                    d += "+"
                for j in range(self.shape[1]):
                    d += border_horizontal * (effective_col_width + len(column_separator))
                    if j < self.shape[1] - 1:
                        d += "+"
                d += border_vertical
                d += row_separator

        if add_border:
            d += border_corner
            border_width = effective_col_width * self.shape[1] + len(column_separator) * (self.shape[1] - 1)
            if add_rows:
                row_header_width = max(max_row_num_width, effective_col_width) + len(column_separator)
                border_width += row_header_width
            if not external_border_only:
                border_width += self.shape[1] - 1
                if add_rows:
                    border_width += 1
            d += border_horizontal * border_width
            d += border_corner
            d += row_separator
        return d

    def rotate90(self):
        return Board(board=[list(row) for row in zip(*self.board[::-1])])

    def rotate180(self):
        return Board(board=[row[::-1] for row in self.board[::-1]])

    def rotate270(self):
        return Board(board=[list(row) for row in zip(*self.board)][::-1])

    def flip_horizontal(self):
        return Board(board=[row[::-1] for row in self.board])

    def flip_vertical(self):
        return Board(board=self.board[::-1])

    def transpose(self):
        return Board(board=[list(row) for row in zip(*self.board)])

    def permute(self, permutation: list[int]):
        perm_len = len(permutation)
        permuted_board = [[permutation[cell] if 0 <= cell < perm_len else cell for cell in row] for row in self.board]
        return Board(board=permuted_board)


class Pair(BaseModel):
    input: Board
    output: Board

    @property
    def shape(self):
        return self.input.shape, self.output.shape

    @classmethod
    def from_dict(cls, data: dict):
        return cls(input=Board(board=data["input"]), output=Board(board=data["output"]))


class Problem(BaseModel):
    problem_id: Optional[str] = None
    examples: list[Pair]
    input: Board
    output: Optional[Board] = None

    def with_solution(self, solution: Board):
        return Problem(problem_id=self.problem_id, examples=self.examples, input=self.input, output=solution)

    def apply_augmentation(self, augmentation: str, permutation: list[int] = None) -> "Problem":
        if augmentation == "none" or augmentation is None:
            return self

        if augmentation == "shuffle_boards":
            return self._shuffle_boards()

        if augmentation == "permute" and permutation is None:
            permutation = list(range(10))
            random.shuffle(permutation)

        augmentation_methods = {
            "rotate90": lambda b: b.rotate90(),
            "rotate180": lambda b: b.rotate180(),
            "rotate270": lambda b: b.rotate270(),
            "flip_horizontal": lambda b: b.flip_horizontal(),
            "flip_vertical": lambda b: b.flip_vertical(),
            "transpose": lambda b: b.transpose(),
            "permute": lambda b: b.permute(permutation),
        }

        if augmentation not in augmentation_methods:
            raise ValueError(
                f"Unknown augmentation: {augmentation}. "
                f"Must be one of: {list(augmentation_methods.keys()) + ['none', 'shuffle_boards']}"
            )

        augment_fn = augmentation_methods[augmentation]

        augmented_examples = [
            Pair(input=augment_fn(pair.input), output=augment_fn(pair.output)) for pair in self.examples
        ]
        augmented_input = augment_fn(self.input)
        augmented_output = augment_fn(self.output) if self.output else None

        return Problem(
            problem_id=self.problem_id,
            examples=augmented_examples,
            input=augmented_input,
            output=augmented_output,
        )

    def _shuffle_boards(self) -> "Problem":
        if self.output:
            all_pairs = list(self.examples)
            all_pairs.append(Pair(input=self.input, output=self.output))
            shuffled_pairs = all_pairs.copy()
            random.shuffle(shuffled_pairs)
            if len(shuffled_pairs) > 0:
                new_test = shuffled_pairs[-1]
                new_examples = shuffled_pairs[:-1]
            else:
                new_test = Pair(input=self.input, output=self.output)
                new_examples = []
            return Problem(
                problem_id=self.problem_id,
                examples=new_examples,
                input=new_test.input,
                output=new_test.output,
            )
        else:
            shuffled_examples = list(self.examples)
            random.shuffle(shuffled_examples)
            return Problem(
                problem_id=self.problem_id,
                examples=shuffled_examples,
                input=self.input,
                output=self.output,
            )


class TextualProblemGenerator(BaseModel):
    problem: Problem
    system_prompt: str = Field(default_factory=lambda: "")
    add_border: bool = True
    add_rows: bool = False
    add_columns: bool = False
    external_border_only: bool = True
    border_horizontal: str = "-"
    border_vertical: str = "|"
    border_corner: str = "+"
    column_separator: str = " "
    row_separator: str = "\n"
    color_palette: ColorPalette = Field(default_factory=ColorPalette)

    def _grid_description(self, board) -> str:
        return board.description(
            color_palette=self.color_palette,
            add_border=self.add_border,
            add_rows=self.add_rows,
            add_columns=self.add_columns,
            external_border_only=self.external_border_only,
            border_horizontal=self.border_horizontal,
            border_vertical=self.border_vertical,
            border_corner=self.border_corner,
            column_separator=self.column_separator,
            row_separator=self.row_separator,
        )

    def _format_example(self, i: int, pe) -> str:
        t = f"Train Example {i + 1}:\n\nInput:\n"
        t += self._grid_description(pe.input)
        t += "\nOutput:\n"
        t += self._grid_description(pe.output)
        t += "\n"
        return t

    def _format_test_input(self) -> str:
        t = "\n\nTest Input:\n"
        t += self._grid_description(self.problem.input)
        return t

    @property
    def problem_textual_description(self):
        t = ""
        for i, pe in enumerate(self.problem.examples):
            t += self._format_example(i, pe)
        t += self._format_test_input()
        return t
