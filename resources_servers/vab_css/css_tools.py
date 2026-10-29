# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""The VAB-CSS editing tools and score, ported from THUDM/VisualAgentBench (Apache-2.0).

Source: src/server/tasks/css_agent/{actions,task}.py at commit 9055fc299c366ef34700d1710215fb60a0d8c35e.
The selector matching, the edit and revert semantics and the SSIM score are kept as upstream has them, so
scores stay comparable with the paper. One change: upstream passes the model's
get_selectors_by_html_elements argument to eval(); here it is parsed and only literals are accepted.
"""

import ast
import logging
import os
from pathlib import Path
from typing import Any, Optional

import cssutils
import numpy as np
from bs4 import BeautifulSoup
from skimage import io
from skimage.metrics import structural_similarity
from skimage.transform import resize


cssutils.log.setLevel(logging.CRITICAL)


def compare_images_ssim_gray(img1_path: str | Path, img2_path: str | Path) -> float:
    """Upstream's similarity: grayscale SSIM, with img2 resized to img1 and a bonus for small size changes."""
    img1 = io.imread(img1_path, as_gray=True)
    img2 = io.imread(img2_path, as_gray=True)
    fake_size_change = False
    if img1.shape != img2.shape:
        resize_percentage = abs(np.array(img2.shape) - np.array(img1.shape)) / np.array(img2.shape)
        if all(resize_percentage < 0.1):
            fake_size_change = True
        img2 = resize(img2, img1.shape)
    img1 = (img1 * 255).astype(np.uint8)
    img2 = (img2 * 255).astype(np.uint8)
    ssim_val = float(structural_similarity(img1, img2))
    if fake_size_change:
        ssim_val = 0.5 + ssim_val / 2
    return ssim_val


def matches_simple_selector(element: Any, selector: str) -> bool:
    if selector.startswith("#"):
        if element.get("id") != selector[1:]:
            return False
    elif selector.startswith("."):
        for cls in selector[1:].split("."):
            if cls not in element.get("class", []):
                return False
    else:
        parts = selector.split(".")
        tag = parts[0] if parts[0] else None
        if "[" in tag:
            tag = tag[: tag.find("[")]
        if tag and tag != element.name:
            return False
        for cls in parts[1:]:
            if cls not in element.get("class", []):
                return False
        if "[" in selector and "]" in selector:
            attr = selector[selector.find("[") + 1 : selector.find("]")].split("=")
            attr_value = attr[1].strip('"') if len(attr) > 1 else None
            if attr_value:
                return element.get(attr[0]) == attr_value
            return element.has_attr(attr[0])
    return True


def matches_complex_selector(element: Any, selector: str) -> bool:
    while " + " in selector:
        selector = selector.replace(" + ", "+")
    while " > " in selector:
        selector = selector.replace(" > ", ">")
    current_element = element
    for i, part in enumerate(reversed(selector.split())):
        if ">" in part:
            parent_selector, child_selector = part.split(">")
            if not matches_simple_selector(current_element, child_selector.strip()):
                return False
            current_element = current_element.parent
            if current_element is None or not matches_simple_selector(current_element, parent_selector.strip()):
                return False
        elif "+" in part:
            prev_sibling_selector, next_sibling_selector = part.split("+")
            if not matches_simple_selector(current_element, next_sibling_selector.strip()):
                return False
            current_element = current_element.find_previous_sibling()
            if current_element is None or not matches_simple_selector(current_element, prev_sibling_selector.strip()):
                return False
        elif i > 0:
            while current_element is not None and not matches_simple_selector(current_element, part.strip()):
                current_element = current_element.parent
            if current_element is None:
                return False
        elif not matches_simple_selector(current_element, part.strip()):
            return False
    return True


def get_selector_by_html_element(element: Any, css_rules: list[str]) -> list[str]:
    matched_selectors = []
    for selector in css_rules:
        for subgroup in selector.split(",") if "," in selector else [selector]:
            subgroup = subgroup.split("::")[0].split(":")[0]
            if matches_complex_selector(element, subgroup):
                matched_selectors.append(selector)
    return matched_selectors


def parse_find_all_arguments(arguments: str) -> tuple[list, dict]:
    """Parse "'a', {'data-x': 'y'}, class_='z'" into find_all's args and kwargs, accepting literals only."""
    call = ast.parse(f"find_all({arguments})", mode="eval").body
    if not isinstance(call, ast.Call):
        raise ValueError("the argument must be a list of find_all arguments")
    args = [ast.literal_eval(arg) for arg in call.args]
    kwargs = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise ValueError("**kwargs is not supported")
        kwargs[keyword.arg] = ast.literal_eval(keyword.value)
    return args, kwargs


class CSSInteractor:
    """One task's HTML page and its local stylesheets; edits are written back to the CSS files."""

    def __init__(self, html_file_path: str | Path) -> None:
        self.html_file_path = str(html_file_path)
        self.root_path = os.path.dirname(self.html_file_path)
        with open(self.html_file_path) as f:
            self.soup = BeautifulSoup(f, "html.parser")
        self.css_files = [
            os.path.join(self.root_path, link["href"])
            for link in self.soup.find_all("link", rel="stylesheet")
            if "http" not in link["href"]
        ]
        self.css_sheet = {css_file: cssutils.parseFile(css_file) for css_file in self.css_files}
        self.last_edit: Optional[dict] = None

    def get_selectors_by_html_elements(self, find_all_arguments: str) -> tuple[list, list[str]]:
        args, kwargs = parse_find_all_arguments(find_all_arguments)
        elements = self.soup.find_all(*args, **kwargs)
        matched_selectors = set()
        for element in elements:
            for sheet in self.css_sheet.values():
                for rule in sheet:
                    if rule.type != rule.STYLE_RULE:
                        continue
                    try:
                        matched_selectors.update(get_selector_by_html_element(element, [rule.selectorText]))
                    except Exception:  # upstream skips selectors its matcher cannot parse
                        continue
        return elements, list(matched_selectors)

    def select_rule(self, selector: str) -> tuple[Any, Optional[str]]:
        selector = selector.strip().strip('"').strip("'")
        for css_file, sheet in self.css_sheet.items():
            for rule in sheet:
                if rule.type == rule.STYLE_RULE and rule.selectorText == selector:
                    return rule, css_file
        return None, None

    def edit_rule(self, selector: str, property_name: str, value: str) -> bool:
        """Set one property of a rule and write the sheet; the caller takes the new screenshot."""
        selector = selector.strip().strip('"').strip("'")
        property_name = property_name.strip().strip('"').strip("'")
        value = value.strip().strip('"').strip("'")
        rule, css_file = self.select_rule(selector)
        if css_file is None:
            return False
        old_value = rule.style.getPropertyValue(property_name)
        self.last_edit = {
            "type": "alter" if old_value else "add_property",
            "css_file": css_file,
            "selector": selector,
            "property_name": property_name,
            "old_value": old_value,
            "new_value": value,
        }
        rule.style.setProperty(property_name, value)
        self._write(css_file)
        return True

    def revert_last_edit(self) -> bool:
        if not self.last_edit:
            return False
        edit = self.last_edit
        if edit["type"] == "alter":
            rule, _ = self.select_rule(edit["selector"])
            rule.style.setProperty(edit["property_name"], edit["old_value"])
        elif edit["type"] == "add_property":
            rule, _ = self.select_rule(edit["selector"])
            rule.style.removeProperty(edit["property_name"])
        self._write(edit["css_file"])
        return True

    def _write(self, css_file: str) -> None:
        with open(css_file, "w") as f:
            f.write(self.css_sheet[css_file].cssText.decode("utf-8"))
