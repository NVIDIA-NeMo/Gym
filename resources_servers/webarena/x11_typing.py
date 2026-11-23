# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keymap-consistent typing over XTest, a drop-in for ``pyautogui.write`` on Xvfb.

Why pyautogui types ``>`` for ``<``: ``pyautogui.write`` picks a keycode with
python-xlib's ``keysym_to_keycode``, which returns the binding with the lowest
*level*, and then decides on its own whether to hold Shift (a fixed US list of
"shifted" characters). On the default Xvfb keymap (evdev, pc105, us) the keysym
``less`` is level 1 of the 102nd key <LSGT> (keycode 94: less / greater) and
level 2 of the comma key (59). pyautogui takes 94 and holds Shift because ``<``
is in its list, so the browser receives Shift+<LSGT>, which is ``>``. Every
``<`` comes out as ``>``, whatever came before it. Characters with no binding in
the keymap (``™``, ``é``, ``—``, curly quotes, CJK, emoji) are skipped silently.

``XKeymapTyper`` takes the keycode and the Shift state from the same binding: a
character bound at level 1 (core index 0) is typed without modifiers, one bound
at level 2 (index 1) with Shift; among several bindings the lowest keycode wins
(main-block key before the 102nd key: ``<`` = Shift+comma, as on a US keyboard).
A character with no level-1/2 binding is typed through a spare keycode that is
bound to its keysym on the fly (the technique ``xdotool type`` uses), handed to an
``insert_text`` callback, or dropped, as configured. The Chromium of the training
image does not use re-bound keycodes (browser check 2026-10-17: remap typed nothing
for them), so cuagym hands these characters to Playwright's ``insert_text`` by
default. Modifier keys still held down are released before typing.

The module imports nothing from Xlib: the worker passes pyautogui's python-xlib
``Display`` and ``Xlib.ext.xtest.fake_input``, tests pass fakes.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal


X_KEY_PRESS = 2  # Xlib.X.KeyPress
X_KEY_RELEASE = 3  # Xlib.X.KeyRelease
NO_SYMBOL = 0
XK_BACKSPACE = 0xFF08
XK_TAB = 0xFF09
XK_RETURN = 0xFF0D
XK_SHIFT_L = 0xFFE1

# Control characters pyautogui.write also maps to keys ("\r" and "\n" both press Return).
_CONTROL_KEYSYMS = {"\n": XK_RETURN, "\r": XK_RETURN, "\t": XK_TAB, "\b": XK_BACKSPACE}

UnmappedChars = Literal["remap", "insert_text", "drop"]


def char_to_keysym(char: str) -> int | None:
    """X11 keysym of one character; None for characters that are not typeable."""
    if char in _CONTROL_KEYSYMS:
        return _CONTROL_KEYSYMS[char]
    code = ord(char)
    if 0x20 <= code <= 0x7E or 0xA0 <= code <= 0xFF:
        return code  # Latin-1 keysyms equal the code point
    if code < 0xA0 or 0xD800 <= code <= 0xDFFF:
        return None  # other C0/C1 controls, DEL, lone surrogates
    return 0x01000000 | code  # Unicode keysym


def choose_binding(bindings: Iterable[tuple[int, int]]) -> tuple[int, bool] | None:
    """Pick ``(keycode, shift)`` from ``(keycode, core index)`` pairs.

    Index 0 is typed without modifiers and index 1 with Shift; other indices need
    AltGr or a group switch and are not used. The lowest keycode wins.
    """
    usable = sorted((keycode, index) for keycode, index in bindings if keycode > 0 and index in (0, 1))
    if not usable:
        return None
    keycode, index = usable[0]
    return keycode, index == 1


@dataclass(frozen=True)
class Stroke:
    char: str
    keysym: int | None  # None: not typeable
    keycode: int | None  # None with a keysym: no level-1/2 binding in the current keymap
    shift: bool = False


def plan_strokes(text: str, keysym_to_keycodes: Callable[[int], Iterable[tuple[int, int]]]) -> list[Stroke]:
    """One stroke per character of ``text`` against the given keymap lookup."""
    strokes: list[Stroke] = []
    for char in text:
        keysym = char_to_keysym(char)
        if keysym is None:
            strokes.append(Stroke(char, None, None))
            continue
        binding = choose_binding(keysym_to_keycodes(keysym))
        if binding is None:
            strokes.append(Stroke(char, keysym, None))
        else:
            strokes.append(Stroke(char, keysym, binding[0], binding[1]))
    return strokes


@dataclass
class TypingResult:
    typed: int = 0  # characters sent as key events
    inserted: str = ""  # characters handed to insert_text
    dropped: str = ""  # characters that produced nothing
    released_modifiers: list[int] = field(default_factory=list)


class XKeymapTyper:
    """Types text on one X display with keymap-consistent keycode/Shift choices.

    ``display`` is a python-xlib ``Display`` (pyautogui's ``platformModule._display``);
    ``fake_input`` is ``Xlib.ext.xtest.fake_input``. One instance per display/process.
    """

    def __init__(
        self,
        display: Any,
        fake_input: Callable[..., Any],
        *,
        unmapped: UnmappedChars = "remap",
        insert_text: Callable[[str], None] | None = None,
        remap_settle_seconds: float = 0.05,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if unmapped not in ("remap", "insert_text", "drop"):
            raise ValueError(f"unknown unmapped-character mode: {unmapped!r}")
        if unmapped == "insert_text" and insert_text is None:
            raise ValueError("unmapped='insert_text' needs an insert_text callback")
        self.display = display
        self.fake_input = fake_input
        self.unmapped = unmapped
        self.insert_text = insert_text
        self.remap_settle_seconds = remap_settle_seconds
        self.sleep = sleep
        shift = choose_binding(display.keysym_to_keycodes(XK_SHIFT_L))
        self._shift_keycode = shift[0] if shift else None
        self._spare: list[int] | None = None  # keycodes with no keysym, found lazily
        self._bound: OrderedDict[int, int] = OrderedDict()  # keysym -> spare keycode, LRU order

    # -- public -------------------------------------------------------------

    def release_held_modifiers(self) -> list[int]:
        """Release every modifier key the X server reports as pressed."""
        pressed = self.display.query_keymap()
        released: list[int] = []
        for group in self.display.get_modifier_mapping():
            for keycode in group:
                if keycode and keycode not in released and pressed[keycode // 8] & (1 << (keycode % 8)):
                    self.fake_input(self.display, X_KEY_RELEASE, keycode)
                    released.append(keycode)
        if released:
            self.display.sync()
        return released

    def type_text(self, text: str, *, interval: float = 0.02) -> TypingResult:
        result = TypingResult(released_modifiers=self.release_held_modifiers())
        pending_insert: list[str] = []
        used: set[int] = set()
        for stroke in plan_strokes(text, self.display.keysym_to_keycodes):
            if stroke.keysym is None:
                result.dropped += stroke.char
                continue
            keycode, shift = stroke.keycode, stroke.shift
            if keycode is None or (shift and self._shift_keycode is None):
                if self.unmapped == "insert_text":
                    pending_insert.append(stroke.char)
                    continue
                keycode = self._spare_keycode_for(stroke.keysym, used) if self.unmapped == "remap" else None
                shift = False
                if keycode is None:
                    result.dropped += stroke.char
                    continue
            if pending_insert:
                self._flush_insert(pending_insert, result)
            self._tap(keycode, shift)
            used.add(keycode)
            result.typed += 1
            self.sleep(interval)
        if pending_insert:
            self._flush_insert(pending_insert, result)
        return result

    # -- internals ------------------------------------------------------------

    def _tap(self, keycode: int, shift: bool) -> None:
        if shift:
            self.fake_input(self.display, X_KEY_PRESS, self._shift_keycode)
        self.fake_input(self.display, X_KEY_PRESS, keycode)
        self.fake_input(self.display, X_KEY_RELEASE, keycode)
        if shift:
            self.fake_input(self.display, X_KEY_RELEASE, self._shift_keycode)
        self.display.sync()

    def _flush_insert(self, pending: list[str], result: TypingResult) -> None:
        text = "".join(pending)
        pending.clear()
        # Let queued key events reach the browser before text arrives on the other channel.
        self.sleep(self.remap_settle_seconds)
        self.insert_text(text)
        result.inserted += text

    def _spare_keycodes(self) -> list[int]:
        if self._spare is None:
            info = self.display.display.info
            low, high = int(info.min_keycode), int(info.max_keycode)
            mapping = self.display.get_keyboard_mapping(low, high - low + 1)
            modifiers = {keycode for group in self.display.get_modifier_mapping() for keycode in group if keycode}
            spare = [
                low + offset
                for offset, keysyms in enumerate(mapping)
                if not any(keysyms) and (low + offset) not in modifiers
            ]
            self._spare = sorted(spare, reverse=True)  # high keycodes first, like xdotool's scratch key
        return self._spare

    def _spare_keycode_for(self, keysym: int, used: set[int]) -> int | None:
        """A spare keycode bound to ``keysym`` (binding it now if needed), or None."""
        if keysym in self._bound:
            self._bound.move_to_end(keysym)
            return self._bound[keysym]
        taken = set(self._bound.values())
        free = [keycode for keycode in self._spare_keycodes() if keycode not in taken]
        if free:
            keycode = free[0]
        else:
            # Rebind the least recently used spare keycode, but never one typed in this call:
            # the browser may not have translated that key event yet.
            victim = next((sym for sym, code in self._bound.items() if code not in used), None)
            if victim is None:
                return None
            keycode = self._bound.pop(victim)
        # Same keysym on both levels so a stray Shift cannot change the result. The binding is
        # kept (not restored after the key event), so the browser can never translate the key
        # event with a stale or restored keymap.
        self.display.change_keyboard_mapping(keycode, [(keysym, keysym)])
        self.display.sync()
        self._bound[keysym] = keycode
        self.sleep(self.remap_settle_seconds)  # let clients process MappingNotify first
        return keycode
