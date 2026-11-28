# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline tests for the keymap-consistent typer (no X server, no Xlib, no pyautogui).

A fake X display carries the core keymap of a default Xvfb (evdev, pc105, us) for the
keys that matter, and a replay of the recorded XTest events stands in for the browser:
it applies Shift as XKB does (core index 1 = Shift level) and honours keymap changes in
event order, as a client that processes MappingNotify does. The same model reproduces
the observed pyautogui bug ("<title>" typed as ">title>"), which checks the model.

    python3 -I resources_servers/webarena/tests/test_x11_typing.py
"""

from __future__ import annotations

import string
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from resources_servers.webarena.x11_typing import (  # noqa: E402
    X_KEY_PRESS,
    X_KEY_RELEASE,
    XK_BACKSPACE,
    XK_RETURN,
    XK_TAB,
    XKeymapTyper,
    char_to_keysym,
    plan_strokes,
)


ESCAPE, ISO_LEFT_TAB, CONTROL_L, CONTROL_R = 0xFF1B, 0xFE20, 0xFFE3, 0xFFE4
SHIFT_L, SHIFT_R, ALT_L, ALT_R, META_L, SUPER_L = 0xFFE1, 0xFFE2, 0xFFE9, 0xFFEA, 0xFFE7, 0xFFEB
CAPS_LOCK, NUM_LOCK, ISO_LEVEL3_SHIFT = 0xFFE5, 0xFF7F, 0xFE03
KP_HOME, KP_7, KP_ADD, KP_DELETE, KP_DECIMAL, KP_ENTER = 0xFF95, 0xFFB7, 0xFFAB, 0xFF9F, 0xFFAE, 0xFF8D
BROKENBAR = 0xA6


def _two(a, b):
    a, b = (ord(x) if isinstance(x, str) else x for x in (a, b))
    return [a, b, a, b]  # XKB core conversion repeats group 1 in the group 2 slots


def default_xvfb_keymap() -> dict[int, list[int]]:
    """Core keymap of a default Xvfb (rules evdev, model pc105, layout us), relevant keys only."""
    km: dict[int, list[int]] = {9: [ESCAPE], 22: [XK_BACKSPACE], 23: [XK_TAB, ISO_LEFT_TAB], 36: [XK_RETURN]}
    for keycode, (low, high) in zip(range(10, 20), zip("1234567890", "!@#$%^&*()")):
        km[keycode] = _two(low, high)
    for keycode, row in ((24, "qwertyuiop"), (38, "asdfghjkl"), (52, "zxcvbnm")):
        for offset, letter in enumerate(row):
            km[keycode + offset] = _two(letter, letter.upper())
    for keycode, pair in {20: "-_", 21: "=+", 34: "[{", 35: "]}", 47: ";:", 48: "'\"", 49: "`~", 51: "\\|"}.items():
        km[keycode] = _two(*pair)
    for keycode, pair in {59: ",<", 60: ".>", 61: "/?"}.items():
        km[keycode] = _two(*pair)
    km.update(
        {
            37: [CONTROL_L],
            50: [SHIFT_L],
            62: [SHIFT_R],
            64: [ALT_L, META_L],
            65: [ord(" ")],
            66: [CAPS_LOCK],
            77: [NUM_LOCK],
            79: [KP_HOME, KP_7],
            86: [KP_ADD, KP_ADD],
            91: [KP_DELETE, KP_DECIMAL],
            92: [ISO_LEVEL3_SHIFT],
            94: [ord("<"), ord(">"), ord("<"), ord(">"), ord("|"), BROKENBAR],  # <LSGT>, the 102nd key
            104: [KP_ENTER],
            105: [CONTROL_R],
            108: [ALT_R],
            133: [SUPER_L],
        }
    )
    return km


MODIFIERS = [[50, 62], [66], [37, 105], [64, 108], [77], [], [133], [92]]
SHIFT_CODES = {50, 62}
CONTROL_CODES = {37, 105}


class FakeDisplay:
    """Just enough of python-xlib's Display, recording XTest events and keymap changes in order."""

    def __init__(self, keymap: dict[int, list[int]], low: int = 8, high: int = 255) -> None:
        self.keymap = {keycode: list(syms) for keycode, syms in keymap.items()}
        self.initial = {keycode: list(syms) for keycode, syms in keymap.items()}
        self.display = SimpleNamespace(info=SimpleNamespace(min_keycode=low, max_keycode=high))
        self.pressed: set[int] = set()
        self.events: list[tuple] = []
        # python-xlib builds its keysym -> keycode cache when the connection opens; pyautogui's
        # display never processes MappingNotify, so the cache does not see later changes.
        self._cache: dict[int, list[tuple[int, int]]] = {}
        for keycode, syms in sorted(self.keymap.items()):
            for index, sym in enumerate(syms):
                if sym:
                    self._cache.setdefault(sym, []).append((index, keycode))
        for entries in self._cache.values():
            entries.sort()

    def keysym_to_keycodes(self, keysym):
        return [(keycode, index) for index, keycode in self._cache.get(keysym, [])]

    def keysym_to_keycode(self, keysym):  # what pyautogui uses: lowest index, then lowest keycode
        entries = self._cache.get(keysym)
        return entries[0][1] if entries else 0

    def get_keyboard_mapping(self, first, count):
        return [list(self.keymap.get(first + i) or [0, 0]) for i in range(count)]

    def change_keyboard_mapping(self, first, keysyms):
        for offset, syms in enumerate(keysyms):
            self.keymap[first + offset] = list(syms)
            self.events.append(("remap", first + offset, tuple(syms)))

    def get_modifier_mapping(self):
        return [list(group) for group in MODIFIERS]

    def query_keymap(self):
        bits = [0] * 32
        for keycode in self.pressed:
            bits[keycode // 8] |= 1 << (keycode % 8)
        return bits

    def sync(self):
        pass


def fake_input(display: FakeDisplay, event_type: int, detail: int) -> None:
    if event_type == X_KEY_PRESS:
        display.pressed.add(detail)
        display.events.append(("press", detail))
    elif event_type == X_KEY_RELEASE:
        display.pressed.discard(detail)
        display.events.append(("release", detail))


def _keysym_char(sym: int) -> str | None:
    if sym == XK_RETURN:
        return "\n"
    if sym == XK_TAB:
        return "\t"
    if 0x20 <= sym <= 0xFF:
        return chr(sym)
    if sym & 0xFF000000 == 0x01000000:
        return chr(sym & 0xFFFFFF)
    return None


def browser_text(display: FakeDisplay) -> str:
    """Replay the recorded events the way the browser receives them."""
    keymap = {keycode: list(syms) for keycode, syms in display.initial.items()}
    held: set[int] = set()
    out: list[str] = []
    for event in display.events:
        if event[0] == "remap":
            keymap[event[1]] = list(event[2])
            continue
        if event[0] == "insert":
            out.append(event[1])
            continue
        kind, keycode = event
        if keycode in SHIFT_CODES or keycode in CONTROL_CODES:
            (held.add if kind == "press" else held.discard)(keycode)
            continue
        if kind != "press" or held & CONTROL_CODES:
            continue
        syms = keymap.get(keycode) or [0]
        level = 1 if held & SHIFT_CODES else 0
        sym = syms[level] if len(syms) > level and syms[level] else syms[0]
        if sym == XK_BACKSPACE:
            if out:
                out.pop()
            continue
        char = _keysym_char(sym)
        if char is not None:
            out.append(char)
    return "".join(out)


PYAUTOGUI_SHIFT_CHARACTERS = set('~!@#$%^&*()_+{}|:"<>?')  # pyautogui.isShiftCharacter


def pyautogui_write(display: FakeDisplay, text: str) -> None:
    """pyautogui 0.9.54 write() on X11: keycode from keysym_to_keycode, Shift from its own list."""
    for char in text:
        if not (char in string.printable[:-2] or char == "\b"):
            continue  # no keyboardMapping entry -> _keyDown returns silently
        keycode = display.keysym_to_keycode(char_to_keysym(char))
        shift = char.isupper() or char in PYAUTOGUI_SHIFT_CHARACTERS
        if shift:
            fake_input(display, X_KEY_PRESS, 50)
        fake_input(display, X_KEY_PRESS, keycode)
        if shift:
            fake_input(display, X_KEY_RELEASE, 50)
        fake_input(display, X_KEY_RELEASE, keycode)


def _typer(display: FakeDisplay, **kwargs) -> XKeymapTyper:
    return XKeymapTyper(display, fake_input, sleep=lambda _seconds: None, **kwargs)


ASCII = "".join(c for c in string.printable if c not in "\x0b\x0c\r")


# --------------------------------------------------------------------------- tests


def test_model_reproduces_the_observed_pyautogui_bug() -> None:
    # E30 Qwen gitlab 442 and step_239's real-eval task 441 screenshots: '>title>Welcome to my site>/title>'.
    display = FakeDisplay(default_xvfb_keymap())
    pyautogui_write(display, "<title>Welcome to my site</title> café ™")
    assert browser_text(display) == ">title>Welcome to my site>/title> caf "


def test_typer_types_markup_exactly() -> None:
    display = FakeDisplay(default_xvfb_keymap())
    result = _typer(display).type_text("<title>Welcome to my site</title>")
    assert browser_text(display) == "<title>Welcome to my site</title>"
    assert result.typed == 33 and result.dropped == "" and result.inserted == ""
    assert not any(event[0] == "remap" for event in display.events), "ASCII must not touch the keymap"


def test_typer_round_trips_all_printable_ascii() -> None:
    display = FakeDisplay(default_xvfb_keymap())
    _typer(display).type_text(ASCII)
    assert browser_text(display) == ASCII
    legacy = FakeDisplay(default_xvfb_keymap())
    pyautogui_write(legacy, ASCII)
    assert browser_text(legacy) == ASCII.replace("<", ">"), "pyautogui gets only '<' wrong"


def test_plan_takes_keycode_and_shift_from_the_same_binding() -> None:
    display = FakeDisplay(default_xvfb_keymap())
    plan = {s.char: (s.keycode, s.shift) for s in plan_strokes("<>|aA\n\t ", display.keysym_to_keycodes)}
    assert plan["<"] == (59, True)  # Shift+comma, not Shift+<LSGT>
    assert plan[">"] == (60, True)
    assert plan["|"] == (51, True)  # level 2 of backslash, not level 3 of <LSGT>
    assert plan["a"] == (38, False) and plan["A"] == (38, True)
    assert plan["\n"] == (36, False) and plan["\t"] == (23, False) and plan[" "] == (65, False)
    unmapped = plan_strokes("™\x07", display.keysym_to_keycodes)
    assert (unmapped[0].keysym, unmapped[0].keycode) == (0x1002122, None)
    assert unmapped[1].keysym is None


def test_char_to_keysym() -> None:
    assert char_to_keysym("a") == 0x61 and char_to_keysym("é") == 0xE9 and char_to_keysym("\xa0") == 0xA0
    assert char_to_keysym("™") == 0x1002122 and char_to_keysym("舒") == 0x1000000 | ord("舒")
    assert char_to_keysym("😀") == 0x101F600
    assert char_to_keysym("\n") == XK_RETURN and char_to_keysym("\r") == XK_RETURN
    for char in ("\x00", "\x07", "\x1b", "\x7f", "\x85", "\ud800"):
        assert char_to_keysym(char) is None, repr(char)


def test_unmapped_characters_go_through_spare_keycodes() -> None:
    # Characters our checkpoints typed in held-out episodes (é, CJK names, emoji) plus ™ and curly quotes.
    text = "café @舒岩 ™ “quoted” — 😀 doesn’t ñ"
    display = FakeDisplay(default_xvfb_keymap())
    typer = _typer(display)
    result = typer.type_text(text)
    assert browser_text(display) == text
    assert result.dropped == "" and result.inserted == ""
    remapped = [event[1] for event in display.events if event[0] == "remap"]
    assert len(remapped) == len(set(remapped)) == len({c for c in text if ord(c) > 0x7E})
    for keycode in remapped:
        assert not any(display.initial.get(keycode, [])), "only keycodes without keysyms are re-bound"
        assert keycode not in {code for group in MODIFIERS for code in group}
    # Bindings are kept: typing the same characters again changes no keymap entry.
    before = len(display.events)
    typer.type_text("ñ ™ 😀")
    assert not any(event[0] == "remap" for event in display.events[before:])
    assert browser_text(display) == text + "ñ ™ 😀"


def test_held_modifiers_are_released_before_typing() -> None:
    display = FakeDisplay(default_xvfb_keymap())
    fake_input(display, X_KEY_PRESS, 50)  # Shift_L left down
    fake_input(display, X_KEY_PRESS, 37)  # Control_L left down: the browser would read keys as shortcuts
    stuck = FakeDisplay(default_xvfb_keymap())
    stuck.events = list(display.events)
    pyautogui_write(stuck, "ab<")
    assert browser_text(stuck) == "", "with Ctrl held, typed keys never reach the text field"
    result = _typer(display).type_text("ab<")
    assert result.released_modifiers == [50, 37]
    assert browser_text(display) == "ab<"
    assert not display.pressed


def test_insert_text_mode_keeps_character_order() -> None:
    display = FakeDisplay(default_xvfb_keymap())
    calls: list[str] = []

    def insert(text: str) -> None:
        calls.append(text)
        display.events.append(("insert", text))

    result = _typer(display, unmapped="insert_text", insert_text=insert).type_text("a™é<b😀")
    assert calls == ["™é", "😀"]
    assert browser_text(display) == "a™é<b😀"
    assert result.inserted == "™é😀" and result.typed == 3
    assert not any(event[0] == "remap" for event in display.events)


def test_drop_mode_matches_pyautogui_for_unmapped_characters() -> None:
    display = FakeDisplay(default_xvfb_keymap())
    result = _typer(display, unmapped="drop").type_text("café ™<")
    assert browser_text(display) == "caf <"
    assert result.dropped == "é™"


def test_spare_keycodes_are_reused_least_recently_used_first() -> None:
    keymap = default_xvfb_keymap()
    for keycode in range(8, 256):  # every keycode bound except two spares
        keymap.setdefault(keycode, [0x1008FF00 + keycode])
    for keycode in (250, 251):
        keymap[keycode] = []
    display = FakeDisplay(keymap)
    typer = _typer(display)
    first = typer.type_text("αβγ")
    assert first.dropped == "γ", "a keycode typed in this call is never re-bound in the same call"
    assert browser_text(display) == "αβ"
    second = typer.type_text("γα")
    assert second.dropped == ""
    assert browser_text(display) == "αβγα"
    remaps = [event for event in display.events if event[0] == "remap"]
    # α->251, β->250; then γ takes α's keycode (least recently used) and α takes β's.
    bound = [(event[1], chr(event[2][0] & 0xFFFFFF)) for event in remaps]
    assert bound == [(251, "α"), (250, "β"), (251, "γ"), (250, "α")]


def test_bad_configuration_is_rejected() -> None:
    display = FakeDisplay(default_xvfb_keymap())
    for kwargs in ({"unmapped": "xdotool"}, {"unmapped": "insert_text"}):
        try:
            _typer(display, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"{kwargs} accepted")


def _main() -> int:
    failures = 0
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception:
            failures += 1
            traceback.print_exc()
            print(f"FAIL {name}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
