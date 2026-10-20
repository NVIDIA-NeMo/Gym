"""Hidden probe tests for type 18: crafted inputs full of HTML-special characters rendered through every construct
the units touch, compared with the oracle's output (recorded at build time). They catch over-escaping and
double-escaping that the projects' own suites do not exercise (found in the verifier audit: escaping the items of
jinja's join filter passed every real test but double-escapes under autoescape)."""

MISTUNE = [
    "Text <with> & \"quotes\" 'single' &amp; &#60; &copy;\n",
    "[l<i>nk & \"x\"](http://x.com/a?b=1&c=<2> \"t<i>tle & 'x'\")\n",
    "![a<l>t & \"x\" *em*](/img.png?a=1&b=2 \"ti<t>le & \\\"q\\\"\")\n",
    "`co<d>e & \"x\" &amp;`\n",
    "```py<th>on & \"x\"\nif a < b && c > \"d\": &amp;\n    pass\n```\n",
    "<div class=\"a&b\">raw & <b>html</b> &amp;</div>\n",
    "a <span title=\"x&y\">x</span> & b\n",
    "<http://x.com/?a=<b>&c=\"d\">\n",
    "[r][1] and [s][]\n\n[1]: http://x.com/?a&b=\"c\" 'ti&tle <x>'\n[s]: </p&q> \"t&t\"\n",
    "*[H&TML]: Hyper <Text> & \"Markup\" 'x'\n\nH&TML is <fun>\n",
    "see http://x.com/?a=1&b=<2>&c=\"3\" now\n",
]
MARKDOWN = [
    "Text <with> & \"quotes\" 'single' &amp; &#60; &copy;\n",
    "    if a < b && c > \"d\": &amp;\n        pass\n",
    "```py<th>on\nif a < b && c > \"d\": &amp;\n```\n",
    "`co<d>e & \"x\" &amp;` and ``a ` <b> & c``\n",
    "*[H&TML]: Hyper <Text> & \"Markup\" 'x'\n\nH&TML is <fun>\n",
    "[l<i>nk & \"x\"](http://x.com/a?b=1&c=<2> \"t<i>tle & 'x'\")\n",
    "![a<l>t & \"x\"](/img.png?a=1&b=2 \"ti<t>le\")\n",
    "<div class=\"a&b\">raw & <b>html</b></div>\n\nafter & <i>it</i>\n",
]
PYMDOWNX = [
    "```py<th>on\nif a < b && c > \"d\": &amp;\n```\n",
    "```{.python title=\"a<b&c\"}\nx = '<y>' & \"z\"\n```\n",
    "/// html | div.c[title=\"a<b&c\"]\n<b>& \"x\"</b> &amp;\n///\n",
    "/// html | pre\n<b>& \"x\"</b> &amp; 'y'\n///\n",
]
JINJA = [
    ('{{ ["<a>", "b&c"]|join(", ") }}', {}),
    ("{{ items|join(sep) }}", {"items": ["<x>", "&y", "\"z\""], "sep": "<&>"}),
    ("{{ items|join(sep) }}", {"items": ["<x>", "MARKUP:<y>"], "sep": "<&>"}),
    ("{{ items|join }}", {"items": [1, "<2>", "MARKUP:<3>&"]}),
    ('{{ "<b>&\'"|forceescape }}', {}),
    ("{{ x|forceescape }}", {"x": "MARKUP:<i>&amp;"}),
    ("<p{{ attrs|xmlattr }}>", {"attrs": {"class": "a\"b<", "data-x": "&y'", "skip": None}}),
    ("<p{{ attrs|xmlattr(false) }}>", {"attrs": {"title": "MARKUP:<b>&"}}),
    ('{{ "see http://x.com/?a=1&b=<2> & www.y.org/\\"q\\" now"|urlize }}', {}),
    ('{{ "mail a<b>@x.com or http://z.io"|urlize(10, true, target="_bl\\"ank&") }}', {}),
    ('{{ s|urlize(rel="no<f>&") }}', {"s": "go to https://ex.com/?q=<1>&r=\"2\""}),
    ('{{ "<const>&\'\\"" }}|{{ 1 ~ "<&>" }}|{{ "%s<%s" % ("&", ">") }}', {}),
    ("{% set x = '<s>&' %}{{ x }}{{ x|upper }}", {}),
]

MODULE = r'''import json
import os

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.basename(os.path.dirname(HERE))
DATA = json.load(open(os.path.join(HERE, "probes.json")))
CASES = DATA["cases"]
EXPECTED = DATA.get("expected", {})


def render(kind, case):
    if kind == "mistune":
        import mistune
        out = []
        for esc in (True, False):
            md = mistune.create_markdown(escape=esc, plugins=["abbr", "url", "strikethrough"])
            out.append(md(case))
        return out
    if kind == "markdown":
        import markdown
        out = []
        for fmt in ("html", "xhtml"):
            out.append(markdown.markdown(case, extensions=["abbr", "fenced_code", "attr_list", "tables"], output_format=fmt))
        return out
    if kind == "pymdownx":
        import markdown
        return [markdown.markdown(case, extensions=["pymdownx.superfences", "pymdownx.blocks.html", "attr_list"])]
    if kind == "jinja":
        import jinja2
        from markupsafe import Markup
        tmpl, ctx = case

        def conv(v):
            if isinstance(v, str) and v.startswith("MARKUP:"):
                return Markup(v[7:])
            if isinstance(v, list):
                return [conv(x) for x in v]
            if isinstance(v, dict):
                return {k: conv(x) for k, x in v.items()}
            return v
        out = []
        for auto in (False, True):
            env = jinja2.Environment(autoescape=auto)
            out.append(env.from_string(tmpl).render(**{k: conv(v) for k, v in ctx.items()}))
        return out
    raise ValueError(kind)


def outcome(kind, case):
    try:
        return render(kind, case)
    except Exception as e:  # the oracle's exception type is part of the expected behavior
        return ["EXC:" + type(e).__name__]


@pytest.mark.parametrize("i", range(len(CASES)))
def test_escape_probe(i):
    got = outcome(DATA["kind"], CASES[i])
    assert got == EXPECTED[str(i)], "output differs from the expected escaping"


if __name__ == "__main__":  # build time: record the oracle's outputs
    json.dump(dict(DATA, expected={str(i): outcome(DATA["kind"], c) for i, c in enumerate(CASES)}),
              open(os.path.join(HERE, "probes.json"), "w"), indent=1, ensure_ascii=False)
'''

CASES = {"mistune": MISTUNE, "markdown": MARKDOWN, "pymdownx": PYMDOWNX, "jinja": JINJA}
