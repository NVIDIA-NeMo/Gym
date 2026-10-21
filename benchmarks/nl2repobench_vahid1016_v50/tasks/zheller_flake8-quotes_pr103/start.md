## flake8-quotes Project Introduction and Goals

flake8-quotes is a plugin for the flake8 linter that enforces a consistent quote style (single or double) in Python source code. It inspects Python tokens via the `tokenize` module and reports violations under the `Q0` warning prefix: `Q000` (bad inline quotes), `Q001` (bad quotes in multiline strings), `Q002` (bad quotes in docstrings), and `Q003` (outer quotes should be changed to avoid escaping inner quotes). The plugin supports flake8 2.x and 3.x option registration and is configurable via CLI flags and flake8 configuration files (`inline-quotes`, `multiline-quotes`, `docstring-quotes`, `avoid-escape`).

The goal of this task is to reconstruct the `flake8_quotes` package (version 3.2.0) from scratch, including the flake8 entry point registration, quote analysis logic, and docstring detection state machine, so that it is importable and installable in the provided environment.

## Natural Language Instructions (Prompt)

1. Create a Python package named `flake8_quotes` containing three modules: `__init__.py`, `__about__.py`, and `docstring_detection.py`.
2. `flake8_quotes/__about__.py` must define the module-level constant `__version__` set to the string `'3.2.0'`.
3. `flake8_quotes/__init__.py` must export a `QuoteChecker` class and a `Token` class, and must import `get_docstring_tokens` from `flake8_quotes.docstring_detection` so it is accessible as `flake8_quotes.get_docstring_tokens`.
4. The package must support flake8 stdin polyfilling: attempt to import `stdin_get_value` and `readlines` from `flake8.engine.pep8` (flake8 2.x); on `ImportError`, fall back to `flake8.utils.stdin_get_value` and `pycodestyle.readlines` (flake8 3.x).
5. `QuoteChecker` must be a flake8 plugin class with `name = __name__` and `version = __version__`, constructible as `QuoteChecker(tree, lines=None, filename='(none)')`, storing `filename` and `lines` as instance attributes (ignoring `tree`).
6. `QuoteChecker` must define three class-level dictionaries with alias keys:
   - `INLINE_QUOTES`: keys `'`, `"`, `'single'`, `'double'` mapping to dicts with `good_single` and `bad_single` keys; `'` maps to `{'good_single': "'", 'bad_single': '"'}` and `"` maps to `{'good_single': '"', 'bad_single': "'"}`.
   - `MULTILINE_QUOTES`: keys `'`, `"`, `'single'`, `'double'`, `'''`, `"""` with `good_multiline`, `good_multiline_ending`, and `bad_multiline` keys. For `'`: `good_multiline` is `'''`, `good_multiline_ending` is `'"""` (single quote followed by three double quotes, four characters), `bad_multiline` is `"""`. For `"`: `good_multiline` is `"""`, `good_multiline_ending` is `"'` + `'''` (a double quote followed by three single quotes), `bad_multiline` is `'''`.
   - `DOCSTRING_QUOTES`: keys `'`, `"`, `'single'`, `'double'`, `'''`, `"""` with `good_docstring` and `bad_docstring` keys. For `'`: `good_docstring` is `'''`, `bad_docstring` is `"""`; for `"`: `good_docstring` is `"""`, `bad_docstring` is `'''`.
7. `QuoteChecker` must provide a static method `_register_opt(parser, *args, **kwargs)` that registers an option on both flake8 3.x (via `parser.add_option`) and flake8 2.x: on `optparse.OptionError` or `TypeError`, it pops the `parse_from_config` keyword, retries `add_option`, and, when `parse_from_config` was true, appends the option's option string (leading dashes stripped) to `parser.config_options`.
8. `QuoteChecker.add_options(cls, parser)` must register, via `_register_opt`:
   - `--quotes`: deprecated alias for `--inline-quotes`, `action='store'`, `parse_from_config=True`, `type='choice'`, `choices=sorted(cls.INLINE_QUOTES.keys())`.
   - `--inline-quotes`: `default="'"`, `action='store'`, `parse_from_config=True`, `type='choice'`, `choices=sorted(cls.INLINE_QUOTES.keys())`.
   - `--multiline-quotes`: `default=None`, `action='store'`, `parse_from_config=True`, `type='choice'`, `choices=sorted(cls.MULTILINE_QUOTES.keys())`.
   - `--docstring-quotes`: `default=None`, `action='store'`, `parse_from_config=True`, `type='choice'`, `choices=sorted(cls.DOCSTRING_QUOTES.keys())`.
   - `--avoid-escape`: `default=None`, `action='store_true'`, `parse_from_config=True`.
   - `--no-avoid-escape`: `dest='avoid_escape'`, `default=None`, `action='store_false'`, `parse_from_config=False`.
9. `QuoteChecker.parse_options(cls, options)` must build a `cls.config` dict class attribute starting from the single-quote inline config, the `"""` multiline config, and the `"""` docstring config (i.e., defaults are single inline quotes and double triple-quoted multiline/docstring quotes). Then:
   - If `options.quotes` exists and is not None, emit a `DeprecationWarning`-style `warnings.warn` mentioning that `quotes` is deprecated in favor of `inline-quotes`, and update config from `INLINE_QUOTES[options.quotes]`; otherwise update from `INLINE_QUOTES[options.inline_quotes]`.
   - If `options.multiline_quotes` is present and not None, update config with `MULTILINE_QUOTES[options.multiline_quotes]`.
   - If `options.docstring_quotes` is present and not None, update config with `DOCSTRING_QUOTES[options.docstring_quotes]`.
   - If `options.avoid_escape` is present and not None, set `config['avoid_escape']` to it; otherwise default `config['avoid_escape']` to `True`.
10. `QuoteChecker.get_file_contents()` must return: stdin contents split with `splitlines(True)` when `self.filename` is `'stdin'`, `'-'`, or `None`; `self.lines` if truthy; otherwise `readlines(self.filename)`.
11. `QuoteChecker.run()` must yield tuples `(line, col, message, type(self))` for each detected quote error whose line number is not covered by a `# noqa` comment. `get_noqa_lines` collects start rows of COMMENT tokens whose string ends with `noqa`; tokenizer input is the list of file-content lines.
12. `QuoteChecker.get_quotes_errors(file_contents)` must tokenize the file, identify docstring tokens via `get_docstring_tokens`, and for each STRING token:
    - Strip any string prefix (e.g., `u`, `b`, `r`, case-insensitively) by locating the first occurrence of the final quote character.
    - Classify it as docstring, multiline (`unprefixed_string[0] * 3 == unprefixed_string[0:3]`), or inline.
    - Docstrings: skip if `config['good_docstring']` is in the string; otherwise yield Q002 at the token's start position.
    - Multiline strings: skip if `config['good_multiline']` appears anywhere in the string or the string ends with `config['good_multiline_ending']`; otherwise yield Q001.
    - Inline strings: let `string_contents` be the string without its outer quote characters. If the last quote char equals `config['good_single']`: skip unless `config['avoid_escape']` is on and `'r'` is not in the prefix; then yield Q003 only if `string_contents` contains `good_single` and does not contain `bad_single`. If the quote char is not the preferred one: yield Q000 unless `config['good_single']` appears in `string_contents`.
13. `Token` must be a simple wrapper around a 5-tuple token with properties `type`, `string`, `start` (the full `(row, col)` tuple), `start_row`, and `start_col`.
14. `docstring_detection.py` must implement a state machine with six module-level integer state constants (`STATE_EXPECT_MODULE_DOCSTRING`, `STATE_EXPECT_CLASS_COLON`, `STATE_EXPECT_CLASS_DOCSTRING`, `STATE_EXPECT_FUNCTION_COLON`, `STATE_EXPECT_FUNCTION_DOCSTRING`, `STATE_OTHER`, values 0–5), a `TOKENS_TO_IGNORE` list containing NEWLINE, INDENT, DEDENT, NL, and COMMENT token types, and a function `get_docstring_tokens(tokens)` returning the set of `Token` objects corresponding to docstrings: the first string token of th