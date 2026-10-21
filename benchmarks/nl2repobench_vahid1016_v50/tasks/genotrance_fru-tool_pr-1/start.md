## fru Project Introduction and Goals

`fru` (FRU Tool) is a command-line utility and Python library for generating and
converting IPMI FRU binary data files. Every Field Replaceable Unit stores
inventory information — manufacturer, product name, manufacture date, serial
numbers, and other details — in a binary format defined by Intel's FRU
Information Storage specification for IPMI. Because building the binary payload
by hand is error-prone, FRU Tool converts between editable TOML text files and
conforming FRU binaries via a `frutool` CLI (`generate`, `dump`, `sample`
subcommands) and importable `fru_format` / `toml_format` modules.

## Natural Language Instructions (Prompt)

1. Create an importable Python package `fru` laid out under `src/fru/` with a
   `pyproject.toml` declaring the package metadata: name `fru`, version `4.1.0`,
   description "Read and write binary FRU files", MIT license, authors Kurt McKee
   and Ganesh Viswanathan, README file, and a `frutool = 'fru.console:run'` console
   script entry point. Use the poetry-core build backend
   (`requires = ["poetry-core>=1.0.0"`, `build-backend =
   "poetry.core.masonry.api"`). Python requirement `>= 3.8`; dependencies are
   `click ^8.0.0` and `tomli ^2.0.1` only for Python `< 3.11`.
2. The package's `__init__.py` must re-export `dump` and `load` from the
   `fru_format` submodule and set `__all__` accordingly. `__main__.py` must
   invoke `console.run()` so `python -m fru` is equivalent to `frutool`.
3. Implement `fru.shared` with helper functions returning default section
   dictionaries and name tuples: `get_default_common_section`,
   `get_default_board_section` / `get_board_section_names`,
   `get_default_chassis_section` / `get_chassis_section_names`,
   `get_default_product_section` / `get_product_section_names`, and
   `get_default_internal_section`. The common default is
   `{"format_version": 1, "size": 0}`; other defaults populate all string
   fields with `""`, integer codes with `0`, and empty `custom_fields` lists
   (the internal default holds `b""` data).
4. Implement `fru.exceptions` with a `FRUException(Exception)` base class plus
   `DateTimeException` and `TOMLException` subclasses.
5. Implement `fru.fru_format` for the binary side of the format:
   - `validate_checksum(blob, offset, length)` — sums all bytes in the area
     except the last with the last (checksum) byte; raise `ValueError("The data
     do not match the checksum")` unless the low byte of the sum is zero.
   - `extract_values(blob, offset, names)` — parse type/length-byte-delimited
     strings into a dictionary; values past `names` accumulate under
     `custom_fields`; stop when the end-of-fields marker `0xC1` is read; support
     plain ASCII (`encoding` bits 0/1) and 6-bit ASCII (`encoding == 2`).
   - `decode_6_bit_ascii(blob)` — unpack 3-byte groups into four 6-bit
     characters, each offset by `0x20`.
   - `load(path=None, blob=None)` — parse a full FRU structure (common header
     with area offsets in 8-byte units, optional internal, chassis, board, and
     product areas with per-area checksum validation) into a nested dictionary
     keyed `common`, `internal`, `chassis`, `board`, `product`, each carrying
     format versions, sizes, language codes, the board's `mfg_date_time`
     (little-endian 3-byte minute count), extracted strings, and custom fields.
     Specifying neither or both of `path` and `blob` raises `ValueError`.
   - `dump(data)` — build the binary blob: validate `[common]` presence and its
     `format_version` / `size` keys (raising `ValueError` with messages naming
     the missing section/key), serialize the internal area via `make_internal`
     and chassis/board/product areas via `make_chassis`, `make_board`,
     `make_product` (each merges defaults with user config, writes header
     version byte, encodes one language/code/type byte, packs strings as
     type/length bytes with `0xC0` set, appends `0xC1`, pads so the length ends
     at a multiple of 8 minus 3, prepends a 2-byte `{version, length-in-8-byte-
     units}` header, and appends a checksum byte), lay out areas sequentially
     and write the 8-byte common header with offset words, header checksum, and
     pad the result with zero bytes to `common`'s `size`; raise
     `ValueError("Too much content, does not fit")` if impossible.
6. Implement `fru.toml_format` for the text side:
   - `convert_str_to_minutes(stamp)` / `convert_minutes_to_str(minutes)` —
     translate between `"YYYY-MM-DD HH:MM"` strings and minute counts relative
     to 1996-01-01 00:00, validating the range up to 2027-11-24 20:15
     (`0xffffff` minutes) and raising `DateTimeException` with descriptive
     messages on bad formats or out-of-range values.
   - `repr_(value)` — render bools as `true`/`false`, ints as decimal, strings
     with backslash and quote escaping, lists as TOML inline arrays of values;
     raise `TOMLException` for unsupported types. `repr_internal(value)` —
     render a bytes object as a multi-line TOML array of `0x..` numbers, 16 per
     line, or `[]` for empty bytes.
   - `load(path=None, text=None)` — parse TOML (via `tomllib` on Python 3.11+,
     `tomli` otherwise) into the dictionary-of-sections form, merging user
     values over defaults; honor `include_internal`, `include_chassis`,
     `include_board`, `include_product`, and `include_multirecord` boolean keys
     in `[common]`, deleting excluded sections and removing the include keys;
     require listed keys (common size/format_version, board and product
     language_code, chassis type) to be ints and the board `mfg_date_time` to be
     a string converted to minutes (a zero default becomes `"1996-01-01
     00:00"`); normalize the internal section's `data` from a list of ints or a
     string into bytes, or read the file named by the internal `file` key
     relative to the TOML file's directory (raising `TOMLException` if the file
     is missing), removing the `file` key afterward; raise
     `FRUException` if neither `path` nor `text` was given.
   - `dump(data=None)` — render a complete, commented TOML document from the
     data dictionary (or a blank/default one when `data` is `None`), starting
     with a comment header naming `frutool` and its installed version (via
     `importlib.metadata.version("fru")`), including `[common]`,
     `[board]`, `[chassis]`, `[product]`, and `[internal]` sections with
     explanatory comments, `include_*` booleans computed from which sections
     were present, the manufacture date rendered via
     `convert_minutes_to_str`, and the internal bytes rendered via
     `repr_internal`; the result is dedented, stripped, and ends with a newline.
7. Implement `fru.console` as a click-based CLI: a `run` group with
   `@click.version_option()`, plus `generate`, `dump`, and `sample` commands
   added to the group. `generate` takes TOML and binary file arguments and a
   `--force` flag; it refuses to overwrite an existing FRU file (message +
   `sys.exit(1)`) unless `--force`, loads the TOML, dumps the binary, and on any
   `ValueError` prints the error's message and exits 1. `du