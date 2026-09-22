# sec_local_index

Shared SEC search code. This is a library, not a resources server: it has no
`configs/` directory and nothing starts it. Two servers import it.

`finance_sec_search` and `finance_agent_v2` both expose an `edgar_search` tool
and both can answer it either from sec-api.io or from a local corpus. Without a
shared library each server would need its own copy of the search engine, the
sec-api.io call and the HTML reduction, and the copies would drift apart
without anyone noticing.

## Modules

| Module | What it owns |
| --- | --- |
| `local_edgar_search.py` | The SQLite FTS5 search engine, query translation, result shaping, and the metadata sidecar |
| `live_edgar_search.py` | The sec-api.io full-text-search call |
| `edgar_search_service.py` | Argument coercion, normalization, date clamping and error serialization, plus `resolve_sec_mode` |
| `html_text.py` | The HTML-to-text reduction `parse_html_page` returns |
| `sec_urls.py` | Parsing SEC Archives URLs into CIK, accession and document parts |
| `cache.py` | `ToolCache`, the disk-backed tool response cache |
| `scripts/build_local_edgar_metadata.py` | Builds the metadata sidecar beside an index |

## Choosing a mode

Both servers take a `sec_mode` setting of `live` or `local`. Left unset it
follows `local_edgar_index_path`: local when an index is configured, live
otherwise. Asking for local mode without an index fails at startup rather than
once per search.

Local mode makes no network call, which is what training throughput needs. Live
mode is the one that matches the published benchmark.

The cutoff date is per server, not a library default: `finance_sec_search` uses
2025-04-07 and `finance_agent_v2` uses upstream's `MAX_END_DATE`. `max_end_date`
is a required argument so a caller cannot silently inherit the other's window.

## Staying aligned with upstream

`finance_agent_v2` calls Vals' `finance_agent` tools directly.
`finance_sec_search` cannot import that package without also taking on a
multi-provider model SDK stack, so `live_edgar_search.py` and `html_text.py`
restate two small pieces of it.

Those copies are pinned by
`resources_servers/finance_agent_v2/tests/test_live_edgar_conformance.py`, which
runs both implementations over one input matrix and fails when they part. It
lives in that server's test suite because that is the venv where `finance_agent`
is installed, so bumping the upstream pin runs it.

## Building an index

See [docs/local-edgar-index.md](docs/local-edgar-index.md) for the index schema
and how to build the metadata sidecar.
