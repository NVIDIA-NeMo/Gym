# Local EDGAR index

The `edgar_search` tool answers full-text queries over SEC filings from a
read-only SQLite file instead of a hosted API, so rollouts do not depend on
network availability or request rate limits.

It is a different tool from `sec_filing_search`, which looks up filing metadata
by ticker. `edgar_search` searches the *text* of filings and returns filing
metadata for the matches, ranked by relevance.

Set `local_edgar_index_path` to enable the tool. When it is unset, the tool
reports itself unavailable and the rest of the server works normally.

## Files

| File | Required | Purpose |
|------|----------|---------|
| `<index>.sqlite` | yes | Filing text and metadata |
| `<index>.sqlite.metadata` | no | Metadata-only copy that makes searches fast |

The sidecar is found automatically when it sits beside the index with a
`.metadata` suffix. See [Metadata sidecar](#metadata-sidecar).

## Index schema

Two objects are required.

```sql
CREATE TABLE documents (
    id               INTEGER PRIMARY KEY,
    accession_number TEXT NOT NULL,
    cik              TEXT NOT NULL,
    ticker           TEXT NOT NULL,
    company_name     TEXT NOT NULL,
    form_type        TEXT NOT NULL,
    document_type    TEXT NOT NULL,
    description      TEXT,
    filing_date      TEXT NOT NULL,
    url              TEXT NOT NULL,
    body             TEXT NOT NULL
);

CREATE VIRTUAL TABLE documents_fts USING fts5(
    body,
    content='documents',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2',
    prefix='2 3 4'
);
```

`content_rowid='id'` is what lets a full-text hit be joined back to its row, so
it cannot be omitted or renamed. Additional columns on `documents` are ignored.

Searches filter on `cik`, `form_type` and `filing_date` together, so an index
over those three makes filter-only browsing usable on a large corpus:

```sql
CREATE INDEX documents_filters ON documents(cik, form_type, filing_date);
```

`prefix='2 3 4'` is what makes trailing wildcards such as `artific*` resolve
without scanning the whole term list. Queries still work without it, only more
slowly.

## Column formats

Three of these will silently return wrong or empty results if stored
differently, so they are worth getting right.

**`cik`** — store with leading zeros stripped, the form `str(int(cik))`
produces. Incoming CIKs are normalized that way before an exact string
comparison, so an index storing `0000320193` will never match a request for
`320193`, and CIK-filtered searches quietly return nothing.

**`filing_date`** — store as `YYYY-MM-DD`. Date ranges compare as strings, so
any other layout puts filings outside the range the caller asked for.

**`ticker`** — store an empty string, not `NULL`, when a filing has no ticker.
The tool converts empty to `null` on the way out.

`body` holds the filing text that gets indexed. It is never returned to the
agent; results contain filing metadata and URLs only.

## Optional provenance

An `index_metadata` table of `key`/`value` text pairs is read by nothing and is
a good place to record how a corpus was built.

## Metadata sidecar

Filing text and filing metadata share one table, so ranking a common term means
reading tens of thousands of multi-kilobyte rows just to reach the handful of
columns a result needs. The sidecar holds the same columns without the text —
hundreds of megabytes rather than tens of gigabytes — small enough to stay in
the page cache, which turns those reads into memory hits. Results are identical
either way; only the source of the metadata changes.

Build it once per index:

```bash
python resources_servers/finance_sec_search/scripts/build_local_edgar_metadata.py \
  --index /path/to/index.sqlite
```

The default output path is the one the server discovers on its own, so no
config change is needed. Expect common queries to drop from tens of seconds to
well under a second.

**Rebuild the sidecar whenever the index is rebuilt.** The two are joined on row
id, so a sidecar from a different index would pair filings with the wrong
metadata. To prevent that, the sidecar records the document count and a
fingerprint sampled from the index it came from, and the server refuses to start
against an index that does not match.

## Startup checks

When `local_edgar_index_path` is set, the server verifies the index has the
required tables, and, if a sidecar is present, that its schema version,
document count and fingerprint match the index. A configured-but-unusable
sidecar is a startup error rather than a silent fallback, because searches
would still be correct without it but orders of magnitude slower.

## Obtaining an index

Any file matching the schema above works, so an index can be copied between
machines — it is a single self-contained SQLite file plus its sidecar.

Building one means walking a corpus of downloaded filings, extracting text from
each document, and inserting rows into `documents` while keeping `documents_fts`
populated. That is a property of whichever pipeline downloads the filings rather
than of this resource server, so no builder ships here.
