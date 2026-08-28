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
| `<index>.sqlite.metadata` | for any index over 1 GB | Metadata-only copy that makes searches fast |

The sidecar is found automatically when it sits beside the index with a
`.metadata` suffix, so **copy the two together**. A large index without one is a
startup error rather than a slow server. See
[Metadata sidecar](#metadata-sidecar) and [Startup checks](#startup-checks).

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

### Why a separate file rather than a second table

Holding the metadata in its own table inside the index, with `body` moved to a
table of its own and `content=` pointed at that, looks like the same fix without
a second file. It was measured on a 27.7 GB corpus and is slower wherever it
matters: comparable below 8 concurrent readers, then falling behind to about
1.4x the latency, with a search throughput ceiling roughly 16% lower.

What the sidecar buys is a small file that many concurrent readers keep resident
in the page cache, and a narrow table sharing an inode with tens of gigabytes of
filing text does not reproduce that. If you are tempted to fold the sidecar into
the index to simplify the artifact story, measure under your real rollout
concurrency first — the single-process numbers point the other way.

## Startup checks

When `local_edgar_index_path` is set, the server verifies that:

- `documents` and `documents_fts` exist, and `documents` has every column a
  result needs — so a near-miss schema fails at startup instead of on the first
  search, mid-rollout
- a sidecar, if present, matches the index on schema version, document count and
  fingerprint
- the index is not larger than 1 GB while storing filing text in `documents`
  with no sidecar available

That last check is the one to know about. Searches remain correct without a
sidecar, but on a large corpus each one reads filing text to return metadata,
which costs tens of seconds and a rollout experiences as a hang rather than an
error. Refusing to start turns a silent slowdown into an obvious misconfiguration
— the common cause being an index copied to a new machine without its sidecar.
Small indexes are exempt because the penalty scales with the corpus, and an index
whose `documents` table holds no `body` needs no sidecar at all.

A configured-but-missing `local_edgar_metadata_path` is always an error,
whatever the size.

## Obtaining an index

Any file matching the schema above works, so an index can be copied between
machines — it is a single self-contained SQLite file plus its sidecar.

Building one means walking a corpus of downloaded filings, extracting text from
each document, and inserting rows into `documents` while keeping `documents_fts`
populated. That is a property of whichever pipeline downloads the filings rather
than of this resource server, so no builder ships here.
