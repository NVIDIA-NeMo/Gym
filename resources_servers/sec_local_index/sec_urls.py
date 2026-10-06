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
"""One parser for SEC EDGAR Archives document URLs.

Callers need the same URL broken apart in different shapes: the index keys
documents by unpadded CIK and a lowercased filename, filing caches key them by
zero-padded CIK, and filing metadata is addressed by dash-formatted accession.
Deriving those from one parse keeps a URL from resolving differently depending
on which caller looked at it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import unquote


# Accession is matched permissively rather than as the 18 digits EDGAR actually
# issues: a malformed one then resolves to a lookup miss instead of a parse
# failure that a caller might report as a malformed URL.
SEC_ARCHIVES_URL_RE = re.compile(r"sec\.gov/Archives/edgar/data/(\d+)/([0-9A-Za-z]+)/([^?#]*)")

ACCESSION_LENGTH = 18


@dataclass(frozen=True)
class SecArchivesUrl:
    """The CIK, accession and document path carried by an EDGAR Archives URL."""

    cik: str
    accession: str
    document: str

    @property
    def padded_cik(self) -> str:
        return self.cik.zfill(10)

    @property
    def unpadded_cik(self) -> str:
        return str(int(self.cik))

    @property
    def dashed_accession(self) -> str:
        if len(self.accession) != ACCESSION_LENGTH:
            return self.accession
        return f"{self.accession[:10]}-{self.accession[10:12]}-{self.accession[12:]}"

    @property
    def document_path(self) -> str:
        return self.document.strip("/")

    @property
    def document_basename(self) -> str:
        """The filename alone, percent-decoded and lowercased.

        Matches the key the index builder writes, so it must not be changed
        independently of the builder.
        """
        return unquote(self.document).strip("/").rsplit("/", 1)[-1].lower()

    @property
    def flat_document(self) -> str:
        """The document path collapsed into a single filename component."""
        return self.document.strip("/").replace("/", "_")


def parse_sec_archives_url(url: str) -> SecArchivesUrl | None:
    match = SEC_ARCHIVES_URL_RE.search(url)
    if not match:
        return None
    return SecArchivesUrl(cik=match.group(1), accession=match.group(2), document=match.group(3))
