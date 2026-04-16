#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Install GnuCOBOL and dependencies from source (no root required).
#
# Usage:
#   bash install_gnucobol.sh
#   GNUCOBOL_PREFIX=/custom/path bash install_gnucobol.sh

set -euo pipefail

M4_VERSION="1.4.19"
GMP_VERSION="6.3.0"
BDB_VERSION="18.1.40"
GNUCOBOL_VERSION="3.2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${GNUCOBOL_PREFIX:-${SCRIPT_DIR}/../.gnucobol}"
PREFIX="$(cd "$(dirname "$PREFIX")" && pwd)/$(basename "$PREFIX")"

BUILD_DIR="${PREFIX}/build"

echo "==> GnuCOBOL install prefix: ${PREFIX}"

# -------------------------------------------------------------------
# Pre-flight checks
# -------------------------------------------------------------------
for cmd in gcc make tar; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: '${cmd}' is required but not found on PATH." >&2
        exit 1
    fi
done

# xz is needed to extract .tar.xz archives (m4, GMP, GnuCOBOL)
if ! command -v xz &>/dev/null; then
    echo "ERROR: 'xz' is required to extract .tar.xz archives but not found." >&2
    echo "       Install with: apt-get install xz-utils" >&2
    exit 1
fi

# Verify gcc can actually produce executables (catches missing libc6-dev)
if ! echo 'int main(){return 0;}' | gcc -x c - -o /dev/null 2>/dev/null; then
    echo "ERROR: gcc cannot compile a simple program. Missing libc6-dev?" >&2
    echo "       Install with: apt-get install libc6-dev  (or: apt-get install build-essential)" >&2
    exit 1
fi

DOWNLOAD_CMD=""
if command -v wget &>/dev/null; then
    DOWNLOAD_CMD="wget -q -O"
elif command -v curl &>/dev/null; then
    DOWNLOAD_CMD="curl -sSL -o"
else
    echo "ERROR: Either 'wget' or 'curl' is required but neither was found." >&2
    exit 1
fi

mkdir -p "${PREFIX}" "${BUILD_DIR}"

# -------------------------------------------------------------------
# 0. m4 (only if not already installed — required by GMP)
# -------------------------------------------------------------------
if command -v m4 &>/dev/null; then
    echo "==> m4 found ($(command -v m4)), skipping."
elif [ -x "${PREFIX}/bin/m4" ]; then
    echo "==> m4 already built locally, skipping."
else
    echo "==> m4 not found — building m4 ${M4_VERSION} from source ..."
    cd "${BUILD_DIR}"
    M4_ARCHIVE="m4-${M4_VERSION}.tar.xz"
    if [ ! -f "${M4_ARCHIVE}" ]; then
        $DOWNLOAD_CMD "${M4_ARCHIVE}" "https://ftp.gnu.org/gnu/m4/${M4_ARCHIVE}"
    fi
    tar xf "${M4_ARCHIVE}"
    cd "m4-${M4_VERSION}"
    ./configure --prefix="${PREFIX}" --quiet
    make -j"$(nproc)" --quiet
    make install --quiet
    echo "==> m4 installed."
fi
# Ensure locally-built m4 is on PATH for GMP configure
export PATH="${PREFIX}/bin:${PATH}"

# -------------------------------------------------------------------
# 1. GMP (GNU Multiple Precision Library)
# -------------------------------------------------------------------
if [ -f "${PREFIX}/lib/libgmp.a" ] || [ -f "${PREFIX}/lib/libgmp.so" ]; then
    echo "==> GMP already installed, skipping."
else
    echo "==> Building GMP ${GMP_VERSION} ..."
    cd "${BUILD_DIR}"
    GMP_ARCHIVE="gmp-${GMP_VERSION}.tar.xz"
    if [ ! -f "${GMP_ARCHIVE}" ]; then
        $DOWNLOAD_CMD "${GMP_ARCHIVE}" "https://gmplib.org/download/gmp/${GMP_ARCHIVE}"
    fi
    tar xf "${GMP_ARCHIVE}"
    cd "gmp-${GMP_VERSION}"
    ./configure --prefix="${PREFIX}" --quiet
    # MAKEINFO=true prevents make from trying to regenerate .info docs
    # when container timestamps cause make to think .texi is newer.
    make -j"$(nproc)" MAKEINFO=true --quiet
    make install MAKEINFO=true --quiet
    echo "==> GMP installed."
fi

# -------------------------------------------------------------------
# 2. Berkeley DB
# -------------------------------------------------------------------
if [ -f "${PREFIX}/lib/libdb.a" ] || [ -f "${PREFIX}/lib/libdb.so" ]; then
    echo "==> Berkeley DB already installed, skipping."
else
    echo "==> Building Berkeley DB ${BDB_VERSION} ..."
    cd "${BUILD_DIR}"
    BDB_ARCHIVE="db-${BDB_VERSION}.tar.gz"
    if [ ! -f "${BDB_ARCHIVE}" ]; then
        $DOWNLOAD_CMD "${BDB_ARCHIVE}" "https://download.oracle.com/berkeley-db/${BDB_ARCHIVE}"
    fi
    tar xzf "${BDB_ARCHIVE}"
    cd "db-${BDB_VERSION}/build_unix"
    ../dist/configure --prefix="${PREFIX}" --quiet
    make -j"$(nproc)" --quiet
    # install_lib + install_utilities only — skip install_docs which fails
    # on BDB 18.1.40 due to missing bdb-sql/gsg_db_server doc dirs.
    make install_lib install_utilities install_include --quiet
    echo "==> Berkeley DB installed."
fi

# -------------------------------------------------------------------
# 3. GnuCOBOL
# -------------------------------------------------------------------
if [ -x "${PREFIX}/bin/cobc" ]; then
    echo "==> GnuCOBOL already installed, skipping."
else
    echo "==> Building GnuCOBOL ${GNUCOBOL_VERSION} ..."
    cd "${BUILD_DIR}"
    COBOL_ARCHIVE="gnucobol-${GNUCOBOL_VERSION}.tar.xz"
    if [ ! -f "${COBOL_ARCHIVE}" ]; then
        $DOWNLOAD_CMD "${COBOL_ARCHIVE}" \
            "https://sourceforge.net/projects/gnucobol/files/gnucobol/${GNUCOBOL_VERSION}/${COBOL_ARCHIVE}"
    fi
    tar xf "${COBOL_ARCHIVE}"
    cd "gnucobol-${GNUCOBOL_VERSION}"
    LD_LIBRARY_PATH="${PREFIX}/lib:${LD_LIBRARY_PATH:-}" \
    ./configure \
        CPPFLAGS="-I${PREFIX}/include" \
        LDFLAGS="-L${PREFIX}/lib -Wl,-rpath,${PREFIX}/lib" \
        LIBS="-lgmp" \
        BDB_CFLAGS="-I${PREFIX}/include" \
        BDB_LIBS="-L${PREFIX}/lib -Wl,-rpath,${PREFIX}/lib -ldb" \
        --prefix="${PREFIX}" \
        --quiet
    make -j"$(nproc)" MAKEINFO=true --quiet
    make install MAKEINFO=true --quiet
    echo "==> GnuCOBOL installed."
fi

# -------------------------------------------------------------------
# Verify
# -------------------------------------------------------------------
echo "==> Verifying installation ..."
"${PREFIX}/bin/cobc" --version
echo "==> GnuCOBOL is ready at ${PREFIX}/bin/cobc"
