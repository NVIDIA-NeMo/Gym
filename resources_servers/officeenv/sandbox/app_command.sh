#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# An OfficeEnv app as a command in a `cli` episode's sandbox: `<app> <command> [--parameter=value ...]`, `<app> --help`.
# The server writes this file once and links it under each connected app's name, so the name it is run by is the
# app. It sends the arguments, NUL-separated, to the episode's OfficeEnv server (/v1/cli), which runs them as
# OfficeEnv's CLI exposure does, then prints the answer as a command would: the result on stdout with exit 0, or the
# app's error on stderr with exit 1. POSIX sh and curl only, so it runs on a bare image. It names itself in its
# User-Agent (app.py's CLI_USER_AGENT), without which the server refuses the call: the commands are the way in. The
# sandbox gets it without comments, so reading it shows a company CLI and nothing of the environment behind it.
# A shell started without the sandbox's environment (some harnesses pass their shell a few variables only) reads the
# API's URL and token from the file the server writes (app.py's CREDENTIALS_PATH).
if [ -z "${COMPANY_API_URL-}" ] && [ -r /etc/company-api.env ]; then
  . /etc/company-api.env
fi
body=$(mktemp) && head=$(mktemp) || exit 125
printf '%s\0' "${0##*/}" "$@" | curl -sS -X POST --data-binary @- -D "$head" -o "$body" \
  -A "${0##*/}-cli/1.0" -H "Authorization: Bearer ${COMPANY_API_TOKEN-}" \
  "${COMPANY_API_URL-}/v1/cli"
sent=$?
code=$(tr -d '\r' < "$head" | sed -n 's/^[Xx]-[Ee][Xx][Ii][Tt]-[Cc][Oo][Dd][Ee]: *\([0-9][0-9]*\).*/\1/p' | tail -n 1)
if [ "$sent" -ne 0 ]; then
  code=125
elif [ -z "$code" ]; then
  # no command ran (a refused token, an unknown episode): the server's answer is the error
  cat "$body" >&2
  code=1
elif [ "$code" -eq 0 ]; then
  cat "$body"
else
  cat "$body" >&2
fi
rm -f "$body" "$head"
exit "$code"
