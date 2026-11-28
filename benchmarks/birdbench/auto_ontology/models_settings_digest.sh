#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Print the digest that keys an Auto Ontology catalog build's cache by its models.
#
#   models_settings_digest.sh MODELS_ENV
#
# BuildKit leaves a secret's contents out of the cache key, so the catalog builds take this digest of MODELS_ENV as a
# build argument. Each key is cut to its type (sk-, nvapi-), which picks Auto Ontology's default models: another model,
# endpoint or key type rebuilds the catalog, and a new key of the same type does not. No key reaches the digest.
set -euo pipefail

sed -E -e '/^[[:space:]]*(#|$)/d' \
  -e 's/^([^=]*(KEY|TOKEN|SECRET|PASSWORD)[^=]*=)[[:space:]]*([a-z]+-)?.*$/\1\3/' \
  "${1:?usage: models_settings_digest.sh MODELS_ENV}" \
  | sort | sha256sum | cut -d' ' -f1
