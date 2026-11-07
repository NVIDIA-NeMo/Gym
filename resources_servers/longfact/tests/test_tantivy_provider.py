# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from resources_servers.longfact.tantivy_provider import TantivyProvider


def test_document_value_normalizes_tantivy_scalar_and_list_fields():
    assert TantivyProvider._value({"docid": [123]}, "docid") == "123"
    assert TantivyProvider._value({"docid": "abc"}, "docid") == "abc"
    assert TantivyProvider._value({"docid": []}, "docid") == ""
    assert TantivyProvider._value({}, "docid") == ""
