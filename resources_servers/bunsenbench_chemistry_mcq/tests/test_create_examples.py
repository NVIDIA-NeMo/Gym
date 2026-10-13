# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from resources_servers.bunsenbench_chemistry_mcq.create_examples import AGENT_REF, build_example_rows


class TestCreateExamples:
    def test_build_example_rows_shape(self) -> None:
        rows = build_example_rows()
        assert len(rows) == 5
        for row in rows:
            assert row["agent_ref"] == AGENT_REF
            assert row["responses_create_params"]["input"]
            assert row["options"]
            assert row["expected_answer"]
            assert row["uuid"].startswith("bunsen:example:")
            assert row["metadata"]["source"] == "example"
            assert "question" not in row
            assert "options_text" not in row
            assert "grading_mode" not in row
