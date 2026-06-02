# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from responses_api_models.vllm_model.client import main


if __name__ == "__main__":
    from asyncio import run

    run(main())
