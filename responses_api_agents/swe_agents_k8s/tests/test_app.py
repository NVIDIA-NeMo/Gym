# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from app import SWEBenchK8sConfig

from nemo_gym.config_types import ModelServerRef


class TestSWEBenchK8sConfig:
    def test_config_defaults(self) -> None:
        config = SWEBenchK8sConfig(
            host="localhost",
            port=9003,
            name="test_swe_k8s",
            entrypoint="app.py",
            container_formatter=["docker://test/{instance_id}"],
            swebench_tests_timeout=900,
            model_server=ModelServerRef(type="responses_api_models", name="test_model"),
            concurrency=1,
        )
        assert config.k8s_namespace == "default"
        assert config.workspace_pvc_name == "swe-workspace"
        assert config.setup_pvc_name == "swe-setup"
        assert config.workspace_mount_path == "/nemogym-workspace"
        assert config.setup_mount_path == "/nemogym-setup"
        assert config.k8s_memory_limit == "32Gi"

    def test_config_overrides(self) -> None:
        config = SWEBenchK8sConfig(
            host="localhost",
            port=9003,
            name="test_swe_k8s",
            entrypoint="app.py",
            container_formatter=["docker://test/{instance_id}"],
            swebench_tests_timeout=900,
            model_server=ModelServerRef(type="responses_api_models", name="test_model"),
            concurrency=1,
            k8s_namespace="custom-ns",
            workspace_pvc_name="my-workspace",
            setup_pvc_name="my-setup",
            k8s_memory_limit="64Gi",
            k8s_cpu_limit="8",
        )
        assert config.k8s_namespace == "custom-ns"
        assert config.workspace_pvc_name == "my-workspace"
        assert config.k8s_memory_limit == "64Gi"
