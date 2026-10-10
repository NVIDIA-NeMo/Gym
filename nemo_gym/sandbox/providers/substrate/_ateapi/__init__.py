# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""gRPC bindings for the Substrate control API.

Generated from Substrate's ateapi.proto; regenerate with scripts/generate_ateapi_stubs.sh
rather than editing by hand.
"""

from nemo_gym.sandbox.providers.substrate._ateapi import ateapi_pb2, ateapi_pb2_grpc


__all__ = ["ateapi_pb2", "ateapi_pb2_grpc"]
