# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os
import shutil
import stat
import tempfile

# imageio-ffmpeg is installed from source (no bundled binary) to avoid shipping
# royalty-bearing codec executables. moviepy (a stirrup dependency) calls
# imageio_ffmpeg.get_ffmpeg_exe() at import time — set IMAGEIO_FFMPEG_EXE so
# the import succeeds. The stirrup agent never encodes or decodes video.
if not os.environ.get("IMAGEIO_FFMPEG_EXE"):
    _system_ffmpeg = shutil.which("ffmpeg")
    if _system_ffmpeg:
        os.environ["IMAGEIO_FFMPEG_EXE"] = _system_ffmpeg
    else:
        _stub = os.path.join(tempfile.mkdtemp(), "ffmpeg")
        with open(_stub, "w") as _f:
            _f.write("#!/bin/sh\n")
        os.chmod(_stub, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)
        os.environ["IMAGEIO_FFMPEG_EXE"] = _stub
