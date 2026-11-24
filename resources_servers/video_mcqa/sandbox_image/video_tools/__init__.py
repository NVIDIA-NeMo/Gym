# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The watch_video and view_image tools of the video MCQA sandbox image.

The OpenCode plugin ``video_tools.mjs`` runs ``python3 -I -m video_tools.cli`` once per tool call. watch_video
decodes the frames the model asks for and returns each as a JPEG image, with a label that gives its sampling time;
view_image returns an image file unchanged. ``contract`` holds the sampling rules and needs only Pydantic, so the
resources server validates the same configuration; ``media`` needs FFmpeg, NumPy and Pillow, which the image has.
"""
