# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Render the five synthetic example charts and write data/example.jsonl.

Every chart is drawn from known values, so each reference answer is exact; numeric parts get an acceptable
range for reading error. Needs matplotlib and numpy (not dependencies of the server itself):

    python resources_servers/chartography/make_example_data.py
"""

import base64
import io
import json
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


OUT = Path(__file__).resolve().parent / "data/example.jsonl"


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=100, metadata={"Software": None})
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def pump_curves():
    flow = np.linspace(0, 80, 161)
    shutoff = {"200 mm": 32.0, "225 mm": 40.0, "250 mm": 50.0}
    head = lambda h0, q: h0 - 0.0045 * q**2  # noqa: E731
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, h0 in shutoff.items():
        ax.plot(flow, head(h0, flow), label=name)
    ax.set(xlabel="Flow (L/s)", ylabel="Head (m)", xlim=(0, 80), ylim=(0, 60), title="Pump performance curves")
    ax.grid(alpha=0.3)
    ax.legend(title="Impeller diameter")
    q = "At a flow of 35 L/s, what heads do the 225 mm and 250 mm impellers deliver? Give each to one decimal in m."
    a, b = head(40.0, 35), head(50.0, 35)
    g = (
        f"225 mm: {a:.1f} m (acceptable range {a - 1.0:.1f} to {a + 1.0:.1f} m)\n"
        f"250 mm: {b:.1f} m (acceptable range {b - 1.0:.1f} to {b + 1.0:.1f} m)"
    )
    return "curve_family", q, g, fig


def close_bars():
    quarters, north, south = ["Q1", "Q2", "Q3", "Q4"], [355, 380, 405, 398], [362, 371, 412, 390]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    x = np.arange(4)
    ax.bar(x - 0.2, north, 0.4, label="North plant")
    ax.bar(x + 0.2, south, 0.4, label="South plant")
    ax.set(xticks=x, xticklabels=quarters, ylabel="Output (t)", ylim=(300, 430), title="Quarterly output")
    ax.legend()
    q = "Which plant had the higher output in Q3?"
    return "grouped_bar_chart", q, "Plant: South plant", fig


def log_axis_gain():
    f = np.logspace(1, 4, 400)
    gain = 20 * np.log10(1 / np.sqrt(1 + (f / 450.0) ** 2))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogx(f, gain)
    ax.set(xlabel="Frequency (Hz)", ylabel="Gain (dB)", ylim=(-30, 2), title="Low-pass filter response")
    ax.grid(which="both", alpha=0.3)
    g = 20 * np.log10(1 / np.sqrt(1 + (2000 / 450.0) ** 2))
    q = "What is the gain at 2 kHz, in dB?"
    return "frequency_response", q, f"Gain: {g:.1f} dB (acceptable range {g - 1.0:.1f} to {g + 1.0:.1f} dB)", fig


def threshold_list():
    t = np.linspace(0, 60, 121)
    curves = {"S1": 40 + 0.95 * t, "S2": 55 + 0.45 * t, "S3": 30 + 1.20 * t, "S4": 62 + 0.30 * t}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, y in curves.items():
        ax.plot(t, y, label=name)
    ax.axhline(75, color="gray", linestyle="--", linewidth=1)
    ax.set(xlabel="Time (min)", ylabel="Temperature (°C)", ylim=(20, 110), title="Sensor temperatures")
    ax.grid(alpha=0.3)
    ax.legend()
    hot = [n for n, y in curves.items() if np.interp(40, t, y) > 75]
    q = "Which sensors are above 75 °C at 40 minutes? List them in legend order."
    return "line_chart_threshold", q, f"Sensors: {', '.join(hot)} (in this order)", fig


def count_bars():
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    rain = [92, 71, 84, 63, 55, 41, 38, 46, 79, 88, 97, 81]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(months, rain)
    ax.set(ylabel="Rainfall (mm)", ylim=(0, 110), title="Monthly rainfall")
    ax.grid(axis="y", alpha=0.3)
    q = "In how many months was rainfall above 80 mm?"
    return "bar_chart_count", q, f"Months: {sum(r > 80 for r in rain)}", fig


def main() -> None:
    rows = []
    for i, make in enumerate((pump_curves, close_bars, log_axis_gain, threshold_list, count_bars)):
        domain, question, golden, fig = make()
        rows.append(
            {
                "responses_create_params": {
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": question},
                                {"type": "input_image", "image_url": _png(fig), "detail": "auto"},
                            ],
                        }
                    ]
                },
                "verifier_metadata": {
                    "task_id": f"example-{i}",
                    "domain": domain,
                    "prompt": question,
                    "golden_answer": golden,
                },
            }
        )
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
