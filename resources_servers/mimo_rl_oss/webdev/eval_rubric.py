# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""The seven-dimension evaluation rubric. **The wording below is the ruler itself.**

One vision call returns seven dimensions, and the evaluation score is an equal-weight mean
of three of them::

    score = (visual + query_fulfillment + premium_assets) / 3
    visual = mean(layout_integrity, typography_hierarchy, color_harmony,
                  whitespace, content_richness)

``BANDS`` and ``RICHNESS`` are kept **byte for byte** as the run that produced this arm's
published numbers used them, and they are in Chinese for the same reason. They are not
documentation of a scoring policy -- they ARE the scoring policy, handed to the judge model
verbatim. Translating or rewording them measures something else and makes every historical
result incomparable. To change the rubric, copy this file and change ``RUBRIC_ID``; a test
pins the rendered prompt's hash so an accidental edit fails loudly instead of silently
shifting the reward distribution.

Two deliberate differences from the earlier four-term formula:

* The fourth term (a static code check, weight 0.1) is gone. Its dominant component was an
  INVERSE indicator: bucketing by ``premium_assets``, rows in [0, 0.2) averaged 0.900 on it
  while rows in [0.8, 1.0] averaged only 0.425, with ``visual`` rising monotonically across
  the same buckets. It was paying a bonus for not daring to reference external assets --
  the highest-scoring arm on that term was the one that referenced the fewest (0.823,
  against 0.638 and 0.648 for the RL arms).
* The remaining three are equal-weight rather than 0.6 / 0.1 / 0.2. ``premium_assets``
  separates the arms most sharply (0.417 against 0.619, a gap of 0.20, where ``visual``
  differs by 0.042 and ``query`` by 0.054), so equal weighting stops a 0.6-weighted
  ``visual`` from drowning out the dimension that actually discriminates.
"""

BANDS = """1. layout_integrity（布局完整性）
- [0.0, 0.2)：布局严重损坏，核心内容大面积重叠、截断、溢出或无法阅读
- [0.2, 0.4)：多处明显布局错误，显著影响主要内容使用
- [0.4, 0.6)：主体可用，但有一处严重或多处明显的对齐、重叠、超框问题
- [0.6, 0.8)：整体完整，仅有少量轻微对齐、间距或边界问题
- [0.8, 1.0]：布局完整稳定，未见可感知的重叠、错位、截断或溢出

2. typography_hierarchy（排版层级）
- [0.0, 0.2)：文字普遍不可读，字号/层级严重失控
- [0.2, 0.4)：标题正文层级混乱，多处字号、行高或字重明显不当
- [0.4, 0.6)：基本可读，但层级辨识或排版节奏存在明显缺陷
- [0.6, 0.8)：层级清楚、阅读顺畅，仅有少量轻微排版问题
- [0.8, 1.0]：标题、正文、辅助文字层级鲜明且比例、行高、字重均成熟

3. color_harmony（配色与对比）
- [0.0, 0.2)：配色严重冲突或关键文字几乎无法辨认
- [0.2, 0.4)：多处颜色不协调、对比不足，明显损害观感或可读性
- [0.4, 0.6)：基本可用，但色彩体系、对比或一致性有明显问题
- [0.6, 0.8)：配色协调且可读，仅有少量轻微不一致或对比问题
- [0.8, 1.0]：色彩体系统一成熟，对比清晰，无明显问题

4. whitespace（留白与密度）
- [0.0, 0.2)：页面极度拥挤或空洞，信息组织基本失效
- [0.2, 0.4)：多处明显过密/过空，严重破坏浏览节奏
- [0.4, 0.6)：整体可浏览，但局部密度、间距或区块节奏有明显问题
- [0.6, 0.8)：留白与内容密度总体合理，仅有少量轻微问题
- [0.8, 1.0]：留白、间距和区块节奏自然一致，密度控制成熟

{richness}
{nq}. query_fulfillment（需求实现度）
- [0.0, 0.2)：截图可见内容基本没有实现 query，或主题完全错误
- [0.2, 0.4)：只实现少量表面元素，多数核心要求缺失
- [0.4, 0.6)：实现主要方向，但多个关键板块、要素或功能界面明显缺失
- [0.6, 0.8)：主体要求已实现，仅有少量次要缺失或偏差
- [0.8, 1.0]：截图可见范围内完整、准确地实现 query 的主题、板块、要素和功能界面

{na}. premium_assets（高级素材与质感）
- [0.0, 0.2)：无有效视觉素材，或主要依赖简陋 emoji、裸占位、低质/模糊图像
- [0.2, 0.4)：素材较普通或粗糙，缺乏统一设计感；少量装饰不能形成质感
- [0.4, 0.6)：素材清晰且基本协调，但主要是普通图片或纯 CSS 视觉，没有高级素材
- [0.6, 0.8)：使用了高品质真实图片、艺术化图像、成体系插画或精致 mockup，整体质感良好但仍有小缺陷
- [0.8, 1.0]：高级素材质量突出、体系完整，并与版式高度融合，达到成熟专业作品水准"""

RICHNESS = """5. content_richness（内容丰富度）
- [0.0, 0.2)：近乎空壳，一屏都撑不满，只有标题、空架子或极少占位内容
- [0.2, 0.4)：只有 1–2 个简单板块，内容高度重复、空泛或以占位为主
- [0.4, 0.6)：已有若干实际板块，但数量或内容深度明显不足以支撑完整网站
- [0.6, 0.8)：板块较完整、内容较具体，仅少数部分偏单薄
- [0.8, 1.0]：约 4 个及以上成形板块，内容具体充实，足以支撑完整网站体验"""


VISUAL_KEYS = ("layout_integrity", "typography_hierarchy", "color_harmony", "whitespace", "content_richness")
SCORE_KEYS = ("visual", "query_fulfillment", "premium_assets")  # 等权平均的三项
ALL_KEYS = VISUAL_KEYS + ("query_fulfillment", "premium_assets")
RUBRIC_ID = "rva1:mean(visual,query,asset)"


def build_prompt() -> str:
    """评测用的 7 维 prompt（含 content_richness）。`{query}` 占位符留给调用方 format。"""
    dims = BANDS.format(richness=RICHNESS, nq=6, na=7)
    return f"""你是网页 UI 评审。只依据 fullpage 截图可见证据评分。前 5 项构成 visual：前四项只评客观视觉问题，第五项评完整网站所需的信息量；第 6 项依据 query；第 7 项评素材品质。各维独立判断，不要因为某一维表现好或差而连带改变其它维。

query: {{query}}

每个维度输出 0 到 1 的连续分数，可使用任意小数。先依据该维的五段描述确定分数所在区间，再根据问题严重程度、数量和影响在区间内连续取值；不要只输出区间端点或固定档位。五段边界统一为 [0.0,0.2)、[0.2,0.4)、[0.4,0.6)、[0.6,0.8)、[0.8,1.0]。临界情况选择与可见证据最匹配的相邻区间。premium_assets 没有高级素材时不得进入 [0.6,1.0]。

{dims}

空白、严重加载失败或完全不可见时，所有维度输出 0。仅输出 JSON，不要 markdown：
{{{{"layout_integrity":0.0,"typography_hierarchy":0.0,"color_harmony":0.0,"whitespace":0.0,"content_richness":0.0,"query_fulfillment":0.0,"premium_assets":0.0,"reason":"核心依据"}}}}"""


__all__ = ["BANDS", "RICHNESS", "VISUAL_KEYS", "SCORE_KEYS", "ALL_KEYS", "RUBRIC_ID", "build_prompt"]
