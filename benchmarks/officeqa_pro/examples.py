# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Example questions about the June 2001 bulletin. They are separate from the gated Pro questions.

Each question refers to the bulletin's capsule analysis of second-quarter receipts. The monthly table on the next
page rounds each month separately. For three sources, the monthly values add up to 0.1 billion more or less than the
quarterly figure in the analysis.
"""

_SOURCE = "According to the capsule analysis of second-quarter receipts in the June 2001 Treasury Bulletin,"

EXAMPLES: tuple[tuple[str, str, str], ...] = (
    (
        "officeqa_example_1",
        f"{_SOURCE} what were individual income tax receipts, net of refunds, for the second quarter of fiscal 2001, "
        "in billions of dollars?",
        "217.3",
    ),
    (
        "officeqa_example_2",
        f"{_SOURCE} what were net corporate income tax receipts for the second quarter of fiscal 2001, in billions "
        "of dollars?",
        "26.4",
    ),
    (
        "officeqa_example_3",
        f"{_SOURCE} what were employment taxes and contributions receipts for the second quarter of fiscal 2001, in "
        "billions of dollars?",
        "172.6",
    ),
    (
        "officeqa_example_4",
        f"{_SOURCE} what were net excise tax receipts plus net estate and gift tax receipts for the second quarter of "
        "fiscal 2001, in billions of dollars?",
        "22.9",
    ),
    (
        "officeqa_example_5",
        f"{_SOURCE} by what percent did individual income tax receipts, net of refunds, grow over the comparable "
        "quarter of fiscal 2000? Use the receipts and the increase stated there, and round to two decimal places.",
        "7.68",
    ),
)
