# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Constants for the CVDP benchmark system.

This module centralizes commonly used constants across the codebase to improve
maintainability and reduce duplication.
"""

# ----------------------------------------
# Data Categories
# ----------------------------------------

# Categories that require code comprehension evaluation
CODE_COMPREHENSION_CATEGORIES = [6, 8, 9, 10]

# Categories that use LLM-based subjective scoring (subset of code comprehension categories)
LLM_SUBJECTIVE_CATEGORIES = [9, 10]

# Categories that use BLEU-based scoring for reporting (subset of code comprehension categories)
BLEU_SCORING_CATEGORIES = [6, 8]

# Categories that generate code (commented out in original - can be enabled if needed)
CODE_GEN_CATEGORIES = [2, 3, 4, 5, 7, 12, 13, 14, 16]

# Categories that require commercial EDA verification tools and license network connectivity
# Note: This is category-based detection only. Additional datapoints in other categories
# (e.g., cid003, cid005) may also require commercial EDA tools if they contain __VERIF_EDA_IMAGE__
# template variables. Use requires_commercial_eda_tools() for comprehensive detection.
VERIF_EDA_CATEGORIES = [12, 13, 14]

# ----------------------------------------
# Scoring Configuration
# ----------------------------------------

# Scoring behavior modes
SCORING_MODE_THRESHOLD = "threshold"  # Traditional binary pass/fail based on threshold
SCORING_MODE_SCORE = "score"  # Use actual scores (0-1 range) as fractional pass rates

# Default scoring thresholds and parameters
SCORING_CONFIG = {
    # LLM retry configuration
    "LLM_RETRY_COUNT_DEFAULT": 3,
    # Subjective scoring thresholds
    "SUBJECTIVE_THRESHOLD_DEFAULT": 0.7,
    # Text similarity thresholds
    "ROUGE_THRESHOLD": 0.40,
    "BLEU_THRESHOLD": 0.40,
    # N-gram configuration
    "N_GRAM_DEFAULT": 2,
    # Model timeout defaults (in seconds)
    "MODEL_TIMEOUT_DEFAULT": None,
    "TASK_TIMEOUT_DEFAULT": None,
    "QUEUE_TIMEOUT_DEFAULT": None,
    # Scoring behavior configuration
    "DEFAULT_SCORING_MODE": SCORING_MODE_THRESHOLD,  # Default to traditional behavior
}

# Category-specific scoring mode configuration
# This determines whether each category uses threshold-based or score-based scoring
CATEGORY_SCORING_MODES = {
    # BLEU-based categories use score-based scoring by default
    6: SCORING_MODE_SCORE,  # BLEU scoring
    8: SCORING_MODE_SCORE,  # BLEU scoring
    # LLM subjective categories use score-based scoring
    9: SCORING_MODE_SCORE,  # LLM subjective (score-based)
    10: SCORING_MODE_SCORE,  # LLM subjective (score-based)
    # Other categories default to threshold-based (traditional behavior)
    # Categories not listed here will use SCORING_CONFIG['DEFAULT_SCORING_MODE']
}

# ----------------------------------------
# Backward Compatibility Aliases
# ----------------------------------------

# For easy importing of individual constants
LLM_RETRY_COUNT_DEFAULT = SCORING_CONFIG["LLM_RETRY_COUNT_DEFAULT"]
SUBJECTIVE_THRESHOLD_DEFAULT = SCORING_CONFIG["SUBJECTIVE_THRESHOLD_DEFAULT"]
ROUGE_THRESHOLD = SCORING_CONFIG["ROUGE_THRESHOLD"]
BLEU_THRESHOLD = SCORING_CONFIG["BLEU_THRESHOLD"]
N_GRAM_DEFAULT = SCORING_CONFIG["N_GRAM_DEFAULT"]

# Legacy alias for backward compatibility (was [9, 10] hardcoded)
LLM_SCORE_CATEGORIES = LLM_SUBJECTIVE_CATEGORIES

# Alias for BLEU scoring categories
BLEU_SCORE_CATEGORIES = BLEU_SCORING_CATEGORIES

# ----------------------------------------
# License Network Configuration
# ----------------------------------------

# License network requirements
LICENSE_CONFIG = {
    # Default license network name for EDA tools
    "DEFAULT_LICENSE_NETWORK": "licnetwork",
    # Categories that require license network connectivity
    "LICENSE_REQUIRED_CATEGORIES": VERIF_EDA_CATEGORIES,
}

# ----------------------------------------
# Helper Functions
# ----------------------------------------


def get_scoring_mode(category_num: int) -> str:
    """
    Get the scoring mode for a given category number.

    Args:
        category_num: The category number (e.g., 6, 8, 9, 10)

    Returns:
        SCORING_MODE_THRESHOLD or SCORING_MODE_SCORE
    """
    return CATEGORY_SCORING_MODES.get(category_num, SCORING_CONFIG["DEFAULT_SCORING_MODE"])


def is_score_based_category(category_num: int) -> bool:
    """
    Check if a category uses score-based scoring (0-1 fractional pass rates).

    Args:
        category_num: The category number (e.g., 6, 8, 9, 10)

    Returns:
        True if category uses score-based scoring, False for threshold-based
    """
    return get_scoring_mode(category_num) == SCORING_MODE_SCORE


def is_threshold_based_category(category_num: int) -> bool:
    """
    Check if a category uses threshold-based scoring (traditional binary pass/fail).

    Args:
        category_num: The category number (e.g., 6, 8, 9, 10)

    Returns:
        True if category uses threshold-based scoring, False for score-based
    """
    return get_scoring_mode(category_num) == SCORING_MODE_THRESHOLD


# ----------------------------------------
# Example Configurations
# ----------------------------------------

# Example 1: Switch LLM categories to use score-based scoring
# To enable score-based scoring for LLM categories 9 and 10, uncomment these lines:
# CATEGORY_SCORING_MODES[9] = SCORING_MODE_SCORE   # Enable LLM score-based scoring
# CATEGORY_SCORING_MODES[10] = SCORING_MODE_SCORE  # Enable LLM score-based scoring

# Example 2: Revert BLEU categories to threshold-based scoring
# To revert categories 6 and 8 to traditional threshold-based scoring, uncomment these lines:
# CATEGORY_SCORING_MODES[6] = SCORING_MODE_THRESHOLD   # Revert to threshold-based
# CATEGORY_SCORING_MODES[8] = SCORING_MODE_THRESHOLD   # Revert to threshold-based

# Example 3: Change default scoring mode for all unlisted categories
# To make score-based scoring the default for all categories, uncomment this line:
# SCORING_CONFIG['DEFAULT_SCORING_MODE'] = SCORING_MODE_SCORE
