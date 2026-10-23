# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the shared Gym test-data inventory without importing server dependencies."""

import argparse
import os
from pathlib import Path


IFBENCH_COMMIT = "c6767a19bd82ac0536cab950f2f8f6bcc6fabe7c"  # pragma: allowlist secret
REQUIRED_FILES = (
    f"ifbench/{IFBENCH_COMMIT}/REVISION",
    f"ifbench/{IFBENCH_COMMIT}/instructions.py",
    f"ifbench/{IFBENCH_COMMIT}/instructions_util.py",
    f"ifbench/{IFBENCH_COMMIT}/instructions_registry.py",
    "nltk_data/tokenizers/punkt/english.pickle",
    "nltk_data/tokenizers/punkt_tab/english/abbrev_types.txt",
    "nltk_data/tokenizers/punkt_tab/english/collocations.tab",
    "nltk_data/tokenizers/punkt_tab/english/ortho_context.tab",
    "nltk_data/tokenizers/punkt_tab/english/sent_starters.txt",
    "nltk_data/corpora/stopwords/english",
    "nltk_data/taggers/averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng.classes.json",
    "nltk_data/taggers/averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng.tagdict.json",
    "nltk_data/taggers/averaged_perceptron_tagger_eng/averaged_perceptron_tagger_eng.weights.json",
    "SHA256SUMS",
)


def check_test_data(root: Path) -> None:
    """Require nonempty files and the pinned IFBench revision in a mounted data root."""
    missing = [
        relative
        for relative in REQUIRED_FILES
        if not (root / relative).is_file() or not (root / relative).stat().st_size
    ]
    if missing:
        raise FileNotFoundError(f"Missing or empty Gym test data under {root}: {', '.join(missing)}")
    if (root / REQUIRED_FILES[0]).read_text().strip() != IFBENCH_COMMIT:
        raise ValueError(f"IFBench revision mismatch under {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=os.environ.get("NEMO_GYM_TEST_DATA_ROOT"))
    args = parser.parse_args()
    if not args.root:
        parser.error("pass ROOT or set NEMO_GYM_TEST_DATA_ROOT")
    check_test_data(Path(args.root))
    print(f"Gym test-data inventory passed: {len(REQUIRED_FILES)} files under {args.root}")


if __name__ == "__main__":
    main()
