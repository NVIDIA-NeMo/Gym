# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the shared Gym test-data inventory without importing server dependencies."""

import argparse
import hashlib
import os
import shutil
import tempfile
from pathlib import Path


IFBENCH_COMMIT = "c6767a19bd82ac0536cab950f2f8f6bcc6fabe7c"  # pragma: allowlist secret
SACREBLEU_MODEL = "sacrebleu/models/flores200sacrebleuspm"
SACREBLEU_MODEL_SHA256 = "14bb8dfb35c0ffdea7bc01e56cea38b9e3d5efcdcb9c251d6b40538e1aab555a"
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
    SACREBLEU_MODEL,
    "SHA256SUMS",
)


def check_sacrebleu_model(path: Path) -> None:
    """Reject a tokenizer model that differs from the pinned FLORES-200 artifact."""
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != SACREBLEU_MODEL_SHA256:
        raise ValueError(f"FLORES-200 tokenizer checksum mismatch: {path}")


def check_test_data(root: Path) -> None:
    """Require nonempty files, pinned IFBench source, and a verified tokenizer."""
    missing = [
        relative
        for relative in REQUIRED_FILES
        if not (root / relative).is_file() or not (root / relative).stat().st_size
    ]
    if missing:
        raise FileNotFoundError(f"Missing or empty Gym test data under {root}: {', '.join(missing)}")
    if (root / REQUIRED_FILES[0]).read_text().strip() != IFBENCH_COMMIT:
        raise ValueError(f"IFBench revision mismatch under {root}")
    check_sacrebleu_model(root / SACREBLEU_MODEL)


def prepare_nltk_cache(root: Path, cache_dir: Path) -> Path:
    """Copy supplied data into a private, process-owned tree accepted by NLTK."""
    source = (root / "nltk_data").resolve()
    cache_dir = cache_dir.resolve()
    if cache_dir.is_relative_to(source):
        raise ValueError("NLTK cache must be outside the supplied NLTK tree")
    cache_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir.stat().st_uid != os.geteuid():
        raise ValueError("NLTK cache must be owned by the current process user")
    cache_dir.chmod(0o700)
    private = Path(tempfile.mkdtemp(prefix="gym-nltk-", dir=cache_dir))
    target = private / "nltk_data"
    shutil.copytree(source, target, copy_function=shutil.copyfile)
    for path in (target, *target.rglob("*")):
        path.chmod(0o700 if path.is_dir() else 0o600)
    return target


def main() -> None:
    """Check the mounted bundle and optionally export GitHub test-data paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=os.environ.get("NEMO_GYM_TEST_DATA_ROOT"))
    parser.add_argument("--nltk-cache", type=Path, help="Prepare a private NLTK copy under this cache directory")
    args = parser.parse_args()
    if not args.root:
        parser.error("pass ROOT or set NEMO_GYM_TEST_DATA_ROOT")
    check_test_data(Path(args.root))
    print(f"Gym test-data inventory passed: {len(REQUIRED_FILES)} files under {args.root}")
    if args.nltk_cache:
        target = prepare_nltk_cache(Path(args.root), args.nltk_cache)
        with Path(os.environ["GITHUB_ENV"]).open("a") as output:
            output.write(f"NLTK_DATA={target}\n")
            output.write(f"SACREBLEU={Path(args.root).resolve() / 'sacrebleu'}\n")
        print(f"Prepared private NLTK cache: {target}")


if __name__ == "__main__":
    main()
