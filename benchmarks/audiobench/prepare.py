# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare AudioBench (63 audio-LLM test sets) for Gym.

AudioBench (https://github.com/AudioLLMs/AudioBench, revision ``85dcdb88c41e6ebb2bcb3ce19b15d061dfef9dba``)
evaluates audio language models on speech recognition, speech translation, spoken question answering and
instruction following, spoken math, paralinguistics (emotion, gender, accent) and audio-scene/music understanding.
Each test set lives in a Hugging Face dataset repository pinned below. By default all 63 are prepared, one
profile per test set; ``subsets`` and ``categories`` narrow the selection. The parquet files are read in place
through the Hugging Face file system (nothing is cached besides the written WAV files) and every test set is
checked against its pinned row count. Audio is decoded and written as WAV files to ``data/audio/<subset>/``.
"""

from __future__ import annotations

import io
import os
import random
from dataclasses import dataclass
from pathlib import Path

from benchmarks import audio_manifest


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "audiobench_benchmark.jsonl"
PROFILES = BENCHMARK_DIR / "profiles.json"
CODE_FILES = (Path(__file__), PROFILES, Path(audio_manifest.__file__))

AUDIOBENCH_REVISION = "85dcdb88c41e6ebb2bcb3ce19b15d061dfef9dba"
# The system prompt of the source rows; the profiles' prompt_override replaces it in the request.
SYSTEM_PROMPT = "You are a helpful assistant. /no_think"


@dataclass(frozen=True)
class Subset:
    repo: str  # Hugging Face dataset repository
    path: str  # directory of the parquet files inside the repository
    split: str  # file-name prefix of the split (``<split>-00000-of-N.parquet``)
    rows: int  # released row count


REVISIONS = {
    "AudioLLMs/MMAU-mini": "db5fcbf6ceafb0b9e9d0d733f909052e26175ed0",
    "AudioLLMs/aishell_1_zh_test": "e73dc7f096ecea5c8e5c722446c8d3b4014fdd18",
    "AudioLLMs/alpaca_audio_test": "7eb2ab279975033690d67bee66e45eb612a430ee",
    "AudioLLMs/audiocaps_qa_test": "6cf6853c52f55482126f226a8859eb71f2021e0e",
    "AudioLLMs/audiocaps_test": "fb42aac15212cbddd723fbbf04b6071b60a9f8fe",
    "AudioLLMs/clotho_aqa_test": "a7f4a7983d490e87a79eb041fede83e418c0ea4f",
    "AudioLLMs/cn_college_listen_mcq_test": "708d8ef278b5eadae714c314889f5d81ded18a25",
    "AudioLLMs/common_voice_15_en_test": "d7af6c5f2e2965f019b369f86a7a9f1be3ccc0a9",
    "AudioLLMs/covost2_en_id_test": "51a4b1517f07e1dcf9180aa270f264961165cab3",
    "AudioLLMs/covost2_en_ta_test": "23c23a71d698faf6c940e18753ade03acbcef86b",
    "AudioLLMs/covost2_en_zh_test": "6fe92d7fabaee215b078b022abbe8959280f3187",
    "AudioLLMs/covost2_id_en_test": "3998e50bec6eefdae5b38278238c359acb583f4c",
    "AudioLLMs/covost2_ta_en_test": "76a89636191da366b0d0d51f4c23dbe93257d5ba",
    "AudioLLMs/covost2_zh_en_test": "f9aec51472baf5ecad82ee9440e5a99e64fb0fef",
    "AudioLLMs/dream_tts_mcq_test": "a495d7c2a1facf4eb27af6dfc4c3878de8828cf8",
    "AudioLLMs/earnings21_test": "1edcc0d56dc6cdb6d21420f3155936e48485928d",
    "AudioLLMs/earnings22_test": "19da542f94ac74d603b9cb5eed0e13792517ccf4",
    "AudioLLMs/gigaspeech2-test": "77f58d8c13f1ede8dcef4b7214255c37f056a5bc",
    "AudioLLMs/gigaspeech_test": "77582c9e37dc7cc98ab2dd21702a94ddcf95f976",
    "AudioLLMs/iemocap_emotion_recognition": "72ca1bd6e70a9522e91c9ddfe4cf3e1d4407cd80",
    "AudioLLMs/iemocap_gender_recognition": "4e44e63a84eb1a33035b973193b50fccf3452738",
    "AudioLLMs/librispeech_test_clean": "80ac8624e83d495d4bb9313dbb311b57ea13b4da",
    "AudioLLMs/librispeech_test_other": "4417cb8c3515a71afeadf36df1d30eb42952d671",
    "AudioLLMs/meld_emotion_test": "005648394595a101e7c4ebeddd70043e7ba4a7a7",
    "AudioLLMs/meld_sentiment_test": "912e2aea9d2b2b8f79678fe4d5e89c98539548e7",
    "AudioLLMs/mu_chomusic_test": "47531859d1be6f6389c5dfd6a1f8c1354f1ba375",
    "AudioLLMs/openhermes_instruction_test": "574f252dd1847b08bbb4331c9d7100bd872f0cfe",
    "AudioLLMs/peoples_speech_test": "a9982cc64c0b586336472a101ca52d8f82dec4bb",
    "AudioLLMs/public_sg_speech_qa_test": "ec4bcaf2615eaba0ca30b86f1f53ce4acb620f85",
    "AudioLLMs/seame_dev_man": "4a510d3e42acdf71a08441219b70fab0df9d255d",
    "AudioLLMs/seame_dev_sge": "9b41e3e4b96ac23a05636b46a5792ed3ef54378b",
    "AudioLLMs/slue_p2_sqa5_test": "24809a6ef2e1243543d97a993c9b12765f0a0bff",
    "AudioLLMs/spoken_squad_test": "b55aab98726d0eab95eeef1ee9992a0532b3226e",
    "AudioLLMs/tedlium3_long_form_test": "097511c11b7b8550e17f938bd2bcb127de0ded13",
    "AudioLLMs/tedlium3_test": "aef8b18c0ee54a0123de5e671b2b889dbdd0b5c6",
    "AudioLLMs/voxceleb_accent_test": "d89c114a8337920f51879e1cf7b93a9c01697263",
    "AudioLLMs/voxceleb_gender_test": "0844e44484101caf53beaf8b243ec574de016359",
    "AudioLLMs/wavcaps_qa_test": "9be38b7c3c8b971ec42cee607a7670a3ec2d116a",
    "AudioLLMs/wavcaps_test": "3f87f655ebfb769579890d952504fd32fb853649",
    "MERaLiON/Multitask-National-Speech-Corpus-v1": "d169debbce4a2dc07f7293360d3eed33c06793b9",
    "amao0o0/spoken-mqa": "a1f9ec6143c6ce12966d7c6b3283905d679645c9",
}

# Test sets scored by an LLM judge (AudioBench's ``llama3_70b_judge`` / ``gpt4o_judge``), then the automatic ones.
JUDGE_SUBSETS = {
    "alpaca_audio_test": Subset("AudioLLMs/alpaca_audio_test", "data", "test", 100),
    "audiocaps_qa_test": Subset("AudioLLMs/audiocaps_qa_test", "data", "test", 313),
    "audiocaps_test": Subset("AudioLLMs/audiocaps_test", "data", "test", 4400),
    "clotho_aqa_test": Subset("AudioLLMs/clotho_aqa_test", "data", "test", 2057),
    "cn_college_listen_mcq_test": Subset("AudioLLMs/cn_college_listen_mcq_test", "data", "test", 2271),
    "dream_tts_mcq_test": Subset("AudioLLMs/dream_tts_mcq_test", "data", "test", 1913),
    "iemocap_emotion_test": Subset("AudioLLMs/iemocap_emotion_recognition", "data", "test", 1004),
    "iemocap_gender_test": Subset("AudioLLMs/iemocap_gender_recognition", "data", "test", 1004),
    "imda_ar_dialogue": Subset("MERaLiON/Multitask-National-Speech-Corpus-v1", "PQA-AR-Dialogue-Test", "train", 3000),
    "imda_ar_sentence": Subset("MERaLiON/Multitask-National-Speech-Corpus-v1", "PQA-AR-Sentence-Test", "train", 6000),
    "imda_gr_dialogue": Subset("MERaLiON/Multitask-National-Speech-Corpus-v1", "PQA-GR-Dialogue-Test", "train", 3000),
    "imda_gr_sentence": Subset("MERaLiON/Multitask-National-Speech-Corpus-v1", "PQA-GR-Sentence-Test", "train", 6000),
    "imda_part3_30s_ds_human_test": Subset(
        "MERaLiON/Multitask-National-Speech-Corpus-v1", "SDS-PART3-Test", "train", 100
    ),
    "imda_part4_30s_ds_human_test": Subset(
        "MERaLiON/Multitask-National-Speech-Corpus-v1", "SDS-PART4-Test", "train", 100
    ),
    "imda_part5_30s_ds_human_test": Subset(
        "MERaLiON/Multitask-National-Speech-Corpus-v1", "SDS-PART5-Test", "train", 100
    ),
    "imda_part6_30s_ds_human_test": Subset(
        "MERaLiON/Multitask-National-Speech-Corpus-v1", "SDS-PART6-Test", "train", 100
    ),
    "imda_part3_30s_sqa_human_test": Subset(
        "MERaLiON/Multitask-National-Speech-Corpus-v1", "SQA-PART3-Test", "train", 100
    ),
    "imda_part4_30s_sqa_human_test": Subset(
        "MERaLiON/Multitask-National-Speech-Corpus-v1", "SQA-PART4-Test", "train", 100
    ),
    "imda_part5_30s_sqa_human_test": Subset(
        "MERaLiON/Multitask-National-Speech-Corpus-v1", "SQA-PART5-Test", "train", 100
    ),
    "imda_part6_30s_sqa_human_test": Subset(
        "MERaLiON/Multitask-National-Speech-Corpus-v1", "SQA-PART6-Test", "train", 100
    ),
    "meld_emotion_test": Subset("AudioLLMs/meld_emotion_test", "data", "test", 2610),
    "meld_sentiment_test": Subset("AudioLLMs/meld_sentiment_test", "data", "test", 2610),
    "mmau_mini": Subset("AudioLLMs/MMAU-mini", "MMAU-mini", "train", 1000),
    "muchomusic_test": Subset("AudioLLMs/mu_chomusic_test", "data", "test", 1187),
    "openhermes_audio_test": Subset("AudioLLMs/openhermes_instruction_test", "data", "test", 100),
    "public_sg_speech_qa_test": Subset("AudioLLMs/public_sg_speech_qa_test", "data", "test", 688),
    "slue_p2_sqa5_test": Subset("AudioLLMs/slue_p2_sqa5_test", "data", "test", 408),
    "spoken_squad_test": Subset("AudioLLMs/spoken_squad_test", "data", "test", 5351),
    "voxceleb_accent_test": Subset("AudioLLMs/voxceleb_accent_test", "data", "test", 4874),
    "voxceleb_gender_test": Subset("AudioLLMs/voxceleb_gender_test", "data", "test", 4874),
    "wavcaps_qa_test": Subset("AudioLLMs/wavcaps_qa_test", "data", "test", 304),
    "wavcaps_test": Subset("AudioLLMs/wavcaps_test", "data", "test", 1730),
}
AUTOMATIC_SUBSETS = {
    "aishell_asr_zh_test": Subset("AudioLLMs/aishell_1_zh_test", "data", "test", 6920),
    "common_voice_15_en_test": Subset("AudioLLMs/common_voice_15_en_test", "data", "test", 16348),
    "covost2_en_id_test": Subset("AudioLLMs/covost2_en_id_test", "data", "test", 15531),
    "covost2_en_ta_test": Subset("AudioLLMs/covost2_en_ta_test", "data", "test", 15531),
    "covost2_en_zh_test": Subset("AudioLLMs/covost2_en_zh_test", "data", "test", 15531),
    "covost2_id_en_test": Subset("AudioLLMs/covost2_id_en_test", "data", "test", 844),
    "covost2_ta_en_test": Subset("AudioLLMs/covost2_ta_en_test", "data", "test", 754),
    "covost2_zh_en_test": Subset("AudioLLMs/covost2_zh_en_test", "data", "test", 4898),
    "earnings21_test": Subset("AudioLLMs/earnings21_test", "data", "test", 44),
    "earnings22_test": Subset("AudioLLMs/earnings22_test", "data", "test", 125),
    "gigaspeech_test": Subset("AudioLLMs/gigaspeech_test", "data", "test", 18650),
    "gigaspeech2_indo": Subset("AudioLLMs/gigaspeech2-test", "id-test", "train", 17273),
    "gigaspeech2_thai": Subset("AudioLLMs/gigaspeech2-test", "th-test", "train", 8823),
    "gigaspeech2_viet": Subset("AudioLLMs/gigaspeech2-test", "vi-test", "train", 4422),
    "imda_part1_asr_test": Subset("MERaLiON/Multitask-National-Speech-Corpus-v1", "ASR-PART1-Test", "train", 3000),
    "imda_part2_asr_test": Subset("MERaLiON/Multitask-National-Speech-Corpus-v1", "ASR-PART2-Test", "train", 3000),
    "imda_part3_30s_asr_test": Subset("MERaLiON/Multitask-National-Speech-Corpus-v1", "ASR-PART3-Test", "train", 1000),
    "imda_part4_30s_asr_test": Subset("MERaLiON/Multitask-National-Speech-Corpus-v1", "ASR-PART4-Test", "train", 1000),
    "imda_part5_30s_asr_test": Subset("MERaLiON/Multitask-National-Speech-Corpus-v1", "ASR-PART5-Test", "train", 1000),
    "imda_part6_30s_asr_test": Subset("MERaLiON/Multitask-National-Speech-Corpus-v1", "ASR-PART6-Test", "train", 1000),
    "librispeech_test_clean": Subset("AudioLLMs/librispeech_test_clean", "data", "test", 2617),
    "librispeech_test_other": Subset("AudioLLMs/librispeech_test_other", "data", "test", 2935),
    "peoples_speech_test": Subset("AudioLLMs/peoples_speech_test", "data", "test", 32603),
    "seame_dev_man": Subset("AudioLLMs/seame_dev_man", "seame_dev_man", "train", 2610),
    "seame_dev_sge": Subset("AudioLLMs/seame_dev_sge", "seame_dev_sge", "train", 3222),
    "spoken-mqa_long_digit": Subset("amao0o0/spoken-mqa", "data", "long_digit", 173),
    "spoken-mqa_multi_step_reasoning": Subset("amao0o0/spoken-mqa", "data", "multi_step_reasoning", 1402),
    "spoken-mqa_short_digit": Subset("amao0o0/spoken-mqa", "data", "short_digit", 100),
    "spoken-mqa_single_step_reasoning": Subset("amao0o0/spoken-mqa", "data", "single_step_reasoning", 594),
    "tedlium3_test": Subset("AudioLLMs/tedlium3_test", "data", "test", 1142),
    "tedlium3_long_form_test": Subset("AudioLLMs/tedlium3_long_form_test", "data", "test", 11),
}
SUBSETS = {**JUDGE_SUBSETS, **AUTOMATIC_SUBSETS}
CATEGORIES = ("judge", "nonjudge")

# The four Spoken-MQA test sets (spoken arithmetic; their answers are numbers scored by exact match).
SPOKEN_MQA = {
    "spoken-mqa_short_digit": "arithmetic",
    "spoken-mqa_long_digit": "arithmetic",
    "spoken-mqa_single_step_reasoning": "reasoning",
    "spoken-mqa_multi_step_reasoning": "reasoning",
}
# Long recordings scored by WER with AudioBench's Earnings normalization; the prompt is drawn with a fixed seed from
# a list of transcription instructions written for this benchmark (AudioBench draws from its own list, so the
# prompt wording differs from AudioBench's).
EARNINGS = {"earnings21_test": "arxiv:2104.11348", "earnings22_test": "arxiv:2203.15591"}
EARNINGS_INSTRUCTIONS = (
    "Write down what is said in this recording.",
    "Transcribe this recording into text.",
    "Listen to the audio and give me its transcript.",
    "Convert the speech in this clip into a written transcript.",
    "Provide a written transcription of everything that is spoken.",
    "Identify the words spoken in the audio and type them out.",
    "Produce a text transcript of this speech.",
    "Handle the audio and return the transcript as text.",
    "Transcribe the conversation you hear into text.",
)
EARNINGS_PROMPT_SEED = 42
# Speech translation (BLEU): target language of each CoVoST 2 direction.
TRANSLATION_TARGETS = {
    "covost2_en_id_test": "id",
    "covost2_en_ta_test": "ta",
    "covost2_en_zh_test": "zh",
    "covost2_id_en_test": "en",
    "covost2_ta_en_test": "en",
    "covost2_zh_en_test": "en",
}
# Test sets whose official metric is a corpus word error rate (the other automatic ones are Spoken-MQA accuracy
# and BLEU).
WER_SUBSETS = tuple(name for name in AUTOMATIC_SUBSETS if name not in SPOKEN_MQA and name not in TRANSLATION_TARGETS)
# Test sets whose instruction needs the answer options appended (AudioBench's ``Question:`` / ``Choices:`` layout).
CHOICES_SUBSETS = ("cn_college_listen_mcq_test", "dream_tts_mcq_test", "muchomusic_test", "mmau_mini")


def profile_name(subset: str) -> str:
    """``audiobench.spoken-mqa.<split>`` for Spoken-MQA, ``audiobench.<subset>`` otherwise."""
    return (
        "audiobench.spoken-mqa." + subset.removeprefix("spoken-mqa_")
        if subset in SPOKEN_MQA
        else f"audiobench.{subset}"
    )


def select_subsets(
    subsets: str | list[str] | tuple[str, ...] | None, categories: str | list[str] | tuple[str, ...] | None
) -> tuple[str, ...]:
    """Normalize ``subsets`` (names, with or without the ``_test`` suffix) and ``categories`` to subset names.

    Both accept a list or a comma-separated string; given together they intersect. The default is every subset.
    """

    selected = list(SUBSETS)
    if categories is not None:
        wanted = audio_manifest.select(categories, CATEGORIES, "AudioBench categories")
        pool = {"judge": JUDGE_SUBSETS, "nonjudge": AUTOMATIC_SUBSETS}
        selected = [name for name in selected if any(name in pool[category] for category in wanted)]
    if subsets is not None:
        names = [name.strip() for name in subsets.split(",")] if isinstance(subsets, str) else list(subsets)
        wanted = audio_manifest.select(
            [name if name in SUBSETS else f"{name}_test" for name in names if name], SUBSETS, "AudioBench subsets"
        )
        selected = [name for name in selected if name in wanted]
    if not selected:
        raise ValueError("The AudioBench subsets and categories selected do not intersect")
    return tuple(selected)


def parquet_files(subset: Subset) -> list[str]:
    """The pinned parquet files of the subset's split, in dataset order."""
    from huggingface_hub import HfFileSystem

    directory = f"datasets/{subset.repo}@{REVISIONS[subset.repo]}/{subset.path}"
    names = sorted(HfFileSystem().ls(directory, detail=False))
    return [name for name in names if name.endswith(".parquet") and Path(name).name.startswith(f"{subset.split}-")]


def iter_samples(subset: Subset, limit: int | None = None):
    """Yield the samples of the subset in order, reading only the row groups needed."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    count = 0
    for name in parquet_files(subset):
        with HfFileSystem().open(name, "rb") as handle:
            for batch in pq.ParquetFile(handle).iter_batches(batch_size=4):
                for sample in batch.to_pylist():
                    if limit is not None and count >= limit:
                        return
                    yield sample
                    count += 1
    if limit is None and count != subset.rows:
        raise ValueError(f"AudioBench {subset.repo}/{subset.path}: expected {subset.rows} rows, found {count}")


def decode_audio(audio: dict):
    """Decode the encoded audio of a sample to a mono float32 array and its sampling rate."""
    import numpy as np
    import soundfile

    if audio.get("bytes") is None:
        raise ValueError("AudioBench sample without embedded audio bytes")
    array, rate = soundfile.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
    if array.ndim == 2:
        array = array.mean(axis=1)
    return np.asarray(array), int(rate)


def earnings_instruction(index: int) -> str:
    """The seeded draw from the transcription instruction list for the ``index``-th sample."""
    rng = random.Random(EARNINGS_PROMPT_SEED)
    for _ in range(index + 1):
        instruction = rng.choice(EARNINGS_INSTRUCTIONS)
    return instruction


def _instruction(subset: str, sample: dict, index: int) -> str:
    instruction = earnings_instruction(index) if subset in EARNINGS else sample.get("instruction", "Process the audio")
    if isinstance(instruction, dict):
        instruction = instruction["text"]
    if not isinstance(instruction, str):
        raise ValueError(f"Instruction must be a string for {subset} sample {index}: {instruction!r}")
    choices = sample.get("choices")
    if subset in CHOICES_SUBSETS and choices:
        choices = " ".join(choices) if isinstance(choices, list) else choices
        separator = "\n Choices:\n" if subset != "mmau_mini" else "\nChoices:\n"
        instruction = f"Question:\n{instruction}{separator}{choices}"
    return instruction


def format_source_row(subset: str, sample: dict, index: int, audio_path: str, duration: float) -> dict:
    """The source row of one sample (``index`` is its 0-based position in the test set)."""
    spec = SUBSETS[subset]
    instruction = _instruction(subset, sample, index)
    audio = {"path": audio_path, "duration": duration}
    if subset in SPOKEN_MQA:
        answer = sample["answer"]
        if not isinstance(answer, dict) or not isinstance(answer.get("text"), list):
            raise ValueError(f"Spoken-MQA answer.text must be a list for {subset} sample {index}: {answer!r}")
        reference, task_type = [str(value) for value in answer["text"]], "MathQA"
    else:
        reference = sample.get("reference", sample.get("answer", ""))
        task_type = "ASR" if subset in EARNINGS or subset in WER_SUBSETS else "unknown"
        if subset in TRANSLATION_TARGETS:
            task_type = "Translation"
    row = {
        "expected_answer": reference,
        "audio_path": [audio_path],
        # The audio scorer parses a judge reply only for open-ended rows.
        "category": "open" if subset in JUDGE_SUBSETS else "closed",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction, "audio": audio, "audios": [audio]},
        ],
        "dataset": subset,
        "subset_for_metrics": subset,
        "sample_id": index,
        "task_type": task_type,
        "question": instruction,
    }
    if subset in TRANSLATION_TARGETS:
        row["extra_fields"] = {"tgt_lang": TRANSLATION_TARGETS[subset]}
    provenance = {
        "benchmark": "AudioBench",
        "benchmark_revision": AUDIOBENCH_REVISION,
        "dataset": spec.repo,
        "dataset_revision": REVISIONS[spec.repo],
        "split": spec.split,
    }
    if subset in SPOKEN_MQA:
        row["spoken_mqa_protocol"] = SPOKEN_MQA[subset]
        provenance["split"] = subset.removeprefix("spoken-mqa_")
    elif subset in EARNINGS:
        provenance |= {
            "split": "test",
            "paper": EARNINGS[subset],
            "metric": "wer",
            "normalization": "audiobench_earnings",
            "prompt_contract": "official_prompt_list_deterministic_seed",
            "prompt_seed": EARNINGS_PROMPT_SEED,
        }
    row["source_provenance"] = provenance
    for key in (
        "choices",
        "options",
        "audio_text_instruction",
        "audio_gt",
        "dimension",
        "rule_type",
        "rule_target",
        "task",
    ):
        if key in sample:
            row[key] = sample[key]
    return row


def write_audio(subset: str, sample: dict, index: int, audio_dir: Path) -> tuple[str, float] | None:
    """Write the sample's audio as ``<subset>_<index>.wav``; returns (file name, duration), None if it is empty."""
    import soundfile

    audio = sample.get("context") or sample.get("audio")
    array, rate = decode_audio(audio)
    if len(array) == 0:
        print(f"Warning: {subset} sample {index} has empty audio, skipped")
        return None
    filename = f"{subset}_{index:06d}.wav"
    audio_dir.mkdir(parents=True, exist_ok=True)
    staged = audio_dir / (filename + ".tmp")
    soundfile.write(staged, array, rate, format="WAV")
    os.replace(staged, audio_dir / filename)
    return filename, float(len(array) / rate)


is_prepared_data_current = audio_manifest.currency_check(CODE_FILES)


def prepare(
    *,
    subsets: str | list[str] | tuple[str, ...] | None = None,
    categories: str | list[str] | tuple[str, ...] | None = None,
    audio_root: str | None = None,
    max_samples_per_profile: int | None = None,
) -> Path:
    """Download AudioBench's test sets and write the benchmark JSONL.

    Args:
        subsets: Test-set names (a list or a comma-separated string; the ``_test`` suffix is optional). Default: all
            63.
        categories: ``judge`` (scored by an LLM judge) and/or ``nonjudge`` (automatic metrics); intersected with
            ``subsets`` when both are given.
        audio_root: Prefix of the audio paths in the rows (see ``audio_manifest.audio_prefix``).
        max_samples_per_profile: Keep only the first N samples of each test set (a profile is one test set), so a
            run over a few test sets stays small; only those N recordings are read and written.
    """
    if max_samples_per_profile is not None and (
        isinstance(max_samples_per_profile, bool)
        or not isinstance(max_samples_per_profile, int)
        or max_samples_per_profile < 1
    ):
        raise ValueError(f"max_samples_per_profile must be a positive integer, got {max_samples_per_profile!r}")
    selected = select_subsets(subsets, categories)
    prefix = f"{audio_manifest.audio_prefix(DATA_DIR, audio_root)}/audio"

    def sources(subset):
        for sample_index, sample in enumerate(iter_samples(SUBSETS[subset], max_samples_per_profile)):
            written = write_audio(subset, sample, sample_index, DATA_DIR / "audio" / subset)
            if written is not None:
                filename, duration = written
                yield format_source_row(subset, sample, sample_index, f"{prefix}/{subset}/{filename}", duration)

    prepared = audio_manifest.profile_rows(PROFILES, {profile_name(subset): sources(subset) for subset in selected})

    options = {
        "subsets": None if selected == tuple(SUBSETS) else list(selected),
        "audio_root": audio_root,
        "max_samples_per_profile": max_samples_per_profile,
    }
    return audio_manifest.write_rows(OUTPUT_FPATH, prepared, code_files=CODE_FILES, **options)
