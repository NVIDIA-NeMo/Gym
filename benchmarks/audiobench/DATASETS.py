# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-of-truth registry for AudioBench sub-datasets.

One row per upstream Skills sub-dataset. Mirrors
``nemo_skills/dataset/audiobench/prepare.py``'s JUDGE_DATASETS +
NONJUDGE_DATASETS lists plus their per-dataset HF metadata.

`bucket` selects the verifier:
  * "judge"        — open-ended LLM judge (audiobench_judge resource server)
  * "asr"          — Whisper + audiobench-style WER (asr_with_pc, task_type=ASR,
                     normalization_mode=audiobench)
  * "bleu"         — sacrebleu (asr_with_pc, task_type=BLEU)
  * "exact_match"  — case-insensitive punct-stripped match (asr_with_pc,
                     task_type=EXACT_MATCH)

`instruction` is the per-dataset text wrapper used as the model's user message
(`prompts/default.yaml` is `user: {instruction}`). For most datasets the
upstream HF row has its own ``instruction`` field that we reuse directly; the
override here is used when:
  * the upstream row's instruction is empty / generic and a more pointed
    English prompt helps the model (ASR, MQA);
  * the bucket needs language-specific cues (multilingual ASR, translation
    direction).

`license_gated=True` flags datasets hosted on a HuggingFace repo that
requires explicit license acceptance — `huggingface-cli login` plus a click
on the dataset page; otherwise `load_dataset` raises `GatedRepoError`.
"""

from __future__ import annotations

from typing import Optional, TypedDict


class DatasetSpec(TypedDict, total=False):
    hf_repo: str
    hf_split: str
    hf_data_dir: Optional[str]
    bucket: str  # judge | asr | bleu | exact_match
    license_gated: bool
    instruction: Optional[str]
    description: str


# fmt: off
DATASETS: dict[str, DatasetSpec] = {
    # ──── audiobench.judge (32 datasets) ────────────────────────────────────
    "alpaca_audio_test":              {"hf_repo": "AudioLLMs/alpaca_audio_test",                    "hf_split": "test",  "bucket": "judge", "description": "Alpaca instruction-following converted to spoken form."},
    "audiocaps_qa_test":              {"hf_repo": "AudioLLMs/audiocaps_qa_test",                    "hf_split": "test",  "bucket": "judge", "description": "AudioCaps audio question answering."},
    "audiocaps_test":                 {"hf_repo": "AudioLLMs/audiocaps_test",                       "hf_split": "test",  "bucket": "judge", "description": "AudioCaps captioning with judge-graded captions."},
    "clotho_aqa_test":                {"hf_repo": "AudioLLMs/clotho_aqa_test",                      "hf_split": "test",  "bucket": "judge", "description": "Clotho-AQA: audio QA on Clotho clips."},
    "cn_college_listen_mcq_test":     {"hf_repo": "AudioLLMs/cn_college_listen_mcq_test",           "hf_split": "test",  "bucket": "judge", "description": "Chinese college-listening MCQ — Mandarin listening comprehension."},
    "dream_tts_mcq_test":             {"hf_repo": "AudioLLMs/dream_tts_mcq_test",                   "hf_split": "test",  "bucket": "judge", "description": "DREAM dialogue reading comprehension via TTS-rendered prompts."},
    "iemocap_emotion_test":           {"hf_repo": "AudioLLMs/iemocap_emotion_recognition",          "hf_split": "test",  "bucket": "judge", "description": "IEMOCAP emotion recognition phrased as open-ended QA."},
    "iemocap_gender_test":            {"hf_repo": "AudioLLMs/iemocap_gender_recognition",           "hf_split": "test",  "bucket": "judge", "description": "IEMOCAP gender recognition phrased as open-ended QA."},
    "imda_ar_dialogue":               {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "AR-DIALOGUE", "bucket": "judge", "license_gated": True, "description": "IMDA Singlish accent recognition (dialogue level). License-gated."},
    "imda_ar_sentence":               {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "AR-SENTENCE", "bucket": "judge", "license_gated": True, "description": "IMDA Singlish accent recognition (sentence level). License-gated."},
    "imda_gr_dialogue":               {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "GR-DIALOGUE", "bucket": "judge", "license_gated": True, "description": "IMDA Singlish gender recognition (dialogue level). License-gated."},
    "imda_gr_sentence":               {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "GR-SENTENCE", "bucket": "judge", "license_gated": True, "description": "IMDA Singlish gender recognition (sentence level). License-gated."},
    "imda_part3_30s_ds_human_test":   {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "SDS-PART3-Test", "bucket": "judge", "license_gated": True, "description": "IMDA NSC Part 3 dialogue summarization (30s). License-gated."},
    "imda_part4_30s_ds_human_test":   {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "SDS-PART4-Test", "bucket": "judge", "license_gated": True, "description": "IMDA NSC Part 4 dialogue summarization (30s). License-gated."},
    "imda_part5_30s_ds_human_test":   {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "SDS-PART5-Test", "bucket": "judge", "license_gated": True, "description": "IMDA NSC Part 5 dialogue summarization (30s). License-gated."},
    "imda_part6_30s_ds_human_test":   {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "SDS-PART6-Test", "bucket": "judge", "license_gated": True, "description": "IMDA NSC Part 6 dialogue summarization (30s). License-gated."},
    "imda_part3_30s_sqa_human_test":  {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "SQA-PART3-Test", "bucket": "judge", "license_gated": True, "description": "IMDA NSC Part 3 spoken QA (30s). License-gated."},
    "imda_part4_30s_sqa_human_test":  {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "SQA-PART4-Test", "bucket": "judge", "license_gated": True, "description": "IMDA NSC Part 4 spoken QA (30s). License-gated."},
    "imda_part5_30s_sqa_human_test":  {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "SQA-PART5-Test", "bucket": "judge", "license_gated": True, "description": "IMDA NSC Part 5 spoken QA (30s). License-gated."},
    "imda_part6_30s_sqa_human_test":  {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "SQA-PART6-Test", "bucket": "judge", "license_gated": True, "description": "IMDA NSC Part 6 spoken QA (30s). License-gated."},
    "meld_emotion_test":              {"hf_repo": "AudioLLMs/meld_emotion_test",                   "hf_split": "test",  "bucket": "judge", "description": "MELD emotion recognition on multi-speaker dialogue."},
    "meld_sentiment_test":            {"hf_repo": "AudioLLMs/meld_sentiment_test",                 "hf_split": "test",  "bucket": "judge", "description": "MELD sentiment classification on multi-speaker dialogue."},
    "mmau_mini":                      {"hf_repo": "AudioLLMs/MMAU-mini",                            "hf_split": "test",  "bucket": "judge", "description": "MMAU-mini: massive multitask audio understanding (sound, music, speech)."},
    "muchomusic_test":                {"hf_repo": "AudioLLMs/mu_chomusic_test",                     "hf_split": "test",  "bucket": "judge", "description": "MuChoMusic: music understanding multiple-choice."},
    "openhermes_audio_test":          {"hf_repo": "AudioLLMs/openhermes_instruction_test",          "hf_split": "test",  "bucket": "judge", "description": "OpenHermes-derived speech instruction following."},
    "public_sg_speech_qa_test":       {"hf_repo": "AudioLLMs/public_sg_speech_qa_test",             "hf_split": "test",  "bucket": "judge", "description": "Public Singapore speech QA — spoken QA on Singapore-context clips."},
    "slue_p2_sqa5_test":              {"hf_repo": "AudioLLMs/slue_p2_sqa5_test",                    "hf_split": "test",  "bucket": "judge", "description": "SLUE Phase-2 SQA5 spoken QA."},
    "spoken_squad_test":              {"hf_repo": "AudioLLMs/spoken_squad_test",                    "hf_split": "test",  "bucket": "judge", "description": "Spoken-SQuAD: SQuAD reading comprehension over spoken passages."},
    "voxceleb_accent_test":           {"hf_repo": "AudioLLMs/voxceleb_accent_test",                 "hf_split": "test",  "bucket": "judge", "description": "VoxCeleb accent recognition phrased as open-ended QA."},
    "voxceleb_gender_test":           {"hf_repo": "AudioLLMs/voxceleb_gender_test",                 "hf_split": "test",  "bucket": "judge", "description": "VoxCeleb gender recognition phrased as open-ended QA."},
    "wavcaps_qa_test":                {"hf_repo": "AudioLLMs/wavcaps_qa_test",                      "hf_split": "test",  "bucket": "judge", "description": "WavCaps audio QA."},
    "wavcaps_test":                   {"hf_repo": "AudioLLMs/wavcaps_test",                         "hf_split": "test",  "bucket": "judge", "description": "WavCaps audio captioning with judge-graded captions."},

    # ──── audiobench.nonjudge / ASR (21 datasets) ───────────────────────────
    "common_voice_15_en_test":        {"hf_repo": "AudioLLMs/common_voice_15_en_test",              "hf_split": "test",  "bucket": "asr",  "instruction": "Transcribe the audio.", "description": "Common Voice 15 English ASR."},
    "earnings21_test":                {"hf_repo": "AudioLLMs/earnings21_test",                      "hf_split": "test",  "bucket": "asr",  "instruction": "Transcribe the audio.", "description": "Earnings-21 long-form English ASR."},
    "earnings22_test":                {"hf_repo": "AudioLLMs/earnings22_test",                      "hf_split": "test",  "bucket": "asr",  "instruction": "Transcribe the audio.", "description": "Earnings-22 long-form English ASR."},
    "gigaspeech_test":                {"hf_repo": "AudioLLMs/gigaspeech_test",                      "hf_split": "test",  "bucket": "asr",  "instruction": "Transcribe the audio.", "description": "GigaSpeech English ASR."},
    "librispeech_test_clean":         {"hf_repo": "AudioLLMs/librispeech_test_clean",               "hf_split": "test",  "bucket": "asr",  "instruction": "Transcribe the audio.", "description": "LibriSpeech test-clean — English ASR."},
    "librispeech_test_other":         {"hf_repo": "AudioLLMs/librispeech_test_other",               "hf_split": "test",  "bucket": "asr",  "instruction": "Transcribe the audio.", "description": "LibriSpeech test-other — harder English ASR variant."},
    "peoples_speech_test":            {"hf_repo": "AudioLLMs/peoples_speech_test",                  "hf_split": "test",  "bucket": "asr",  "instruction": "Transcribe the audio.", "description": "People's Speech English ASR."},
    "tedlium3_test":                  {"hf_repo": "AudioLLMs/tedlium3_test",                        "hf_split": "test",  "bucket": "asr",  "instruction": "Transcribe the audio.", "description": "TED-LIUM 3 — TED talks ASR."},
    "tedlium3_long_form_test":        {"hf_repo": "AudioLLMs/tedlium3_long_form_test",              "hf_split": "test",  "bucket": "asr",  "instruction": "Transcribe the audio.", "description": "TED-LIUM 3 long-form: full-talk ASR."},
    "aishell_asr_zh_test":            {"hf_repo": "AudioLLMs/aishell_1_zh_test",                    "hf_split": "test",  "bucket": "asr",  "instruction": "Transcribe the audio in Mandarin Chinese.", "description": "AISHELL-1 Mandarin ASR."},
    "gigaspeech2_thai":               {"hf_repo": "AudioLLMs/gigaspeech2-test",                     "hf_split": "train", "hf_data_dir": "th-test", "bucket": "asr",  "instruction": "Transcribe the audio in Thai.", "description": "GigaSpeech 2 Thai ASR."},
    "gigaspeech2_indo":               {"hf_repo": "AudioLLMs/gigaspeech2-test",                     "hf_split": "train", "hf_data_dir": "id-test", "bucket": "asr",  "instruction": "Transcribe the audio in Indonesian.", "description": "GigaSpeech 2 Indonesian ASR."},
    "gigaspeech2_viet":               {"hf_repo": "AudioLLMs/gigaspeech2-test",                     "hf_split": "train", "hf_data_dir": "vi-test", "bucket": "asr",  "instruction": "Transcribe the audio in Vietnamese.", "description": "GigaSpeech 2 Vietnamese ASR."},
    "imda_part1_asr_test":            {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "ASR-PART1-Test", "bucket": "asr", "license_gated": True, "instruction": "Transcribe the audio.", "description": "IMDA NSC Part 1 ASR. License-gated."},
    "imda_part2_asr_test":            {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "ASR-PART2-Test", "bucket": "asr", "license_gated": True, "instruction": "Transcribe the audio.", "description": "IMDA NSC Part 2 ASR. License-gated."},
    "imda_part3_30s_asr_test":        {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "ASR-PART3-Test", "bucket": "asr", "license_gated": True, "instruction": "Transcribe the audio.", "description": "IMDA NSC Part 3 ASR (30s). License-gated."},
    "imda_part4_30s_asr_test":        {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "ASR-PART4-Test", "bucket": "asr", "license_gated": True, "instruction": "Transcribe the audio.", "description": "IMDA NSC Part 4 ASR (30s). License-gated."},
    "imda_part5_30s_asr_test":        {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "ASR-PART5-Test", "bucket": "asr", "license_gated": True, "instruction": "Transcribe the audio.", "description": "IMDA NSC Part 5 ASR (30s). License-gated."},
    "imda_part6_30s_asr_test":        {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "ASR-PART6-Test", "bucket": "asr", "license_gated": True, "instruction": "Transcribe the audio.", "description": "IMDA NSC Part 6 ASR (30s). License-gated."},
    "seame_dev_man":                  {"hf_repo": "AudioLLMs/seame_dev_man",                        "hf_split": "test",  "bucket": "asr",  "instruction": "Transcribe the audio.", "description": "SEAME English-Mandarin code-switching dev (Mandarin-dominant)."},
    "seame_dev_sge":                  {"hf_repo": "AudioLLMs/seame_dev_sge",                        "hf_split": "test",  "bucket": "asr",  "instruction": "Transcribe the audio.", "description": "SEAME English-Mandarin code-switching dev (Singapore-English-dominant)."},

    # ──── audiobench.nonjudge / BLEU translation (6 datasets) ────────────────
    "covost2_en_id_test":             {"hf_repo": "AudioLLMs/covost2_en_id_test",                   "hf_split": "test",  "bucket": "bleu", "instruction": "Translate the spoken English to Indonesian.", "description": "CoVoST 2 English→Indonesian speech translation."},
    "covost2_en_ta_test":             {"hf_repo": "AudioLLMs/covost2_en_ta_test",                   "hf_split": "test",  "bucket": "bleu", "instruction": "Translate the spoken English to Tamil.",      "description": "CoVoST 2 English→Tamil speech translation."},
    "covost2_en_zh_test":             {"hf_repo": "AudioLLMs/covost2_en_zh_test",                   "hf_split": "test",  "bucket": "bleu", "instruction": "Translate the spoken English to Chinese.",    "description": "CoVoST 2 English→Chinese speech translation."},
    "covost2_id_en_test":             {"hf_repo": "AudioLLMs/covost2_id_en_test",                   "hf_split": "test",  "bucket": "bleu", "instruction": "Translate the spoken Indonesian to English.", "description": "CoVoST 2 Indonesian→English speech translation."},
    "covost2_ta_en_test":             {"hf_repo": "AudioLLMs/covost2_ta_en_test",                   "hf_split": "test",  "bucket": "bleu", "instruction": "Translate the spoken Tamil to English.",      "description": "CoVoST 2 Tamil→English speech translation."},
    "covost2_zh_en_test":             {"hf_repo": "AudioLLMs/covost2_zh_en_test",                   "hf_split": "test",  "bucket": "bleu", "instruction": "Translate the spoken Chinese to English.",    "description": "CoVoST 2 Chinese→English speech translation."},

    # ──── audiobench.nonjudge / EXACT_MATCH spoken-MQA (4 datasets) ─────────
    "spoken_mqa_short_digit":         {"hf_repo": "amao0o0/spoken-mqa", "hf_split": "short_digit",            "bucket": "exact_match", "instruction": "Listen to the question and answer with the numeric result only.", "description": "Spoken-MQA short-digit arithmetic reasoning."},
    "spoken_mqa_long_digit":          {"hf_repo": "amao0o0/spoken-mqa", "hf_split": "long_digit",             "bucket": "exact_match", "instruction": "Listen to the question and answer with the numeric result only.", "description": "Spoken-MQA long-digit arithmetic reasoning."},
    "spoken_mqa_single_step_reasoning": {"hf_repo": "amao0o0/spoken-mqa", "hf_split": "single_step_reasoning", "bucket": "exact_match", "instruction": "Listen to the question and answer with the numeric result only.", "description": "Spoken-MQA single-step reasoning."},
    "spoken_mqa_multi_step_reasoning":  {"hf_repo": "amao0o0/spoken-mqa", "hf_split": "multi_step_reasoning",  "bucket": "exact_match", "instruction": "Listen to the question and answer with the numeric result only.", "description": "Spoken-MQA multi-step reasoning."},
}
# fmt: on


JUDGE_SLUGS = [s for s, c in DATASETS.items() if c["bucket"] == "judge"]
ASR_SLUGS = [s for s, c in DATASETS.items() if c["bucket"] == "asr"]
BLEU_SLUGS = [s for s, c in DATASETS.items() if c["bucket"] == "bleu"]
EXACT_MATCH_SLUGS = [s for s, c in DATASETS.items() if c["bucket"] == "exact_match"]

# Sanity: the registry should match Skills' upstream JUDGE_DATASETS +
# NONJUDGE_DATASETS (32 + 31 = 63 sub-datasets).
assert len(JUDGE_SLUGS) == 32, len(JUDGE_SLUGS)
assert len(ASR_SLUGS) == 21, len(ASR_SLUGS)
assert len(BLEU_SLUGS) == 6, len(BLEU_SLUGS)
assert len(EXACT_MATCH_SLUGS) == 4, len(EXACT_MATCH_SLUGS)
assert len(DATASETS) == 63, len(DATASETS)
