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
    "imda_ar_dialogue":               {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "PQA-AR-Dialogue-Test", "bucket": "judge", "license_gated": True, "description": "IMDA Singlish accent recognition (dialogue level). License-gated."},  # pragma: allowlist secret
    "imda_ar_sentence":               {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "PQA-AR-Sentence-Test", "bucket": "judge", "license_gated": True, "description": "IMDA Singlish accent recognition (sentence level). License-gated."},  # pragma: allowlist secret
    "imda_gr_dialogue":               {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "PQA-GR-Dialogue-Test", "bucket": "judge", "license_gated": True, "description": "IMDA Singlish gender recognition (dialogue level). License-gated."},  # pragma: allowlist secret
    "imda_gr_sentence":               {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "PQA-GR-Sentence-Test", "bucket": "judge", "license_gated": True, "description": "IMDA Singlish gender recognition (sentence level). License-gated."},  # pragma: allowlist secret
    "imda_part3_30s_ds_human_test":   {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "SDS-PART3-Test", "bucket": "judge", "license_gated": True, "description": "IMDA NSC Part 3 dialogue summarization (30s). License-gated."},  # pragma: allowlist secret
    "imda_part4_30s_ds_human_test":   {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "SDS-PART4-Test", "bucket": "judge", "license_gated": True, "description": "IMDA NSC Part 4 dialogue summarization (30s). License-gated."},  # pragma: allowlist secret
    "imda_part5_30s_ds_human_test":   {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "SDS-PART5-Test", "bucket": "judge", "license_gated": True, "description": "IMDA NSC Part 5 dialogue summarization (30s). License-gated."},  # pragma: allowlist secret
    "imda_part6_30s_ds_human_test":   {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "SDS-PART6-Test", "bucket": "judge", "license_gated": True, "description": "IMDA NSC Part 6 dialogue summarization (30s). License-gated."},  # pragma: allowlist secret
    "imda_part3_30s_sqa_human_test":  {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "SQA-PART3-Test", "bucket": "judge", "license_gated": True, "description": "IMDA NSC Part 3 spoken QA (30s). License-gated."},  # pragma: allowlist secret
    "imda_part4_30s_sqa_human_test":  {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "SQA-PART4-Test", "bucket": "judge", "license_gated": True, "description": "IMDA NSC Part 4 spoken QA (30s). License-gated."},  # pragma: allowlist secret
    "imda_part5_30s_sqa_human_test":  {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "SQA-PART5-Test", "bucket": "judge", "license_gated": True, "description": "IMDA NSC Part 5 spoken QA (30s). License-gated."},  # pragma: allowlist secret
    "imda_part6_30s_sqa_human_test":  {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "SQA-PART6-Test", "bucket": "judge", "license_gated": True, "description": "IMDA NSC Part 6 spoken QA (30s). License-gated."},  # pragma: allowlist secret
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
    "imda_part1_asr_test":            {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "ASR-PART1-Test", "bucket": "asr", "license_gated": True, "instruction": "Transcribe the audio.", "description": "IMDA NSC Part 1 ASR. License-gated."},  # pragma: allowlist secret
    "imda_part2_asr_test":            {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "ASR-PART2-Test", "bucket": "asr", "license_gated": True, "instruction": "Transcribe the audio.", "description": "IMDA NSC Part 2 ASR. License-gated."},  # pragma: allowlist secret
    "imda_part3_30s_asr_test":        {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "ASR-PART3-Test", "bucket": "asr", "license_gated": True, "instruction": "Transcribe the audio.", "description": "IMDA NSC Part 3 ASR (30s). License-gated."},  # pragma: allowlist secret
    "imda_part4_30s_asr_test":        {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "ASR-PART4-Test", "bucket": "asr", "license_gated": True, "instruction": "Transcribe the audio.", "description": "IMDA NSC Part 4 ASR (30s). License-gated."},  # pragma: allowlist secret
    "imda_part5_30s_asr_test":        {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "ASR-PART5-Test", "bucket": "asr", "license_gated": True, "instruction": "Transcribe the audio.", "description": "IMDA NSC Part 5 ASR (30s). License-gated."},  # pragma: allowlist secret
    "imda_part6_30s_asr_test":        {"hf_repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",   "hf_split": "train", "hf_data_dir": "ASR-PART6-Test", "bucket": "asr", "license_gated": True, "instruction": "Transcribe the audio.", "description": "IMDA NSC Part 6 ASR (30s). License-gated."},  # pragma: allowlist secret
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

# Pinned HF commit for every source repo, so a prepare run is reproducible.
HF_REVISIONS: dict[str, str] = {
    "AudioLLMs/MMAU-mini":                          "db5fcbf6ceafb0b9e9d0d733f909052e26175ed0",  # pragma: allowlist secret
    "AudioLLMs/aishell_1_zh_test":                  "e73dc7f096ecea5c8e5c722446c8d3b4014fdd18",  # pragma: allowlist secret
    "AudioLLMs/alpaca_audio_test":                  "7eb2ab279975033690d67bee66e45eb612a430ee",  # pragma: allowlist secret
    "AudioLLMs/audiocaps_qa_test":                  "6cf6853c52f55482126f226a8859eb71f2021e0e",  # pragma: allowlist secret
    "AudioLLMs/audiocaps_test":                     "fb42aac15212cbddd723fbbf04b6071b60a9f8fe",  # pragma: allowlist secret
    "AudioLLMs/clotho_aqa_test":                    "a7f4a7983d490e87a79eb041fede83e418c0ea4f",  # pragma: allowlist secret
    "AudioLLMs/cn_college_listen_mcq_test":         "708d8ef278b5eadae714c314889f5d81ded18a25",  # pragma: allowlist secret
    "AudioLLMs/common_voice_15_en_test":            "d7af6c5f2e2965f019b369f86a7a9f1be3ccc0a9",  # pragma: allowlist secret
    "AudioLLMs/covost2_en_id_test":                 "51a4b1517f07e1dcf9180aa270f264961165cab3",  # pragma: allowlist secret
    "AudioLLMs/covost2_en_ta_test":                 "23c23a71d698faf6c940e18753ade03acbcef86b",  # pragma: allowlist secret
    "AudioLLMs/covost2_en_zh_test":                 "6fe92d7fabaee215b078b022abbe8959280f3187",  # pragma: allowlist secret
    "AudioLLMs/covost2_id_en_test":                 "3998e50bec6eefdae5b38278238c359acb583f4c",  # pragma: allowlist secret
    "AudioLLMs/covost2_ta_en_test":                 "76a89636191da366b0d0d51f4c23dbe93257d5ba",  # pragma: allowlist secret
    "AudioLLMs/covost2_zh_en_test":                 "f9aec51472baf5ecad82ee9440e5a99e64fb0fef",  # pragma: allowlist secret
    "AudioLLMs/dream_tts_mcq_test":                 "a495d7c2a1facf4eb27af6dfc4c3878de8828cf8",  # pragma: allowlist secret
    "AudioLLMs/earnings21_test":                    "1edcc0d56dc6cdb6d21420f3155936e48485928d",  # pragma: allowlist secret
    "AudioLLMs/earnings22_test":                    "19da542f94ac74d603b9cb5eed0e13792517ccf4",  # pragma: allowlist secret
    "AudioLLMs/gigaspeech2-test":                   "77f58d8c13f1ede8dcef4b7214255c37f056a5bc",  # pragma: allowlist secret
    "AudioLLMs/gigaspeech_test":                    "77582c9e37dc7cc98ab2dd21702a94ddcf95f976",  # pragma: allowlist secret
    "AudioLLMs/iemocap_emotion_recognition":        "72ca1bd6e70a9522e91c9ddfe4cf3e1d4407cd80",  # pragma: allowlist secret
    "AudioLLMs/iemocap_gender_recognition":         "4e44e63a84eb1a33035b973193b50fccf3452738",  # pragma: allowlist secret
    "AudioLLMs/librispeech_test_clean":             "80ac8624e83d495d4bb9313dbb311b57ea13b4da",  # pragma: allowlist secret
    "AudioLLMs/librispeech_test_other":             "4417cb8c3515a71afeadf36df1d30eb42952d671",  # pragma: allowlist secret
    "AudioLLMs/meld_emotion_test":                  "005648394595a101e7c4ebeddd70043e7ba4a7a7",  # pragma: allowlist secret
    "AudioLLMs/meld_sentiment_test":                "912e2aea9d2b2b8f79678fe4d5e89c98539548e7",  # pragma: allowlist secret
    "AudioLLMs/mu_chomusic_test":                   "47531859d1be6f6389c5dfd6a1f8c1354f1ba375",  # pragma: allowlist secret
    "AudioLLMs/openhermes_instruction_test":        "574f252dd1847b08bbb4331c9d7100bd872f0cfe",  # pragma: allowlist secret
    "AudioLLMs/peoples_speech_test":                "a9982cc64c0b586336472a101ca52d8f82dec4bb",  # pragma: allowlist secret
    "AudioLLMs/public_sg_speech_qa_test":           "ec4bcaf2615eaba0ca30b86f1f53ce4acb620f85",  # pragma: allowlist secret
    "AudioLLMs/seame_dev_man":                      "4a510d3e42acdf71a08441219b70fab0df9d255d",  # pragma: allowlist secret
    "AudioLLMs/seame_dev_sge":                      "9b41e3e4b96ac23a05636b46a5792ed3ef54378b",  # pragma: allowlist secret
    "AudioLLMs/slue_p2_sqa5_test":                  "24809a6ef2e1243543d97a993c9b12765f0a0bff",  # pragma: allowlist secret
    "AudioLLMs/spoken_squad_test":                  "b55aab98726d0eab95eeef1ee9992a0532b3226e",  # pragma: allowlist secret
    "AudioLLMs/tedlium3_long_form_test":            "097511c11b7b8550e17f938bd2bcb127de0ded13",  # pragma: allowlist secret
    "AudioLLMs/tedlium3_test":                      "aef8b18c0ee54a0123de5e671b2b889dbdd0b5c6",  # pragma: allowlist secret
    "AudioLLMs/voxceleb_accent_test":               "d89c114a8337920f51879e1cf7b93a9c01697263",  # pragma: allowlist secret
    "AudioLLMs/voxceleb_gender_test":               "0844e44484101caf53beaf8b243ec574de016359",  # pragma: allowlist secret
    "AudioLLMs/wavcaps_qa_test":                    "9be38b7c3c8b971ec42cee607a7670a3ec2d116a",  # pragma: allowlist secret
    "AudioLLMs/wavcaps_test":                       "3f87f655ebfb769579890d952504fd32fb853649",  # pragma: allowlist secret
    "MERaLiON/Multitask-National-Speech-Corpus-v1": "d169debbce4a2dc07f7293360d3eed33c06793b9",  # pragma: allowlist secret
    "amao0o0/spoken-mqa":                           "a1f9ec6143c6ce12966d7c6b3283905d679645c9",  # pragma: allowlist secret
}
# fmt: on
