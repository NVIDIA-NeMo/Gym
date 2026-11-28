# AudioBench

[AudioBench](https://github.com/AudioLLMs/AudioBench) (revision `85dcdb88c41e6ebb2bcb3ce19b15d061dfef9dba`) evaluates
audio language models. By default all 63 test sets (240,701 samples) are prepared, one profile per test set:

| Group | Test sets | Samples | Score |
| --- | --- | --- | --- |
| Speech recognition | LibriSpeech clean and other, Common Voice 15 (en), People's Speech, GigaSpeech, TED-LIUM 3 (and long form), Earnings-21 and -22, AISHELL-1 (zh), GigaSpeech 2 (id, th, vi), SEAME (man, sge), IMDA parts 1-6 (21 sets) | 127,745 | corpus WER |
| Speech translation | CoVoST 2 en-id, en-ta, en-zh, id-en, ta-en, zh-en (6 sets) | 53,089 | corpus BLEU |
| Spoken math | Spoken-MQA short digit, long digit, single-step and multi-step reasoning (4 sets) | 2,269 | answer accuracy |
| Judged | spoken QA and summarisation (cn_college_listen, dream_tts, slue_p2_sqa5, public_sg_speech_qa, spoken_squad, IMDA SQA and SDS parts 3-6), instruction following (alpaca_audio, openhermes_audio), audio scene and music (audiocaps, wavcaps, clotho_aqa and their QA sets, muchomusic, mmau_mini), emotion, gender and accent (iemocap, meld, voxceleb, IMDA AR and GR) (32 sets) | 57,598 | judge score |

The full set is large (see Data), so a run usually narrows it. `subsets` (names, with or without `_test`) and
`categories` (`judge`, `nonjudge`) select test sets and intersect when both are given, and
`max_samples_per_profile` keeps only the first N samples of each selected test set (a profile is one test set; only
those recordings are downloaded and written):

```bash
# a small run across task types: ASR, speech translation, spoken math and one judged set, 20 samples each
gym eval prepare --benchmark audiobench \
    '+prepare_script_args.subsets=[librispeech_test_clean,covost2_zh_en_test,spoken-mqa_short_digit,spoken_squad_test]' \
    +prepare_script_args.max_samples_per_profile=20

# only the four Spoken-MQA sets
gym eval prepare --benchmark audiobench \
    '+prepare_script_args.subsets=[spoken-mqa_short_digit,spoken-mqa_long_digit,spoken-mqa_single_step_reasoning,spoken-mqa_multi_step_reasoning]'
```

A bare comma-separated value (`subsets=a,b`) is rejected by Hydra as an ambiguous sweep; pass a bracketed list. The cap
takes the first N samples in dataset order, so a capped run is a smoke run, not a representative sample. Narrowed or
capped data is never reported as already prepared.

## Data

Each test set is a pinned revision of a Hugging Face dataset (`AudioLLMs/<name>`, `amao0o0/spoken-mqa`,
`MERaLiON/Multitask-National-Speech-Corpus-v1` for the IMDA sets); repositories, revisions and row counts are in
`SUBSETS` and `REVISIONS` in `prepare.py`. Parquet files are read in place and checked against the released row counts
(not when a cap is set). The sources total about 75 GB (Earnings-21/22 about 18 GB, IMDA 12 GB, CoVoST 2 9 GB); the
decoded WAV files are larger, so check free disk space before a full run. The model server must accept the long
recordings. The benchmark ships no audio; it downloads the pinned revisions when you prepare it.

## License

**Mixed; read before use.** AudioBench's source code is under a Creative Commons NonCommercial license (none of it is
copied here, see Provenance) and its README says that "the usage of respective datasets is under their respective
licenses". None of the 41 Hugging Face dataset repositories sets a license in its metadata; only the Spoken-SQuAD card
states one in its text (CC-BY-SA-4.0). The table lists the terms of each test set's source corpus, taken from the
source's own page or repository. Where a Hugging Face card contradicts its source, the source is listed and the card is
noted: GigaSpeech and GigaSpeech 2 (the source's terms of access limit use to non-commercial research and education,
while the card metadata says Apache-2.0), WavCaps (card metadata says CC BY 4.0, its text says academic use only),
AudioCaps (the repository carries an MIT LICENSE, its README says academic use only) and MuChoMusic (CC BY-SA 4.0 in
the GitHub README, CC BY 4.0 on Zenodo and in the paper).

**12 test sets are non-commercial only, 4 are access-restricted at the source (AudioBench mirrors IEMOCAP and SEAME on
ungated repositories, so obtaining the source license is your responsibility), and 11 have no clear license** (29 are
attribution-required and 7 permissive). Do not use results or prepared data commercially, or redistribute the data,
without checking each source; narrow `subsets` to the test sets you may use, for example
`[librispeech_test_clean,librispeech_test_other,common_voice_15_en_test,covost2_en_zh_test,aishell_asr_zh_test,spoken_squad_test]`.
The manifest records the union of these terms as an SPDX expression; `LicenseRef-SODL-1.0` is the Singapore Open Data
License v1.0 (IMDA) and `LicenseRef-AudioBench-Dataset-Terms` stands for the access terms, academic-only statements
and unlicensed sets in the table. The classes are a conservative reading of what the sources state, not legal advice.

| Test set | Source corpus | Terms stated by the source | Class |
| --- | --- | --- | --- |
| `alpaca_audio_test` | Stanford Alpaca text, speech synthesised with XTTS-v2 | text CC BY-NC 4.0; audio synthesised with XTTS-v2 (Coqui Public Model License) | **non-commercial** |
| `audiocaps_qa_test` | AudioCaps (YouTube audio) | repository LICENSE is MIT, its README says academic use only | **non-commercial** |
| `audiocaps_test` | AudioCaps (YouTube audio) | repository LICENSE is MIT, its README says academic use only | **non-commercial** |
| `clotho_aqa_test` | Clotho-AQA | questions MIT; audio under per-clip Freesound licenses, mostly CC BY | attribution required |
| `cn_college_listen_mcq_test` | Chinese college-entrance listening exams (AudioBench compilation, WavLLM) | none found | no clear license |
| `dream_tts_mcq_test` | DREAM text, speech synthesised with XTTS-v2 | non-commercial research only (license.txt); audio synthesised with XTTS-v2 (Coqui Public Model License) | **non-commercial** |
| `iemocap_emotion_test` | IEMOCAP (USC) | USC release license: academic use on request, no commercial use | **access restricted** |
| `iemocap_gender_test` | IEMOCAP (USC) | USC release license: academic use on request, no commercial use | **access restricted** |
| `imda_ar_dialogue` | IMDA National Speech Corpus (MERaLiON MNSC) | Singapore Open Data License v1.0 (per the MERaLiON paper); the derived question and answer texts are not separately licensed | attribution required |
| `imda_ar_sentence` | IMDA National Speech Corpus (MERaLiON MNSC) | Singapore Open Data License v1.0 (per the MERaLiON paper); the derived question and answer texts are not separately licensed | attribution required |
| `imda_gr_dialogue` | IMDA National Speech Corpus (MERaLiON MNSC) | Singapore Open Data License v1.0 (per the MERaLiON paper); the derived question and answer texts are not separately licensed | attribution required |
| `imda_gr_sentence` | IMDA National Speech Corpus (MERaLiON MNSC) | Singapore Open Data License v1.0 (per the MERaLiON paper); the derived question and answer texts are not separately licensed | attribution required |
| `imda_part3_30s_ds_human_test` | IMDA National Speech Corpus (MERaLiON MNSC) | Singapore Open Data License v1.0 (per the MERaLiON paper); the derived question and answer texts are not separately licensed | attribution required |
| `imda_part4_30s_ds_human_test` | IMDA National Speech Corpus (MERaLiON MNSC) | Singapore Open Data License v1.0 (per the MERaLiON paper); the derived question and answer texts are not separately licensed | attribution required |
| `imda_part5_30s_ds_human_test` | IMDA National Speech Corpus (MERaLiON MNSC) | Singapore Open Data License v1.0 (per the MERaLiON paper); the derived question and answer texts are not separately licensed | attribution required |
| `imda_part6_30s_ds_human_test` | IMDA National Speech Corpus (MERaLiON MNSC) | Singapore Open Data License v1.0 (per the MERaLiON paper); the derived question and answer texts are not separately licensed | attribution required |
| `imda_part3_30s_sqa_human_test` | IMDA National Speech Corpus (MERaLiON MNSC) | Singapore Open Data License v1.0 (per the MERaLiON paper); the derived question and answer texts are not separately licensed | attribution required |
| `imda_part4_30s_sqa_human_test` | IMDA National Speech Corpus (MERaLiON MNSC) | Singapore Open Data License v1.0 (per the MERaLiON paper); the derived question and answer texts are not separately licensed | attribution required |
| `imda_part5_30s_sqa_human_test` | IMDA National Speech Corpus (MERaLiON MNSC) | Singapore Open Data License v1.0 (per the MERaLiON paper); the derived question and answer texts are not separately licensed | attribution required |
| `imda_part6_30s_sqa_human_test` | IMDA National Speech Corpus (MERaLiON MNSC) | Singapore Open Data License v1.0 (per the MERaLiON paper); the derived question and answer texts are not separately licensed | attribution required |
| `meld_emotion_test` | MELD (clips of the TV series Friends) | repository states GPL-3.0; the audio is third-party TV content | no clear license |
| `meld_sentiment_test` | MELD (clips of the TV series Friends) | repository states GPL-3.0; the audio is third-party TV content | no clear license |
| `mmau_mini` | MMAU | code Apache-2.0, no dataset license; the pinned copy is deprecated by its authors | no clear license |
| `muchomusic_test` | MuChoMusic (MusicCaps and Song Describer audio) | CC BY-SA 4.0 (GitHub README; Zenodo and the paper say CC BY 4.0); the audio terms are not covered | no clear license |
| `openhermes_audio_test` | OpenHermes 2.5 text, speech synthesised with XTTS-v2 | none stated, mixed component datasets; audio synthesised with XTTS-v2 (Coqui Public Model License) | no clear license |
| `public_sg_speech_qa_test` | Four public speeches by Singaporean politicians | none found | no clear license |
| `slue_p2_sqa5_test` | SLUE-phase-2 SQA-5 | CC BY-SA 4.0 (TriviaQA-derived part Apache-2.0) | attribution required |
| `spoken_squad_test` | Spoken-SQuAD | CC BY-SA 4.0 | attribution required |
| `voxceleb_accent_test` | VoxCeleb1 | CC BY 4.0; copyright of the videos stays with the original owners | attribution required |
| `voxceleb_gender_test` | VoxCeleb1 | CC BY 4.0; copyright of the videos stays with the original owners | attribution required |
| `wavcaps_qa_test` | WavCaps (FreeSound, BBC Sound Effects, SoundBible, AudioSet) | card metadata says CC BY 4.0, its text says academic and research use only | **non-commercial** |
| `wavcaps_test` | WavCaps (FreeSound, BBC Sound Effects, SoundBible, AudioSet) | card metadata says CC BY 4.0, its text says academic and research use only | **non-commercial** |
| `aishell_asr_zh_test` | AISHELL-1 | Apache-2.0 | attribution required |
| `common_voice_15_en_test` | Common Voice 15 | CC0 | permissive |
| `covost2_en_id_test` | CoVoST 2 | CC0 | permissive |
| `covost2_en_ta_test` | CoVoST 2 | CC0 | permissive |
| `covost2_en_zh_test` | CoVoST 2 | CC0 | permissive |
| `covost2_id_en_test` | CoVoST 2 | CC0 | permissive |
| `covost2_ta_en_test` | CoVoST 2 | CC0 | permissive |
| `covost2_zh_en_test` | CoVoST 2 | CC0 | permissive |
| `earnings21_test` | Earnings-21/22 (Rev) | transcripts CC BY-SA 4.0; audio provenance not stated | attribution required |
| `earnings22_test` | Earnings-21/22 (Rev) | transcripts CC BY-SA 4.0; audio provenance not stated | attribution required |
| `gigaspeech_test` | GigaSpeech | gated terms: non-commercial research and education only (card metadata says Apache-2.0) | **non-commercial** |
| `gigaspeech2_indo` | GigaSpeech 2 | gated terms: non-commercial research and education only (card metadata says Apache-2.0) | **non-commercial** |
| `gigaspeech2_thai` | GigaSpeech 2 | gated terms: non-commercial research and education only (card metadata says Apache-2.0) | **non-commercial** |
| `gigaspeech2_viet` | GigaSpeech 2 | gated terms: non-commercial research and education only (card metadata says Apache-2.0) | **non-commercial** |
| `imda_part1_asr_test` | IMDA National Speech Corpus (MERaLiON MNSC) | Singapore Open Data License v1.0 (per the MERaLiON paper); the derived question and answer texts are not separately licensed | attribution required |
| `imda_part2_asr_test` | IMDA National Speech Corpus (MERaLiON MNSC) | Singapore Open Data License v1.0 (per the MERaLiON paper); the derived question and answer texts are not separately licensed | attribution required |
| `imda_part3_30s_asr_test` | IMDA National Speech Corpus (MERaLiON MNSC) | Singapore Open Data License v1.0 (per the MERaLiON paper); the derived question and answer texts are not separately licensed | attribution required |
| `imda_part4_30s_asr_test` | IMDA National Speech Corpus (MERaLiON MNSC) | Singapore Open Data License v1.0 (per the MERaLiON paper); the derived question and answer texts are not separately licensed | attribution required |
| `imda_part5_30s_asr_test` | IMDA National Speech Corpus (MERaLiON MNSC) | Singapore Open Data License v1.0 (per the MERaLiON paper); the derived question and answer texts are not separately licensed | attribution required |
| `imda_part6_30s_asr_test` | IMDA National Speech Corpus (MERaLiON MNSC) | Singapore Open Data License v1.0 (per the MERaLiON paper); the derived question and answer texts are not separately licensed | attribution required |
| `librispeech_test_clean` | LibriSpeech | CC BY 4.0 | attribution required |
| `librispeech_test_other` | LibriSpeech | CC BY 4.0 | attribution required |
| `peoples_speech_test` | The People's Speech | CC BY and CC BY-SA recordings | attribution required |
| `seame_dev_man` | SEAME (LDC2015S04) | LDC member or paid license | **access restricted** |
| `seame_dev_sge` | SEAME (LDC2015S04) | LDC member or paid license | **access restricted** |
| `spoken-mqa_long_digit` | Spoken-MQA | none found | no clear license |
| `spoken-mqa_multi_step_reasoning` | Spoken-MQA | none found | no clear license |
| `spoken-mqa_short_digit` | Spoken-MQA | none found | no clear license |
| `spoken-mqa_single_step_reasoning` | Spoken-MQA | none found | no clear license |
| `tedlium3_test` | TED-LIUM 3 (TED talks) | CC BY-NC-ND 3.0 | **non-commercial** |
| `tedlium3_long_form_test` | TED-LIUM 3 (TED talks) | CC BY-NC-ND 3.0 | **non-commercial** |

## Task

The model answers AudioBench's instruction (multiple-choice sets get the options appended in its `Question:` /
`Choices:` layout). Spoken-MQA uses its own system prompt and the `reasoning` request preset (temperature 0.6, top-p
0.95, seed 0, thinking enabled, 20,480 output tokens); the answer must end in `The answer is: <number>`. All other
profiles drop the system prompt and use greedy decoding with at most 1,024 output tokens (16,384 for the long-form
Earnings and TED-LIUM recordings); these settings were not compared against a reference result.

## Scoring

Recognition and translation use the shared audio scorer, selected with `++eval_type=audiobench`, with normalization
following AudioBench's published definition (the Earnings-21/22 profiles use the Whisper-normalizer snapshot AudioBench
used); the metric is the one AudioBench declares per test set (WER also for Chinese, Thai and code-switched sets;
BLEU with the target language's tokenizer for CoVoST 2). The recognition prompt of Earnings-21/22 is a seeded draw
from transcription instructions written for this benchmark, so it differs from AudioBench's wording. Spoken-MQA
scoring re-implements AudioBench's accuracy definition (the number after the last `The answer is`, compared as a
number) and has not been compared with AudioBench's reported numbers.

**The judge prompt is a paraphrase.** Judged test sets are rated 0-5 by an LLM judge (3 or more counts as correct).
The rubric follows AudioBench's, but the prompt text is reworded and not verified against AudioBench's own, so judged
scores and the Earnings-21/22 WERs (different instruction text) are not comparable with the AudioBench paper or
leaderboard unless verified for your judge. AudioBench's own judges are Llama-3-70B-Instruct and GPT-4o; ratings from
different judges are not comparable.

**Every run needs a judge endpoint, even one that selects no judged test set**, because the server loads all profiles
at startup: provide a chat-completions model server named `judge_model` through `env.yaml` (`judge_base_url`,
`judge_api_key`, `judge_model_name`); the server fails at startup while the endpoint is the `http://unset.local/v1`
placeholder. A failed or unusable judge reply masks the row.

## Metrics

Published as `audio/<profile>/pass@1/*` (profile `audiobench.<test set>`; Spoken-MQA profiles are
`audiobench.spoken-mqa.<split>`), and again under `audio/<profile>-<test set>/pass@1/*`:

| Key | Meaning |
| --- | --- |
| `wer` | Corpus WER in percent (recognition test sets) |
| `bleu` | Corpus BLEU (CoVoST 2) |
| `accuracy` | Spoken-MQA exact-match accuracy in percent (the same number as `success_rate`) |
| `no_numeric_answer_rate` | Spoken-MQA: percent of answers without a number (a diagnostic; such answers count as wrong) |
| `judge_score`, `success_rate` | Judged test sets: mean rating times 20, and percent of ratings of 3 or more |
| `no_answer` | Half of the share of empty generations, as in the other audio benchmarks (see the [audio_eval README](../../resources_servers/audio_eval/README.md)) |

The keys are not all in one unit (percent for `wer`, `accuracy`, `success_rate` and the rates; a 0-100 scale for `judge_score`).

## Provenance

AudioBench's source code is under a Creative Commons NonCommercial license and none of it is copied here: its
repository and revision are referenced for identifiers, dataset pins and metric definitions only, and the scoring
code, prompts and normalization tables in this benchmark are independent implementations (see the headers of
`resources_servers/audio_eval/_scoring/evaluation/evaluator/audiobench.py`).

## Run

```bash
gym eval prepare --benchmark audiobench   # all 63 test sets; add the arguments above to narrow it
```

then see the [audio_eval README](../../resources_servers/audio_eval/README.md#preparing-and-running-an-audio-benchmark)
(`audio_root`, running the benchmark).
