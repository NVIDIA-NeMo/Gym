# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned FLEURS ``test`` split: locale table and data access shared by the FLEURS benchmarks.

Used by ``fleurs_lid`` here; the other FLEURS benchmarks (speech translation, ASR) read the same split through it.
``LOCALES`` maps a locale to ``(language name, region group, number of test clips)``; the counts are those
of the ``test`` split of the pinned ``google/fleurs`` revision and are checked against its TSV files during
preparation. Each locale has one transcript TSV and one audio tarball. Preparation checks the TSV (field count, safe
unique WAV names, the released clip count) and, when it extracts a whole locale, that the tarball holds exactly the
clips the TSV lists. Clips are rewritten as 16-bit PCM WAV files.
"""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path, PurePosixPath
from typing import Iterable, Sequence

from huggingface_hub import hf_hub_download

from benchmarks import audio_manifest
from resources_servers.audio_eval._scoring.dataset.fleurs.languages import FLEURS_LANGUAGES


# Clips per locale in the ``test`` split of the pinned revision (checked against each TSV during preparation).
CLIP_COUNTS: dict[str, int] = {
    "af_za": 264,
    "am_et": 516,
    "ar_eg": 428,
    "as_in": 984,
    "ast_es": 946,
    "az_az": 923,
    "be_by": 967,
    "bg_bg": 658,
    "bn_in": 920,
    "bs_ba": 925,
    "ca_es": 940,
    "ceb_ph": 541,
    "ckb_iq": 922,
    "cmn_hans_cn": 945,
    "cs_cz": 723,
    "cy_gb": 1021,
    "da_dk": 930,
    "de_de": 862,
    "el_gr": 650,
    "en_us": 647,
    "es_419": 908,
    "et_ee": 893,
    "fa_ir": 871,
    "ff_sn": 660,
    "fi_fi": 918,
    "fil_ph": 964,
    "fr_fr": 676,
    "ga_ie": 842,
    "gl_es": 927,
    "gu_in": 1000,
    "ha_ng": 621,
    "he_il": 792,
    "hi_in": 418,
    "hr_hr": 914,
    "hu_hu": 905,
    "hy_am": 932,
    "id_id": 687,
    "ig_ng": 969,
    "is_is": 46,
    "it_it": 865,
    "ja_jp": 650,
    "jv_id": 728,
    "ka_ge": 979,
    "kam_ke": 827,
    "kea_cv": 864,
    "kk_kz": 856,
    "km_kh": 771,
    "kn_in": 838,
    "ko_kr": 382,
    "ky_kg": 977,
    "lb_lu": 934,
    "lg_ug": 723,
    "ln_cd": 478,
    "lo_la": 405,
    "lt_lt": 986,
    "luo_ke": 256,
    "lv_lv": 851,
    "mi_nz": 1008,
    "mk_mk": 973,
    "ml_in": 958,
    "mn_mn": 949,
    "mr_in": 1015,
    "ms_my": 749,
    "mt_mt": 926,
    "my_mm": 880,
    "nb_no": 357,
    "ne_np": 726,
    "nl_nl": 364,
    "nso_za": 790,
    "ny_mw": 761,
    "oc_fr": 998,
    "om_et": 41,
    "or_in": 883,
    "pa_in": 574,
    "pl_pl": 758,
    "ps_af": 512,
    "pt_br": 919,
    "ro_ro": 883,
    "ru_ru": 775,
    "sd_in": 980,
    "sk_sk": 792,
    "sl_si": 834,
    "sn_zw": 925,
    "so_so": 1019,
    "sr_rs": 700,
    "sv_se": 759,
    "sw_ke": 487,
    "ta_in": 591,
    "te_in": 472,
    "tg_tj": 600,
    "th_th": 1021,
    "tr_tr": 743,
    "uk_ua": 750,
    "umb_ao": 379,
    "ur_pk": 299,
    "uz_uz": 862,
    "vi_vn": 857,
    "wo_sn": 371,
    "xh_za": 1041,
    "yo_ng": 831,
    "yue_hant_hk": 819,
    "zu_za": 854,
}

# ``locale -> (language name, region group, number of test clips)``; names and groups are the scoring metadata.
LOCALES: dict[str, tuple[str, str, int]] = {
    locale: (FLEURS_LANGUAGES[locale]["name"], FLEURS_LANGUAGES[locale]["group"], count)
    for locale, count in CLIP_COUNTS.items()
}


HF_REPO_ID = "google/fleurs"
HF_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"  # pragma: allowlist secret (dataset git revision)
SPLIT = "test"
SAMPLE_RATE = 16000


def select_locales(
    languages: str | Sequence[str] | None, *, sets: dict[str, tuple[str, ...]], what: str = "FLEURS"
) -> tuple[str, ...]:
    """Normalize ``languages``: a named set, a comma-separated string or a list of locales (default: all 102)."""
    return audio_manifest.select(languages, LOCALES, f"{what} locales", sets, keep_order=True)


def parse_rows(lines: Iterable[str], locale: str) -> list[dict]:
    """Rows of one locale's transcript TSV in file order, with the format checks of the released data."""
    rows: list[dict] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        line = line.rstrip("\r\n")
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 7:
            raise ValueError(f"FLEURS {locale} TSV row {line_number} has {len(fields)} fields; expected 7")
        row_id, wav_filename, raw_transcription, transcription, *_ = fields
        if not wav_filename.endswith(".wav") or PurePosixPath(wav_filename).name != wav_filename:
            raise ValueError(f"FLEURS {locale} TSV row {line_number} has unsafe WAV filename {wav_filename!r}")
        if wav_filename in seen:
            raise ValueError(f"FLEURS {locale} TSV has duplicate WAV filename {wav_filename!r}")
        seen.add(wav_filename)
        try:
            parsed_id = int(row_id)
        except ValueError as error:
            raise ValueError(f"FLEURS {locale} TSV row {line_number} has non-integer id {row_id!r}") from error
        rows.append(
            {
                "id": parsed_id,
                "raw_transcription": raw_transcription,
                "transcription": transcription,
                "wav_filename": wav_filename,
            }
        )
    return rows


def parse_tsv(path: Path, locale: str) -> list[dict]:
    """Rows of a FLEURS transcript TSV file (see :func:`parse_rows`); ``locale`` names it in error messages."""
    with path.open(encoding="utf-8") as tsv:
        return parse_rows(tsv, locale)


def load_locale(locale: str, cache_dir: Path) -> list[dict]:
    """Download one locale's ``test`` TSV and return its rows; the clip count must match the released count."""
    path = hf_hub_download(
        HF_REPO_ID, f"data/{locale}/{SPLIT}.tsv", repo_type="dataset", revision=HF_REVISION, local_dir=cache_dir
    )
    rows = parse_tsv(Path(path), locale)
    if len(rows) != LOCALES[locale][2]:
        raise ValueError(f"FLEURS {locale} row-count drift: expected {LOCALES[locale][2]}, got {len(rows)}")
    return rows


def extract_locale(
    locale: str, rows: list[dict], wanted: Iterable[str], audio_dir: Path, cache_dir: Path
) -> dict[str, float]:
    """Write the ``wanted`` clips of one locale to ``audio_dir/<locale>/`` and return their durations in seconds.

    ``rows`` are all TSV rows of the locale. The tarball is downloaded, read once and removed; its inventory must
    equal the TSV inventory.
    """
    import soundfile

    wanted = set(wanted)
    all_names = {row["wav_filename"] for row in rows}
    if not wanted <= all_names:
        raise ValueError(f"FLEURS {locale}: requested clips are not in the TSV")
    destination = audio_dir / locale
    destination.mkdir(parents=True, exist_ok=True)
    durations: dict[str, float] = {}
    tar_path = Path(
        hf_hub_download(
            HF_REPO_ID,
            f"data/{locale}/audio/{SPLIT}.tar.gz",
            repo_type="dataset",
            revision=HF_REVISION,
            local_dir=cache_dir,
        )
    )
    try:
        seen: set[str] = set()
        with tarfile.open(tar_path, "r:gz") as archive:
            for member in archive:
                name = PurePosixPath(member.name).name
                if not member.isfile():
                    continue
                if name in seen:
                    raise ValueError(f"FLEURS {locale} tar has duplicate WAV basename {name!r}")
                seen.add(name)
                if name not in wanted:
                    continue
                payload = archive.extractfile(member).read()
                with soundfile.SoundFile(io.BytesIO(payload)) as stream:
                    rate, channels = int(stream.samplerate), int(stream.channels)
                    samples = stream.read(dtype="float32", always_2d=False)
                if rate != SAMPLE_RATE or channels != 1 or len(samples) == 0:
                    raise ValueError(
                        f"FLEURS {locale} clip {name!r} must be non-empty {SAMPLE_RATE} Hz mono; "
                        f"got rate={rate}, channels={channels}, frames={len(samples)}"
                    )
                target = destination / name
                staged = target.with_name(f".{name}.tmp.wav")
                soundfile.write(str(staged), samples, rate, format="WAV", subtype="PCM_16")
                os.replace(staged, target)
                durations[name] = len(samples) / rate
        if seen != all_names:
            raise ValueError(
                f"FLEURS {locale} TSV/tar inventory mismatch: missing={sorted(all_names - seen)[:10]}, "
                f"unexpected={sorted(seen - all_names)[:10]}"
            )
    finally:
        tar_path.unlink(missing_ok=True)
    if set(durations) != wanted:
        raise ValueError(f"FLEURS {locale}: clips missing from the tarball: {sorted(wanted - set(durations))[:10]}")
    return durations
