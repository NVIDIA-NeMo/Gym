# ocsr resources server

Verifier for optical chemical structure recognition (molecule image to SMILES). Used by the
`ocsr` benchmark (USPTO-OCSR-5719, see `benchmarks/ocsr/README.md`) and usable for any
image-to-SMILES set whose rows carry `verifier_metadata.gold_smiles`.

- **Input row**: `responses_create_params.input` with the image as an `input_image` data URL;
  `verifier_metadata = {"id", "gold_smiles", "source_of_label"?}`.
- **Answer**: last `{"smiles": ...}` JSON object in the final message, reasoning removed; else the last
  RDKit-valid SMILES in backticks.
- **Reward**: `exact`, RDKit canonical isomeric SMILES equality (0/1).
- **Diagnostics**: `graph` (no stereo), `inchikey`, `tanimoto` (Morgan r=2, 2048), `valid`, `format_ok`,
  `extracted_smiles`.
- **Masking**: a gold label RDKit cannot parse gives `mask_sample=True`, `failure_kind=ocsr:bad_gold`.
- **Stateless**, no external services; RDKit pinned in `requirements.txt`.

`data/example.jsonl` holds 5 RDKit-rendered drug molecules (caffeine, aspirin, paracetamol,
(S)-naproxen, sulfamethoxazole), not benchmark images.
