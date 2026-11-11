// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Independently written caller of the unmodified Codex 0.144.4 public API.
// Preparation stages this example beside the pinned upstream crate; no upstream implementation
// or prompt is embedded here. See prepare_model_catalog.py for source and build provenance.
use codex_models_manager::ModelsManagerConfig;
use codex_models_manager::bundled_models_response;
use codex_models_manager::manager::{ModelsManager, StaticModelsManager};
use codex_protocol::openai_models::ModelsResponse;
use std::{env, error::Error, fs, path::PathBuf};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 2 || args[0].is_empty() {
        return Err("expected an exact CLI model alias and an output directory".into());
    }
    let alias = &args[0];
    let output = PathBuf::from(&args[1]);
    let mut catalog = bundled_models_response()?;
    // Retain stock instruction templates; the real CLI still applies its own config overrides.
    let settings = ModelsManagerConfig {
        personality_enabled: true,
        ..Default::default()
    };
    let baseline = StaticModelsManager::new(None, catalog.clone())
        .get_model_info(alias, &settings)
        .await;
    if !baseline.used_fallback_model_metadata {
        return Err("the alias already selects a bundled model; fallback export is not applicable".into());
    }
    let mut modified = baseline.clone();
    modified.context_window = None;
    modified.max_context_window = None;
    modified.auto_compact_token_limit = None;
    let mut before = serde_json::to_value(&baseline)?;
    let mut after = serde_json::to_value(&modified)?;
    for field in ["context_window", "max_context_window", "auto_compact_token_limit"] {
        before.as_object_mut().unwrap().remove(field);
        after.as_object_mut().unwrap().remove(field);
    }
    assert_eq!(before, after, "export changed non-context metadata");
    let original_models = catalog.models.clone();
    catalog.models.push(modified.clone());
    let bytes = serde_json::to_vec_pretty(&catalog)?;
    let reloaded: ModelsResponse = serde_json::from_slice(&bytes)?;
    assert_eq!(&reloaded.models[..original_models.len()], original_models.as_slice());
    let resolved = StaticModelsManager::new(None, reloaded)
        .get_model_info(alias, &settings)
        .await;
    assert!(!resolved.used_fallback_model_metadata);
    // Catalog JSON cannot carry this internal warning marker. Every other Rust field must match.
    modified.used_fallback_model_metadata = false;
    assert_eq!(resolved, modified, "catalog reload changed model metadata");
    assert_eq!(resolved.get_model_instructions(None), baseline.get_model_instructions(None));
    assert_eq!(resolved.resolved_context_window(), None);
    assert_eq!(resolved.auto_compact_token_limit(), None);
    fs::write(output.join("fallback.json"), serde_json::to_vec_pretty(&baseline)?)?;
    fs::write(output.join("models.json"), bytes)?;
    println!("descriptor parity and unknown-window checks passed for {alias}");
    Ok(())
}
