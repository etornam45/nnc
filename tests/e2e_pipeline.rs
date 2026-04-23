use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use nnc::{
    codegen::c::{CodeGenMode, WeightsMode},
    compile_model, CompileOptions,
};

#[path = "fixtures/minimal_onnx.rs"]
mod minimal_onnx;

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock drift")
        .as_nanos();
    std::env::temp_dir().join(format!("nnc_{prefix}_{ts}"))
}

#[test]
fn generates_c_and_external_weights_for_minimal_fixture() {
    let work_dir = unique_temp_dir("e2e");
    let out_dir = work_dir.join("out");
    fs::create_dir_all(&out_dir).expect("failed to create output directory");

    let model_path = work_dir.join("minimal_model.onnx");
    fs::write(&model_path, minimal_onnx::minimal_model_bytes()).expect("failed to write fixture");

    let options = CompileOptions {
        model_path,
        out_dir: out_dir.clone(),
        mode: CodeGenMode::LibraryOnly,
        weights_mode: WeightsMode::External,
    };

    let artifacts = compile_model(&options).expect("compile_model should succeed for minimal fixture");

    assert!(artifacts.c_path.exists(), "nn.c was not generated");
    let weights_path = artifacts
        .weights_path
        .as_ref()
        .expect("weights path should exist in external mode");
    assert!(weights_path.exists(), "weights.bin was not generated");

    let generated_c =
        fs::read_to_string(&artifacts.c_path).expect("failed to read generated C source file");
    assert!(
        generated_c.contains("void inference("),
        "generated C is missing inference function"
    );
    assert!(
        generated_c.contains("void init_tensor("),
        "generated C is missing tensor helper functions"
    );

    let weights_len = fs::metadata(weights_path)
        .expect("failed to stat weights.bin")
        .len();
    assert_eq!(
        weights_len, 0,
        "minimal fixture should produce an empty weights file"
    );

    fs::remove_dir_all(&work_dir).expect("failed to clean test temp directory");
}
