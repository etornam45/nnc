use std::{
    collections::HashMap,
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use candle_onnx::onnx::{GraphProto, ModelProto, NodeProto};
use nnc::{
    codegen::ops::OpGen,
    error::CompileError,
    frontend::onnx::OnnxLoader,
    ir::{Attribute, Node, Op},
    capability_report, CompileOptions,
};
use prost::Message;

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock drift")
        .as_nanos();
    std::env::temp_dir().join(format!("nnc_{prefix}_{ts}"))
}

#[test]
fn returns_structured_error_for_unsupported_op() {
    let work_dir = unique_temp_dir("unsupported_op");
    fs::create_dir_all(&work_dir).expect("failed to create temp directory");
    let model_path = work_dir.join("unsupported.onnx");

    let model = ModelProto {
        graph: Some(GraphProto {
            name: "unsupported_graph".to_string(),
            node: vec![NodeProto {
                op_type: "DoesNotExist".to_string(),
                name: "bad_node".to_string(),
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    fs::write(&model_path, model.encode_to_vec()).expect("failed to write fixture model");

    let loader = OnnxLoader::new();
    let err = loader
        .load(model_path.to_str().expect("model path should be valid utf8"))
        .expect_err("expected unsupported op to fail");

    match err {
        CompileError::UnsupportedOp(op) => assert_eq!(op, "DoesNotExist"),
        other => panic!("expected UnsupportedOp, got {other:?}"),
    }
}

#[test]
fn capability_report_lists_supported_and_unsupported_ops() {
    let work_dir = unique_temp_dir("cap_report");
    fs::create_dir_all(&work_dir).expect("failed to create temp directory");
    let model_path = work_dir.join("mixed_ops.onnx");

    let model = ModelProto {
        graph: Some(GraphProto {
            name: "mixed_ops".to_string(),
            node: vec![
                NodeProto {
                    op_type: "Add".to_string(),
                    name: "add0".to_string(),
                    ..Default::default()
                },
                NodeProto {
                    op_type: "Unknown".to_string(),
                    name: "unknown0".to_string(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        }),
        ..Default::default()
    };
    fs::write(&model_path, model.encode_to_vec()).expect("failed to write fixture model");

    let options = CompileOptions {
        model_path: model_path.clone(),
        ..CompileOptions::default()
    };
    let report = capability_report(&options).expect("report should succeed");
    assert!(report.supported_ops.contains(&"Add".to_string()));
    assert!(report.unsupported_ops.contains(&"Unknown".to_string()));
}

#[test]
fn op_generators_emit_expected_shapes_for_new_ops() {
    let reshape = Node {
        id: "reshape_0".to_string(),
        op: Op::Reshape,
        inputs: vec!["x".to_string(), "shape".to_string()],
        outputs: vec!["y".to_string()],
        attr: HashMap::new(),
    };
    let reshape_code = OpGen::gen_reshape(&reshape);
    assert!(reshape_code.contains("reshape_tensor"));

    let mut concat_attr = HashMap::new();
    concat_attr.insert("axis".to_string(), Some(Attribute::Int(1)));
    let concat = Node {
        id: "concat_0".to_string(),
        op: Op::Concat,
        inputs: vec!["a".to_string(), "b".to_string()],
        outputs: vec!["y".to_string()],
        attr: concat_attr,
    };
    let concat_code = OpGen::gen_concat(&concat).expect("concat should compile");
    assert!(concat_code.contains("cols_a"));

    let bn = Node {
        id: "bn_0".to_string(),
        op: Op::BatchNormalization,
        inputs: vec![
            "x".to_string(),
            "scale".to_string(),
            "bias".to_string(),
            "mean".to_string(),
            "var".to_string(),
        ],
        outputs: vec!["y".to_string()],
        attr: HashMap::new(),
    };
    let bn_code = OpGen::gen_batch_norm(&bn).expect("batch norm should compile");
    assert!(bn_code.contains("sqrtf"));
}
