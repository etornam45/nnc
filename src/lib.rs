use std::{
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

use error::{CapabilityReport, CompileError, CompileResult};
use codegen::c::{CCodeGen, CodeGenMode, WeightsMode};
use frontend::onnx::OnnxLoader;

pub mod ir;
pub mod frontend;
pub mod codegen;
pub mod error;

#[derive(Debug, Clone)]
pub struct CompileOptions {
    pub model_path: PathBuf,
    pub out_dir: PathBuf,
    pub mode: CodeGenMode,
    pub weights_mode: WeightsMode,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("out/resnet_model.onnx"),
            out_dir: PathBuf::from("out"),
            mode: CodeGenMode::DummyData,
            weights_mode: WeightsMode::Embedded,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompileArtifacts {
    pub c_path: PathBuf,
    pub weights_path: Option<PathBuf>,
}

pub fn compile_model(options: &CompileOptions) -> CompileResult<CompileArtifacts> {
    ensure_out_dir(&options.out_dir)?;

    let loader = OnnxLoader::new();
    let model_path = options.model_path.to_string_lossy().to_string();
    let graph = loader.load(&model_path)?;
    validate_graph(&graph)?;

    let mut cg = CCodeGen::new(options.mode, options.weights_mode);
    let code = cg.generate(graph.clone());
    let code = code?;

    let c_path = options.out_dir.join("nn.c");
    let mut file = File::create(&c_path)?;
    file.write_all(code.as_bytes())?;

    let weights_path = if options.weights_mode == WeightsMode::External {
        let weights_data = cg.generate_weights_file(&graph);
        let path = options.out_dir.join("weights.bin");
        let mut weights_file = File::create(&path)?;
        weights_file.write_all(&weights_data)?;
        Some(path)
    } else {
        None
    };

    Ok(CompileArtifacts {
        c_path,
        weights_path,
    })
}

pub fn capability_report(options: &CompileOptions) -> CompileResult<CapabilityReport> {
    let loader = OnnxLoader::new();
    loader.capability_report(&options.model_path.to_string_lossy())
}

fn ensure_out_dir(path: &Path) -> CompileResult<()> {
    if !path.exists() {
        fs::create_dir_all(path)?;
    }
    Ok(())
}

fn validate_graph(graph: &ir::Graph) -> CompileResult<()> {
    for node in &graph.nodes {
        let min_inputs = match node.op {
            ir::Op::Add | ir::Op::Sub | ir::Op::Mul | ir::Op::MatMul | ir::Op::Div | ir::Op::Pow => 2,
            ir::Op::Conv2d => 2,
            ir::Op::Reshape => 2,
            ir::Op::Concat | ir::Op::Gather => 2,
            ir::Op::BatchNormalization | ir::Op::LayerNormalization => 3, // X, scale, B (some models use 3, some 2 if bias omitted)
            ir::Op::Relu
            | ir::Op::Softmax
            | ir::Op::Flatten
            | ir::Op::Identity
            | ir::Op::MaxPool
            | ir::Op::GlobalAveragePool
            | ir::Op::Sigmoid
            | ir::Op::Tanh
            | ir::Op::Sqrt
            | ir::Op::Exp
            | ir::Op::Log
            | ir::Op::Gelu
            | ir::Op::ReduceMean
            | ir::Op::Slice
            | ir::Op::Transpose => 1,
            ir::Op::Gemm => 3,
        };
        if node.inputs.len() < min_inputs {
            return Err(CompileError::Validation(format!(
                "node '{}' has {} inputs, expected at least {}",
                node.id,
                node.inputs.len(),
                min_inputs
            )));
        }

        if node.outputs.is_empty() {
            return Err(CompileError::Validation(format!(
                "node '{}' has no outputs",
                node.id
            )));
        }
        for output in &node.outputs {
            if output.is_empty() {
                return Err(CompileError::Validation(format!(
                    "node '{}' has empty output symbol",
                    node.id
                )));
            }
        }

        if matches!(node.op, ir::Op::Concat) && !node.attr.contains_key("axis") {
            return Err(CompileError::Validation(format!(
                "node '{}' (Concat) is missing axis attribute",
                node.id
            )));
        }
    }
    Ok(())
}
