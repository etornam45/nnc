use std::env;

use nnc::{
    codegen::c::{CodeGenMode, WeightsMode},
    compile_model, capability_report, CompileOptions,
};



fn main() {
    let args: Vec<String> = env::args().collect();
    let mut options = CompileOptions::default();
    let mut report_only = false;

    let mut i = 1;
    while i < args.len() {
        if args[i] == "--mode" && i + 1 < args.len() {
            options.mode = match args[i + 1].as_str() {
                "file" => CodeGenMode::FileInput,
                "lib" => CodeGenMode::LibraryOnly,
                "dummy" => CodeGenMode::DummyData,
                _ => {
                    eprintln!("Unknown mode: {}. Using default (dummy).", args[i + 1]);
                    CodeGenMode::DummyData
                }
            };
            i += 2;
        } else if args[i] == "--weights" && i + 1 < args.len() {
            options.weights_mode = match args[i + 1].as_str() {
                "external" => WeightsMode::External,
                "embedded" => WeightsMode::Embedded,
                _ => {
                    eprintln!("Unknown weights mode: {}. Using default (embedded).", args[i + 1]);
                    WeightsMode::Embedded
                }
            };
            i += 2;
        } else if args[i] == "--model" && i + 1 < args.len() {
            options.model_path = args[i + 1].clone().into();
            i += 2;
        } else if args[i] == "--out-dir" && i + 1 < args.len() {
            options.out_dir = args[i + 1].clone().into();
            i += 2;
        } else if args[i] == "--report" {
            report_only = true;
            i += 1;
        } else {
            i += 1;
        }
    }

    println!("Code generation mode: {:?}", options.mode);
    println!("Weights mode: {:?}", options.weights_mode);
    println!("Model path: {}", options.model_path.display());
    println!("Output directory: {}", options.out_dir.display());

    if report_only {
        let report = capability_report(&options).unwrap_or_else(|e| {
            eprintln!("Capability report failed: {e}");
            std::process::exit(1);
        });
        println!("Capability report for {}", report.model_path.display());
        println!("Supported ops: {:?}", report.supported_ops);
        println!("Unsupported ops: {:?}", report.unsupported_ops);
        return;
    }

    let artifacts = compile_model(&options).unwrap_or_else(|e| {
        eprintln!("Compilation failed: {e}");
        std::process::exit(1);
    });

    println!("Generated {}", artifacts.c_path.display());
    if let Some(weights_path) = artifacts.weights_path {
        println!("Generated {}", weights_path.display());
    }

    println!(
        "All files generated successfully in {}",
        options.out_dir.display()
    );
}
