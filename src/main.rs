use std::{env, fs::{self, File}, io::Write, path::Path};
use nnc::{codegen::c::{CCodeGen, CodeGenMode, WeightsMode}, frontend::onnx::OnnxLoader};



fn main() {
    // Parse command-line arguments for mode
    let args: Vec<String> = env::args().collect();
    
    let mut mode = CodeGenMode::DummyData;
    let mut weights_mode = WeightsMode::Embedded;
    
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--mode" && i + 1 < args.len() {
            mode = match args[i + 1].as_str() {
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
            weights_mode = match args[i + 1].as_str() {
                "external" => WeightsMode::External,
                "embedded" => WeightsMode::Embedded,
                _ => {
                    eprintln!("Unknown weights mode: {}. Using default (embedded).", args[i + 1]);
                    WeightsMode::Embedded
                }
            };
            i += 2;
        } else {
            i += 1;
        }
    }

    println!("Code generation mode: {:?}", mode);
    println!("Weights mode: {:?}", weights_mode);

    // Create output directory
    let out_dir = Path::new("out");
    if !out_dir.exists() {
        fs::create_dir(out_dir).expect("Failed to create out directory");
    }

    let loader = OnnxLoader::new();
    let _ir = OnnxLoader::load(&loader, "out/tinynet.onnx").map_err(|e| e.to_string());
    let graph = _ir.expect("Graph is expected");

    let mut cg = CCodeGen::new(mode, weights_mode);
    let code = cg.generate(graph.clone());
    
    // Write nn.c to out directory
    let nn_path = out_dir.join("nn.c");
    let mut file = File::create(&nn_path).expect("Expected nn.c file but not created");
    let _ = file.write(code.as_bytes());
    println!("Generated {}", nn_path.display());

    // Generate weights file if in external mode
    if weights_mode == WeightsMode::External {
        let weights_data = cg.generate_weights_file(&graph);
        let weights_path = out_dir.join("weights.bin");
        let mut weights_file = File::create(&weights_path).expect("Expected weights.bin file but not created");
        let _ = weights_file.write(&weights_data);
        println!("Generated {} ({} bytes)", weights_path.display(), weights_data.len());
    }

    println!("All files generated successfully in out/ directory!");
}
