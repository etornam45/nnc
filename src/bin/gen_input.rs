use nnc::frontend::onnx::OnnxLoader;
use std::env;
use std::fs::File;
use std::io::{Write};
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut model_path = None;
    let mut data_path = None;
    let mut out_path = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                if i + 1 < args.len() {
                    model_path = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    eprintln!("Error: --model requires a path");
                    std::process::exit(1);
                }
            }
            "--data" => {
                if i + 1 < args.len() {
                    data_path = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    eprintln!("Error: --data requires a path");
                    std::process::exit(1);
                }
            }
            "--out" => {
                if i + 1 < args.len() {
                    out_path = Some(args[i + 1].clone());
                    i += 2;
                } else {
                    eprintln!("Error: --out requires a path");
                    std::process::exit(1);
                }
            }
            _ => i += 1,
        }
    }

    let model_path = model_path.expect("Error: --model <path> is required");
    let out_path = out_path.unwrap_or_else(|| "input.bin".to_string());

    let loader = OnnxLoader::new();
    let graph = loader.load(&model_path).expect("Failed to load ONNX model");

    if graph.input_names.is_empty() {
        eprintln!("Error: No inputs found in the ONNX model");
        std::process::exit(1);
    }

    // Currently we handle the first input. If the model has multiple, 
    // the user might need to specify which one or we concatenate them.
    // For the current nnc scope, one input is typical.
    let input_name = &graph.input_names[0];
    let tensor = graph.tensors.get(input_name).expect("Input tensor not found in graph metadata");
    
    let mut size = 1;
    for dim in &tensor.shape.dims {
        size *= dim.unwrap_or(1);
    }
    
    println!("Model: {}", model_path);
    println!("Input name: '{}'", input_name);
    println!("Expected shape: {:?}", tensor.shape.dims);
    println!("Total elements: {}", size);

    let mut floats = Vec::new();

    if let Some(dp) = data_path {
        println!("Reading data from: {}", dp);
        let content = std::fs::read_to_string(dp).expect("Failed to read text data file");
        // Support space, comma, or newline separation
        for word in content.split(|c: char| c.is_whitespace() || c == ',') {
            if word.is_empty() { continue; }
            if let Ok(f) = word.parse::<f32>() {
                floats.push(f);
            }
        }
        
        if floats.len() != size as usize {
            eprintln!("Warning: Data file has {} numbers, but model expects {}. Data will be truncated or zero-padded.", floats.len(), size);
            if floats.len() > size as usize {
                floats.truncate(size as usize);
            } else {
                while floats.len() < size as usize {
                    floats.push(0.0);
                }
            }
        }
    } else {
        println!("No data file provided. Generating dummy data (all 0.5f).");
        for _ in 0..size {
            floats.push(0.5f32);
        }
    }

    let out_path_buf = PathBuf::from(out_path);
    if let Some(parent) = out_path_buf.parent() {
        std::fs::create_dir_all(parent).expect("Failed to create output directory");
    }

    let mut out_file = File::create(&out_path_buf).expect("Failed to create output binary file");
    for f in floats {
        // C code uses native float, which is typically little-endian on most modern systems (macOS M1 included)
        out_file.write_all(&f.to_le_bytes()).expect("Failed to write to binary file");
    }
    
    println!("Successfully wrote {} floats to {}", size, out_path_buf.display());
}
