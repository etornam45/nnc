# NNC: Neural Network Compiler

## Overview

NNC is a proof-of-concept neural network compiler. Its goal is to take neural network models, potentially from formats like ONNX, and compile them into efficient code, such as C.

Compiling it to lower level code means you dont need a runtime to execute it while still being more efficient

## Features

- **Frontend**: Support for parsing neural network models (e.g., ONNX).
- **Intermediate Representation (IR)**: An internal representation for neural network graphs.
- **Code Generation**: Generate target code (e.g., C) from the IR.

## Usage

Generate C from an ONNX model:

`cargo run -- --model <path-to-model.onnx> --out-dir out --mode lib --weights external`

Options:

- `--model <path>`: path to input ONNX model (default: `out/resnet_model.onnx`)
- `--out-dir <path>`: output directory for generated artifacts (default: `out`)
- `--mode <dummy|file|lib>`: generation mode
- `--weights <embedded|external>`: embed weights in C or emit `weights.bin`

### Generating Input for File Mode

When using `--mode file`, you need an `input.bin` file. You can generate this using the included `gen_input` utility:

1. **Random/Dummy Data**:
   ```bash
   cargo run --bin gen_input -- --model <path-to-model.onnx> --out out/input.bin
   ```

2. **From Text File**:
   Create a `.txt` file (e.g., `input.txt`) with numbers separated by spaces, commas, or newlines.
   ```bash
   cargo run --bin gen_input -- --model <path-to-model.onnx> --data input.txt --out out/input.bin
   ```

The utility automatically handles shape extraction from the model and provides zero-padding if your text file is smaller than the required input size.

#### Data File Format (`.txt`)
The text file should contain numeric values separated by spaces, commas, or newlines.
```text
0.1, 0.5, 0.2
0.8 1.0
0.0
0.3
```

## Testing

Run the end-to-end pipeline validation:

`cargo test`

The integration test in `tests/e2e_pipeline.rs` writes a deterministic tiny ONNX fixture, runs the full compiler pipeline, and verifies generated artifacts/symbols.

## Adding New Ops Safely

- Extend parsing in `src/frontend/onnx.rs` and codegen in `src/codegen/c.rs`.
- Run `cargo test` to ensure the E2E generation path still works.
- Prefer adding focused fixtures/tests when introducing new operators to prevent regressions.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.