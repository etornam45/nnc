use crate::{
    codegen::ops::OpGen,
    error::CompileResult,
    ir::{Graph, Node, Op},
};
use std::collections::{HashMap, HashSet};

use super::{planner::InferencePlan, runtime};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CodeGenMode {
    DummyData,
    FileInput,
    LibraryOnly,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightsMode {
    Embedded,
    External,
}

pub struct CCodeGen {
    indent_level: usize,
    code: String,
    mode: CodeGenMode,
    weights_mode: WeightsMode,
    op_set: HashMap<Op, Node>,
}

impl CCodeGen {
    pub fn new(mode: CodeGenMode, weights_mode: WeightsMode) -> CCodeGen {
        return CCodeGen {
            indent_level: 0,
            code: String::new(),
            mode,
            weights_mode,
            op_set: HashMap::new(),
        };
    }

    pub fn generate(&mut self, graph: Graph) -> CompileResult<String> {
        //TODO: Build Hash set of all operators and then I generate the code for it
        for node in &graph.nodes {
            self.op_set.insert(node.op.clone(), node.clone());
        }

        println!("{:?}", self.op_set.keys());

        self.code.clear();
        // return self.code;

        // Header

        self.emit("#include <stdlib.h>");
        self.emit("#include <string.h>");
        self.emit("#include <math.h>");
        self.emit("#include <stdio.h>");
        self.emit("#include <float.h>");
        self.emit("");

        // Tensor Struct
        self.emit("typedef struct {");
        self.indent();
        self.emit("float* data;");
        self.emit("int* shape;");
        self.emit("int ndim;");
        self.emit("int size;");
        self.dedent();
        self.emit("} Tensor; \n");

        // Generate Helper Functions
        self.gen_helpers();

        // NODES
        let ops_to_process: Vec<_> = self.op_set.values().cloned().collect();
        for op_node in ops_to_process {
            self.gen_node(op_node.clone(), graph.clone())?;
        }

        // Inference
        self.gen_inference(graph.clone());

        // Main (if not library-only mode)
        if self.mode != CodeGenMode::LibraryOnly {
            self.emit("");
            self.gen_main_fn(graph);
        }

        Ok(self.code.clone())
    }

    pub fn generate_weights_file(&self, graph: &Graph) -> Vec<u8> {
        let inputs: Vec<String> = graph
            .input_names
            .iter()
            .map(|n| Self::clean_name(n))
            .collect();
        let outputs: Vec<String> = graph
            .output_names
            .iter()
            .map(|n| Self::clean_name(n))
            .collect();

        let input_set: HashSet<_> = inputs.iter().cloned().collect();
        let output_set: HashSet<_> = outputs.iter().cloned().collect();

        let mut sorted_tensor_names: Vec<_> = graph.tensors.keys().cloned().collect();
        sorted_tensor_names.sort();

        let mut weights_data = Vec::new();

        for name in &sorted_tensor_names {
            let clean = Self::clean_name(name);
            if input_set.contains(&clean) || output_set.contains(&clean) {
                continue;
            }

            if let Some(tensor) = graph.tensors.get(name) {
                if let Some(raw_data) = &tensor.data {
                    weights_data.extend_from_slice(raw_data);
                }
            }
        }

        weights_data
    }

    fn gen_main_fn(&mut self, graph: Graph) {
        let inputs: Vec<String> = graph
            .input_names
            .iter()
            .map(|n| Self::clean_name(n))
            .collect();
        let outputs: Vec<String> = graph
            .output_names
            .iter()
            .map(|n| Self::clean_name(n))
            .collect();

        // Identify weights - strict copy of gen_inference logic to match signature
        let mut weights = Vec::new();
        let input_set: HashSet<_> = inputs.iter().cloned().collect();
        let output_set: HashSet<_> = outputs.iter().cloned().collect();

        let mut sorted_tensor_names: Vec<_> = graph.tensors.keys().cloned().collect();
        sorted_tensor_names.sort();

        self.emit("int main(int argc, char** argv) {");
        self.indent();

        // Mode-specific setup
        if self.mode == CodeGenMode::FileInput {
            self.emit("if (argc < 3) {");
            self.indent();
            self.emit("fprintf(stderr, \"Usage: %s <input_file> <output_file>\\n\", argv[0]);");
            self.emit("return 1;");
            self.dedent();
            self.emit("}");
            self.emit("");
        }

        // 1. Initialize Inputs
        for (i, input_clean) in inputs.iter().enumerate() {
            // Retrieve original name to get shape. input_names order matches inputs order.
            let original_name = &graph.input_names[i];

            let mut shape_vals = vec![1];
            if let Some(tensor) = graph.tensors.get(original_name) {
                shape_vals = tensor
                    .shape
                    .dims
                    .iter()
                    .map(|d| d.unwrap_or(1) as i32)
                    .collect();
            }

            let shape_str = shape_vals
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            self.emit(&format!("int {}_shape[] = {{{}}};", input_clean, shape_str));
            self.emit(&format!("Tensor {};", input_clean));
            self.emit(&format!(
                "init_tensor(&{}, {}, {}_shape);",
                input_clean,
                shape_vals.len(),
                input_clean
            ));

            // Fill data based on mode
            if self.mode == CodeGenMode::FileInput {
                self.emit("FILE* input_file = fopen(argv[1], \"rb\");");
                self.emit("if (!input_file) {");
                self.indent();
                self.emit("fprintf(stderr, \"Error opening input file\\n\");");
                self.emit("return 1;");
                self.dedent();
                self.emit("}");
                self.emit(&format!(
                    "size_t read = fread({}.data, sizeof(float), {}.size, input_file);",
                    input_clean, input_clean
                ));
                self.emit(&format!("if (read != {}.size) {{", input_clean));
                self.indent();
                self.emit(&format!(
                    "fprintf(stderr, \"Error: expected %d floats, read %zu\\n\", {}.size, read);",
                    input_clean
                ));
                self.emit("fclose(input_file);");
                self.emit("return 1;");
                self.dedent();
                self.emit("}");
                self.emit("fclose(input_file);");
            } else {
                self.emit(&format!(
                    "for(int i=0; i<{}.size; i++) {} .data[i] = 0.5f;",
                    input_clean, input_clean
                ));
            }
            self.emit("");
        }

        // 2. Initialize Outputs
        for output in &outputs {
            self.emit(&format!("Tensor {} = {{0}};", output));
        }
        self.emit("");

        // 3. Initialize Weights
        if self.weights_mode == WeightsMode::External {
            // Load all weights from external file
            self.emit("// Load weights from file");
            self.emit("FILE* weights_file = fopen(\"weights.bin\", \"rb\");");
            self.emit("if (!weights_file) {");
            self.indent();
            self.emit("fprintf(stderr, \"Error: could not open weights.bin\\n\");");
            self.emit("return 1;");
            self.dedent();
            self.emit("}");
            self.emit("");
        }

        for name in &sorted_tensor_names {
            let clean = Self::clean_name(name);
            if input_set.contains(&clean) || output_set.contains(&clean) {
                continue;
            }

            if let Some(tensor) = graph.tensors.get(name) {
                if tensor.data.is_some() {
                    weights.push(clean.clone());

                    // Init code
                    let shape_vals: Vec<i32> = tensor
                        .shape
                        .dims
                        .iter()
                        .map(|d| d.unwrap_or(1) as i32)
                        .collect();
                    let shape_str = shape_vals
                        .iter()
                        .map(|v| v.to_string())
                        .collect::<Vec<_>>()
                        .join(", ");

                    self.emit(&format!("int {}_shape[] = {{{}}};", clean, shape_str));
                    self.emit(&format!("Tensor {};", clean));
                    self.emit(&format!(
                        "init_tensor(&{}, {}, {}_shape);",
                        clean,
                        shape_vals.len(),
                        clean
                    ));

                    if self.weights_mode == WeightsMode::External {
                        // Read from weights file
                        if let Some(raw_data) = &tensor.data {
                            let num_floats = raw_data.len() / 4;
                            self.emit(&format!(
                                "if (fread({}.data, sizeof(float), {}, weights_file) != {}) {{",
                                clean, num_floats, num_floats
                            ));
                            self.indent();
                            self.emit("fprintf(stderr, \"Error reading weights from file\\n\");");
                            self.emit("fclose(weights_file);");
                            self.emit("return 1;");
                            self.dedent();
                            self.emit("}");
                        }
                    } else {
                        // Embedded mode - use C arrays
                        if let Some(raw_data) = &tensor.data {
                            let floats: Vec<f32> = raw_data
                                .chunks_exact(4)
                                .map(|chunk| {
                                    let b = chunk.try_into().unwrap();
                                    f32::from_le_bytes(b)
                                })
                                .collect();

                            let float_strs: Vec<String> =
                                floats.iter().map(|f| format!("{:?}f", f)).collect();
                            self.emit(&format!(
                                "float {}_data[] = {{{}}};",
                                clean,
                                float_strs.join(", ")
                            ));
                            self.emit(&format!(
                                "memcpy({}.data, {}_data, {} * sizeof(float));",
                                clean,
                                clean,
                                floats.len()
                            ));
                        }
                    }
                    self.emit("");
                }
            }
        }

        if self.weights_mode == WeightsMode::External {
            self.emit("fclose(weights_file);");
            self.emit("");
        }

        // 4. Call Inference
        let mut args = Vec::new();
        for input in &inputs {
            args.push(format!("&{}", input));
        }
        for output in &outputs {
            args.push(format!("&{}", output));
        }
        for weight in &weights {
            args.push(format!("&{}", weight));
        }

        self.emit(&format!("inference({});", args.join(", ")));
        self.emit("");

        // Print or save output based on mode
        if !outputs.is_empty() {
            if self.mode == CodeGenMode::FileInput {
                self.emit("// Write output to file");
                self.emit("FILE* output_file = fopen(argv[2], \"wb\");");
                self.emit("if (!output_file) {");
                self.indent();
                self.emit("fprintf(stderr, \"Error opening output file\\n\");");
                self.emit("return 1;");
                self.dedent();
                self.emit("}");
                self.emit(&format!(
                    "fwrite({}.data, sizeof(float), {}.size, output_file);",
                    outputs[0], outputs[0]
                ));
                self.emit("fclose(output_file);");
                self.emit(&format!(
                    "printf(\"Wrote %d floats to output file\\n\", {}.size);",
                    outputs[0]
                ));
            } else {
                self.emit("// Print first output value for verification");
                self.emit(&format!("if ({}.data != NULL) {{", outputs[0]));
                self.indent();
                self.emit(&format!("printf(\"Output: [\");"));
                self.emit(&format!("for (int i = 0; i < {}.size; i++) {{", outputs[0]));
                self.indent();
                self.emit(&format!("printf(\"%f%s\", {}.data[i], (i == {}.size - 1 ? \"\" : \", \"));", outputs[0], outputs[0]));
                self.dedent();
                self.emit("}");
                self.emit(&format!("printf(\"]\"); \n"));
                self.dedent();
                self.emit("}");
            }
        }

        // 5. Cleanup
        for input in &inputs {
            self.emit(&format!("free_tensor(&{});", input));
        }
        for output in &outputs {
            self.emit(&format!("free_tensor(&{});", output));
        }
        for weight in &weights {
            self.emit(&format!("free_tensor(&{});", weight));
        }

        self.emit("return 0;");
        self.dedent();
        self.emit("}");
    }

    fn gen_inference(&mut self, graph: Graph) {
        let plan = InferencePlan::from_graph(&graph);
        let inputs = plan.inputs;
        let outputs = plan.outputs;
        let weights = plan.weights;
        let intermediates: HashSet<String> = plan.intermediates.into_iter().collect();

        let input_set: HashSet<_> = inputs.iter().cloned().collect();
        let output_set: HashSet<_> = outputs.iter().cloned().collect();

        // Generate function signature
        let mut args = Vec::new();
        for input in &inputs {
            args.push(format!("Tensor* {}", input));
        }
        for output in &outputs {
            args.push(format!("Tensor* {}", output));
        }
        for weight in &weights {
            args.push(format!("Tensor* {}", weight));
        }

        self.emit(&format!("void inference({}) {{", args.join(", ")));
        self.indent();

        // Declare internal (intermediate) tensors
        // We use static to avoid re-allocation every time if possible,
        // relying on reshape_tensor to handle sizing.
        // Initialize to 0 so data is NULL.
        for name in &intermediates {
            self.emit(&format!("static Tensor {} = {{0}};", name));
        }
        self.emit("");

        // Execute graph nodes
        for node in &graph.nodes {
            let func_name = Self::clean_name_num(&node.id);
            let mut call_args = Vec::new();

            // Inputs
            for input in &node.inputs {
                let clean = Self::clean_name(input);
                if input_set.contains(&clean)
                    || output_set.contains(&clean)
                    || weights.contains(&clean)
                {
                    call_args.push(clean);
                } else {
                    call_args.push(format!("&{}", clean));
                }
            }

            // Outputs
            for output in &node.outputs {
                let clean = Self::clean_name(output);
                if output_set.contains(&clean)
                    || input_set.contains(&clean)
                    || weights.contains(&clean)
                {
                    call_args.push(clean);
                } else {
                    call_args.push(format!("&{}", clean));
                }
            }

            self.emit(&format!("{}({});", func_name, call_args.join(", ")));
        }

        self.dedent();
        self.emit("}");
    }

    #[allow(dead_code)]
    fn helper_init_tensor() -> String {
        let mut code = String::new();
        code.push_str(
            r###"
void init_tensor(Tensor* tensor, int ndim, const int* shape_values) {
	tensor->ndim = ndim;
	tensor->shape = (int*)malloc(ndim * sizeof(int));
	if (tensor->shape == NULL) {
			// Handle allocation error
			exit(EXIT_FAILURE);
	}
	memcpy(tensor->shape, shape_values, ndim * sizeof(int));

	tensor->size = 1;
	for (int i = 0; i < ndim; i++) {
			tensor->size *= shape_values[i];
	}

	tensor->data = (float*)malloc(tensor->size * sizeof(float));
	if (tensor->data == NULL) {
			// Handle allocation error, free shape if allocated
			free(tensor->shape);
			exit(EXIT_FAILURE);
	}
}"###,
        );
        code
    }

    #[allow(dead_code)]
    fn helper_free_tensor() -> String {
        let mut code = String::new();
        code.push_str(
            r###"
void free_tensor(Tensor* tensor) {
	if (tensor->data != NULL) {
			free(tensor->data);
			tensor->data = NULL; // Prevent double-free
	}
	if (tensor->shape != NULL) {
			free(tensor->shape);
			tensor->shape = NULL;
	}
}"###,
        );

        code
    }

    fn gen_node(&mut self, node: Node, _graph: Graph) -> CompileResult<()> {
        match node.op {
            Op::Add => self.emit(OpGen::gen_add(&node).as_str()),
            Op::Sub => self.emit(OpGen::gen_sub(&node).as_str()),
            Op::Mul => self.emit(OpGen::gen_mul(&node).as_str()),
            Op::MatMul => self.emit(OpGen::gen_matmul(&node).as_str()),
            Op::Conv2d => self.emit(OpGen::gen_conv_2d(&node).as_str()),
            Op::Relu => self.emit(OpGen::gen_relu(&node).as_str()),
            Op::Softmax => self.emit(OpGen::gen_softmax(&node).as_str()),
            Op::Flatten => self.emit(OpGen::gen_flatten(&node).as_str()),
            Op::Reshape => self.emit(OpGen::gen_reshape(&node).as_str()),
            Op::Transpose => self.emit(OpGen::gen_transpose(&node)?.as_str()),
            Op::Concat => self.emit(OpGen::gen_concat(&node)?.as_str()),
            Op::BatchNormalization => self.emit(OpGen::gen_batch_norm(&node)?.as_str()),
            Op::Gemm => self.emit(OpGen::gen_gemm(&node).as_str()),
            Op::Identity => self.emit(OpGen::gen_identity(&node).as_str()),
            Op::MaxPool => self.emit(OpGen::gen_max_pool(&node).as_str()),
            Op::GlobalAveragePool => self.emit(OpGen::gen_global_average_pool(&node).as_str()),
            Op::Sigmoid => self.emit(OpGen::gen_sigmoid(&node).as_str()),
            Op::Tanh => self.emit(OpGen::gen_tanh(&node).as_str()),
            Op::Sqrt => self.emit(OpGen::gen_sqrt(&node).as_str()),
            Op::Exp => self.emit(OpGen::gen_exp(&node).as_str()),
            Op::Log => self.emit(OpGen::gen_log(&node).as_str()),
            Op::LayerNormalization => self.emit(OpGen::gen_layer_norm(&node).as_str()),
            Op::Gelu => self.emit(OpGen::gen_gelu(&node).as_str()),
            Op::Gather => self.emit(OpGen::gen_gather(&node).as_str()),
            Op::Slice => self.emit(OpGen::gen_slice(&node).as_str()),
            Op::ReduceMean => self.emit(OpGen::gen_reduce_mean(&node).as_str()),
            Op::Div => self.emit(OpGen::gen_div(&node).as_str()),
            Op::Pow => self.emit(OpGen::gen_pow(&node).as_str()),
        }
        Ok(())
    }


    fn gen_helpers(&mut self) {
        self.emit(runtime::helper_init_tensor());
        self.emit(runtime::helper_free_tensor());
        self.emit(runtime::helper_reshape_tensor());
    }

    pub fn clean_name(name: &str) -> String {
        // Split '/' and take the last string,
        // Replace '.' -> '_'
        let parts: Vec<&str> = name.split('/').collect();
        let last_part = parts.last().map(|s| *s).unwrap_or(name);
        last_part.replace('.', "_").replace("onnx::", "")
    }

    pub fn clean_name_num(name: &str) -> String {
        // Split '/' and take the last string,
        // Replace '.' -> '_'
        let parts: Vec<&str> = name.split('/').collect();
        let last_part = parts.last().map(|s| *s).unwrap_or(name);
        let _n = last_part.replace('.', "_").replace("onnx::", "");

        let mut cleaned_name = String::new();
        for c in _n.chars() {
            if !c.is_ascii_digit() {
                cleaned_name.push(c);
            }
        }
        cleaned_name
            .trim_end_matches('_')
            .to_string()
            .replace("onnx::", "")
    }

    fn emit(&mut self, line: &str) {
        let indent = "	".repeat(self.indent_level);
        self.code.push_str(&format!("{}{}\n", indent, line));
    }

    fn indent(&mut self) {
        self.indent_level = self.indent_level + 1
    }

    fn dedent(&mut self) {
        if self.indent_level > 0 {
            self.indent_level = self.indent_level - 1
        }
    }
}
