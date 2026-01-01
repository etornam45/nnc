use crate::{
    codegen::ops::OpGen,
    ir::{Attribute, Graph, Node, Op, Shape, Tensor},
};
use std::collections::{HashMap, HashSet};

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

    pub fn generate(&mut self, graph: Graph) -> String {
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
            let mut n = op_node.clone();
            self.gen_node(n, graph.clone());
        }

        // Inference
        self.gen_inference(graph.clone());

        // Main (if not library-only mode)
        if self.mode != CodeGenMode::LibraryOnly {
            self.emit("");
            self.gen_main_fn(graph);
        }

        return self.code.clone();
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

        // Identify weights and intermediates
        let mut weights = Vec::new();
        let mut intermediates = HashSet::new();

        // Helper sets for lookups
        let input_set: HashSet<_> = inputs.iter().cloned().collect();
        let output_set: HashSet<_> = outputs.iter().cloned().collect();

        // Iterate over tensors to classify
        // Note: graph.tensors contains both Initializers (data=Some) and ValueInfo (data=None)
        let mut sorted_tensor_names: Vec<_> = graph.tensors.keys().cloned().collect();
        sorted_tensor_names.sort();

        for name in &sorted_tensor_names {
            let clean = Self::clean_name(name);
            if input_set.contains(&clean) || output_set.contains(&clean) {
                continue;
            }

            if let Some(tensor) = graph.tensors.get(name) {
                if tensor.data.is_some() {
                    weights.push(clean);
                } else {
                    intermediates.insert(clean);
                }
            }
        }

        // Also check if any node I/O is missing from graph.tensors (e.g. pure intermediates not in ValueInfo)
        // This is important for implicit intermediates
        for node in &graph.nodes {
            for name in node.inputs.iter().chain(node.outputs.iter()) {
                let clean = Self::clean_name(name);
                if !input_set.contains(&clean)
                    && !output_set.contains(&clean)
                    && !weights.contains(&clean)
                    && !intermediates.contains(&clean)
                {
                    intermediates.insert(clean);
                }
            }
        }

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

    fn gen_node(&mut self, node: Node, graph: Graph) {
        match node.op {
            Op::Add => self.emit(OpGen::gen_add(node).as_str()),
            Op::MatMul => self.gen_matmul(node, graph),
            Op::Conv2d => self.gen_conv_2d(node, graph),
            Op::Relu => self.gen_relu(node, graph),
            Op::Softmax => self.gen_softmax(node, graph),
            Op::Flatten => self.gen_flatten(node, graph),
            Op::Gemm => self.gen_gemm(node, graph),
            Op::Identity => self.gen_identity(node),
            Op::MaxPool => self.gen_max_pool(node),
            Op::GlobalAveragePool => self.gen_global_average_pool(node),
            _ => panic!("Op: {:?} not implemented", node.op),
        }
    }

    fn gen_global_average_pool(&mut self, node: Node) {
        let input = Self::clean_name(&node.inputs[0]);
        let output = Self::clean_name(&node.outputs[0]);

        self.emit(&format!(
            "void {}(Tensor* {}, Tensor* {}) {{",
            Self::clean_name(&node.id),
            input,
            output
        ));
        self.indent();

        // Assuming input is NCHW format.
        self.emit(&format!("int N = {}->shape[0];", input));
        self.emit(&format!("int C = {}->shape[1];", input));
        self.emit(&format!("int H_in = {}->shape[2];", input));
        self.emit(&format!("int W_in = {}->shape[3];", input));

        // Output dimensions are N, C, 1, 1
        self.emit("int output_shape[] = {N, C, 1, 1};");
        self.emit(&format!("reshape_tensor({}, 4, output_shape);", output));

        self.emit("for (int n = 0; n < N; n++) {");
        self.indent();
        self.emit("for (int c = 0; c < C; c++) {");
        self.indent();

        self.emit("float sum = 0.0f;");
        self.emit("for (int h = 0; h < H_in; h++) {");
        self.indent();
        self.emit("for (int w = 0; w < W_in; w++) {");
        self.indent();
        self.emit(&format!(
            "int input_idx = n * (C * H_in * W_in) + c * (H_in * W_in) + h * W_in + w;"
        ));
        self.emit(&format!("sum += {}->data[input_idx];", input));
        self.dedent();
        self.emit("}"); // End for w
        self.dedent();
        self.emit("}"); // End for h

        self.emit("float average = sum / (H_in * W_in);");
        self.emit(&format!(
            "int output_idx = n * (C * 1 * 1) + c * (1 * 1) + 0 * 1 + 0;"
        ));
        self.emit(&format!("{}->data[output_idx] = average;", output));

        self.dedent();
        self.emit("}"); // End for c
        self.dedent();
        self.emit("}"); // End for n

        self.dedent();
        self.emit("}"); // End function
        self.emit("");
    }

    fn gen_max_pool(&mut self, node: Node) {
        let input = Self::clean_name(&node.inputs[0]);
        let output = Self::clean_name(&node.outputs[0]);

        // Extract attributes
        let kernel_shape = node
            .attr
            .get("kernel_shape")
            .and_then(|attr| attr.as_ref())
            .and_then(|attr| match attr {
                Attribute::Ints(v) => Some(v.clone()),
                _ => None,
            })
            .unwrap_or_else(|| {
                panic!(
                    "MaxPool: kernel_shape attribute not found for node {}",
                    node.id
                )
            });

        let strides = node
            .attr
            .get("strides")
            .and_then(|attr| attr.as_ref())
            .and_then(|attr| match attr {
                Attribute::Ints(v) => Some(v.clone()),
                _ => None,
            })
            .unwrap_or_else(|| vec![1, 1]); // Default stride of 1, 1

        let pads = node
            .attr
            .get("pads")
            .and_then(|attr| attr.as_ref())
            .and_then(|attr| match attr {
                Attribute::Ints(v) => Some(v.clone()),
                _ => None,
            })
            .unwrap_or_else(|| vec![0, 0, 0, 0]); // Default padding of 0

        let k_h = kernel_shape[0];
        let kW = kernel_shape[1];
        let sH = strides[0];
        let sW = strides[1];
        let pH_start = pads[0];
        let pW_start = pads[1];
        // let pH_end = pads[2]; // Not used explicitly in loop bounds if output dim calc is correct
        // let pW_end = pads[3]; // Not used explicitly in loop bounds if output dim calc is correct

        self.emit(&format!(
            "void {}(Tensor* {}, Tensor* {}) {{",
            Self::clean_name(&node.id),
            input,
            output
        ));
        self.indent();

        // Assuming input is NCHW format. Need to handle other formats later if necessary.
        // For simplicity, let's assume 4D input for now.
        self.emit(&format!("int N = {}->shape[0];", input));
        self.emit(&format!("int C = {}->shape[1];", input));
        self.emit(&format!("int H_in = {}->shape[2];", input));
        self.emit(&format!("int W_in = {}->shape[3];", input));

        // Calculate output dimensions
        // H_out = floor((H_in + 2*pH - kH) / sH) + 1
        // W_out = floor((W_in + 2*pW - kW) / sW) + 1
        self.emit(&format!(
            "int H_padded = H_in + {} + {};",
            pH_start, pads[2]
        ));
        self.emit(&format!(
            "int W_padded = W_in + {} + {};",
            pW_start, pads[3]
        ));
        self.emit(&format!("int H_out = (H_padded - {}) / {} + 1;", k_h, sH));
        self.emit(&format!("int W_out = (W_padded - {}) / {} + 1;", kW, sW));

        // Reshape output tensor
        self.emit(&format!("int output_shape[] = {{N, C, H_out, W_out}};"));
        self.emit(&format!("reshape_tensor({}, 4, output_shape);", output));

        self.emit("for (int n = 0; n < N; n++) {");
        self.indent();
        self.emit("for (int c = 0; c < C; c++) {");
        self.indent();
        self.emit("for (int h_out = 0; h_out < H_out; h_out++) {");
        self.indent();
        self.emit("for (int w_out = 0; w_out < W_out; w_out++) {");
        self.indent();

        self.emit(&format!("int h_start = h_out * {} - {};", sH, pH_start));
        self.emit(&format!("int w_start = w_out * {} - {};", sW, pW_start));
        self.emit(&format!("int h_end = h_start + {};", k_h));
        self.emit(&format!("int w_end = w_start + {};", kW));

        self.emit("float max_val = -FLT_MAX;"); // Initialize with a very small number

        self.emit("for (int kh = h_start; kh < h_end; kh++) {");
        self.indent();
        self.emit("for (int kw = w_start; kw < w_end; kw++) {");
        self.indent();

        self.emit("if (kh >= 0 && kh < H_in && kw >= 0 && kw < W_in) {");
        self.indent();
        self.emit(&format!(
            "int input_idx = n * (C * H_in * W_in) + c * (H_in * W_in) + kh * W_in + kw;"
        ));
        self.emit(&format!("if ({}->data[input_idx] > max_val) {{", input));
        self.indent();
        self.emit(&format!("max_val = {}->data[input_idx];", input));
        self.dedent();
        self.emit("}");
        self.dedent();
        self.emit("}"); // End if (kh >= 0 && kh < H_in && kw >= 0 && kw < W_in)

        self.dedent();
        self.emit("}"); // End for kw
        self.dedent();
        self.emit("}"); // End for kh

        self.emit(&format!("int output_idx = n * (C * H_out * W_out) + c * (H_out * W_out) + h_out * W_out + w_out;"));
        self.emit(&format!("{}->data[output_idx] = max_val;", output));

        self.dedent();
        self.emit("}"); // End for w_out
        self.dedent();
        self.emit("}"); // End for h_out
        self.dedent();
        self.emit("}"); // End for c
        self.dedent();
        self.emit("}"); // End for n

        self.dedent();
        self.emit("}"); // End function
        self.emit("");
    }

    fn gen_identity(&mut self, node: Node) {
        let input = Self::clean_name(&node.inputs[0]);
        let output = Self::clean_name(&node.outputs[0]);

        self.emit(&format!(
            "void {}(Tensor* {}, Tensor* {}) {{",
            Self::clean_name_num(&node.id),
            input,
            output
        ));
        self.indent();

        self.emit(&format!("int size = {}->size;", input));
        self.emit(&format!(
            "reshape_tensor({}, {}->ndim, {}->shape);",
            output, input, input
        ));

        self.emit("for (int i = 0; i < size; i++) {");
        self.indent();
        self.emit(&format!("{}->data[i] = {}->data[i];", output, input));
        self.dedent();
        self.emit("}");

        self.dedent();
        self.emit("}");
        self.emit("");
    }

    fn gen_softmax(&mut self, node: Node, _graph: Graph) {
        let input = Self::clean_name(&node.inputs[0]);
        let output = Self::clean_name(&node.outputs[0]);

        self.emit(&format!(
            "void {}(Tensor* {}, Tensor* {}) {{",
            Self::clean_name(&node.id),
            input,
            output
        ));
        self.indent();

        self.emit(&format!("int size = {}->size;", input));

        // Output same shape as input
        self.emit(&format!(
            "reshape_tensor({}, {}->ndim, {}->shape);",
            output, input, input
        ));

        // For numerical stability, find max first
        self.emit(&format!("float max_val = {}->data[0];", input));
        self.emit("for (int i = 1; i < size; i++) {");
        self.indent();
        self.emit(&format!("if ({}->data[i] > max_val) {{", input));
        self.indent();
        self.emit(&format!("max_val = {}->data[i];", input));
        self.dedent();
        self.emit("}");
        self.dedent();
        self.emit("}");
        self.emit("");

        // Compute exp(x - max) and sum
        self.emit("float sum = 0.0f;");
        self.emit("for (int i = 0; i < size; i++) {");
        self.indent();
        self.emit(&format!(
            "{}->data[i] = expf({}->data[i] - max_val);",
            output, input
        ));
        self.emit(&format!("sum += {}->data[i];", output));
        self.dedent();
        self.emit("}");
        self.emit("");

        // Normalize by sum
        self.emit("for (int i = 0; i < size; i++) {");
        self.indent();
        self.emit(&format!("{}->data[i] /= sum;", output));
        self.dedent();
        self.emit("}");

        self.dedent();
        self.emit("}");
        self.emit("");
    }

    fn gen_gemm(&mut self, node: Node, _graph: Graph) {
        let a_name = Self::clean_name(&node.inputs[0]);
        let b_name = Self::clean_name(&node.inputs[1]);
        let c_name = Self::clean_name(&node.inputs[2]);
        let output_name = Self::clean_name(&node.outputs[0]);

        let alpha = self.get_attribute_float(&node, "alpha", 1.0);
        let beta = self.get_attribute_float(&node, "beta", 1.0);
        let trans_a = self.get_attribute_int(&node, "transA", 0) == 1;
        let trans_b = self.get_attribute_int(&node, "transB", 0) == 1;

        self.emit(&format!(
            "void {}(Tensor* {}, Tensor* {}, Tensor* {}, Tensor* {}) {{",
            Self::clean_name(&node.id),
            a_name,
            b_name,
            c_name,
            output_name
        ));
        self.indent();

        // Get dimensions at runtime from tensor shapes
        if trans_a {
            self.emit(&format!("int M = {}->shape[1];", a_name));
            self.emit(&format!("int K = {}->shape[0];", a_name));
        } else {
            self.emit(&format!("int M = {}->shape[0];", a_name));
            self.emit(&format!("int K = {}->shape[1];", a_name));
        }

        if trans_b {
            self.emit(&format!("int N = {}->shape[0];", b_name));
        } else {
            self.emit(&format!("int N = {}->shape[1];", b_name));
        }

        // Output Shape: [M, N]
        self.emit("int output_shape[] = {M, N};");
        self.emit(&format!(
            "reshape_tensor({}, 2, output_shape);",
            output_name
        ));

        self.emit("for (int i = 0; i < M; i++) {");
        self.indent();
        self.emit("for (int j = 0; j < N; j++) {");
        self.indent();
        self.emit("float sum = 0.0f;");
        self.emit("for (int k_idx = 0; k_idx < K; k_idx++) {");
        self.indent();

        // Access patterns
        let a_access = if trans_a {
            format!("{}->data[k_idx * {}->shape[1] + i]", a_name, a_name)
        } else {
            format!("{}->data[i * {}->shape[1] + k_idx]", a_name, a_name)
        };
        let b_access = if trans_b {
            format!("{}->data[j * {}->shape[1] + k_idx]", b_name, b_name)
        } else {
            format!("{}->data[k_idx * {}->shape[1] + j]", b_name, b_name)
        };

        self.emit(&format!("sum += {} * {};", a_access, b_access));
        self.dedent();
        self.emit("}");
        self.emit(&format!(
            "{}->data[i * N + j] = {} * sum + {} * {}->data[i * N + j];",
            output_name, alpha, beta, c_name
        ));
        self.dedent();
        self.emit("}");
        self.dedent();
        self.emit("}");
        self.dedent();
        self.emit("}");
        self.emit("");
    }

    fn gen_flatten(&mut self, node: Node, _graph: Graph) {
        let input = Self::clean_name(&node.inputs[0]);
        let output = Self::clean_name(&node.outputs[0]);
        let axis = self.get_attribute_int(&node, "axis", 1);

        self.emit(&format!(
            "void {}(Tensor* {}, Tensor* {}) {{",
            Self::clean_name(&node.id),
            input,
            output
        ));
        self.indent();

        // Calculate 2D shape [dim1, dim2] based on axis
        self.emit("int dim1 = 1;");
        self.emit("int dim2 = 1;");
        self.emit(&format!("int axis = {};", axis));
        self.emit(&format!("if (axis < 0) axis += {}->ndim;", input));

        self.emit("for (int i = 0; i < axis; i++) {");
        self.indent();
        self.emit(&format!("dim1 *= {}->shape[i];", input));
        self.dedent();
        self.emit("}");

        self.emit(&format!("for (int i = axis; i < {}->ndim; i++) {{", input));
        self.indent();
        self.emit(&format!("dim2 *= {}->shape[i];", input));
        self.dedent();
        self.emit("}");

        self.emit("int output_shape[] = {dim1, dim2};");
        self.emit(&format!("reshape_tensor({}, 2, output_shape);", output));

        self.emit(&format!("int size = {}->size;", input));
        self.emit("for (int i = 0; i < size; i++) {");
        self.indent();
        self.emit(&format!("{}->data[i] = {}->data[i];", output, input));
        self.dedent();
        self.emit("}");

        self.dedent();
        self.emit("}");
        self.emit("");
    }

    fn gen_conv_2d(&mut self, node: Node, _graph: Graph) {
        let input = Self::clean_name(&node.inputs[0]);
        let weight = Self::clean_name(&node.inputs[1]);
        let bias = if node.inputs.len() > 2 {
            Some(Self::clean_name(&node.inputs[2]))
        } else {
            None
        };
        let output = Self::clean_name(&node.outputs[0]);

        // Get convolution attributes
        let strides = self.get_attribute_ints(&node, "strides", vec![1, 1]);
        let pads = self.get_attribute_ints(&node, "pads", vec![0, 0, 0, 0]);
        let dilations = self.get_attribute_ints(&node, "dilations", vec![1, 1]);
        let group = self.get_attribute_int(&node, "group", 1);

        let stride_h = strides.get(0).copied().unwrap_or(1);
        let stride_w = strides.get(1).copied().unwrap_or(1);
        let pad_h = pads.get(0).copied().unwrap_or(0);
        let pad_w = pads.get(1).copied().unwrap_or(0);
        let dilation_h = dilations.get(0).copied().unwrap_or(1);
        let dilation_w = dilations.get(1).copied().unwrap_or(1);
        let _group = group; // Groups not yet implemented

        // Generate function signature with or without bias
        if let Some(bias_name) = &bias {
            self.emit(&format!(
                "void {}(Tensor* {}, Tensor* {}, Tensor* {}, Tensor* {}) {{",
                Self::clean_name(&node.id),
                input,
                weight,
                bias_name,
                output
            ));
        } else {
            self.emit(&format!(
                "void {}(Tensor* {}, Tensor* {}, Tensor* {}) {{",
                Self::clean_name(&node.id),
                input,
                weight,
                output
            ));
        }
        self.indent();

        // Get dimensions: input is [N, C, H, W], weight is [OC, IC, KH, KW], output is [N, OC, OH, OW]
        self.emit(&format!("int N = {}->shape[0];", input));
        self.emit(&format!("int C = {}->shape[1];", input));
        self.emit(&format!("int H = {}->shape[2];", input));
        self.emit(&format!("int W = {}->shape[3];", input));
        self.emit(&format!("int OC = {}->shape[0];", weight));
        self.emit(&format!("int IC = {}->shape[1];", weight));
        self.emit(&format!("int KH = {}->shape[2];", weight));
        self.emit(&format!("int KW = {}->shape[3];", weight));

        // Calculate output dimensions
        self.emit(&format!(
            "int OH = (H + 2 * {} - {} * ({}->shape[2] - 1) - 1) / {} + 1;",
            pad_h, dilation_h, weight, stride_h
        ));
        self.emit(&format!(
            "int OW = (W + 2 * {} - {} * ({}->shape[3] - 1) - 1) / {} + 1;",
            pad_w, dilation_w, weight, stride_w
        ));
        self.emit("");

        // Runtime Shape Inference and Allocation
        self.emit("int output_shape[] = {N, OC, OH, OW};");
        self.emit(&format!("reshape_tensor({}, 4, output_shape);", output));
        self.emit("");

        // Initialize output to zero (or bias if present)
        self.emit("for (int n = 0; n < N; n++) {");
        self.indent();
        self.emit("for (int oc = 0; oc < OC; oc++) {");
        self.indent();
        self.emit("for (int oh = 0; oh < OH; oh++) {");
        self.indent();
        self.emit("for (int ow = 0; ow < OW; ow++) {");
        self.indent();
        if let Some(bias_name) = &bias {
            self.emit(&format!(
                "{}->data[((n * OC + oc) * OH + oh) * OW + ow] = {}->data[oc];",
                output, bias_name
            ));
        } else {
            self.emit(&format!(
                "{}->data[((n * OC + oc) * OH + oh) * OW + ow] = 0.0f;",
                output
            ));
        }
        self.dedent();
        self.emit("}");
        self.dedent();
        self.emit("}");
        self.dedent();
        self.emit("}");
        self.dedent();
        self.emit("}");
        self.emit("");

        // Perform convolution
        self.emit("for (int n = 0; n < N; n++) {");
        self.indent();
        self.emit("for (int oc = 0; oc < OC; oc++) {");
        self.indent();
        self.emit("for (int ic = 0; ic < IC; ic++) {");
        self.indent();
        self.emit("for (int kh = 0; kh < KH; kh++) {");
        self.indent();
        self.emit("for (int kw = 0; kw < KW; kw++) {");
        self.indent();
        self.emit("for (int oh = 0; oh < OH; oh++) {");
        self.indent();
        self.emit("for (int ow = 0; ow < OW; ow++) {");
        self.indent();

        // Calculate input indices with padding
        self.emit(&format!(
            "int ih = oh * {} - {} + kh * {};",
            stride_h, pad_h, dilation_h
        ));
        self.emit(&format!(
            "int iw = ow * {} - {} + kw * {};",
            stride_w, pad_w, dilation_w
        ));
        self.emit("if (ih >= 0 && ih < H && iw >= 0 && iw < W) {");
        self.indent();
        self.emit(&format!(
            "{}->data[((n * OC + oc) * OH + oh) * OW + ow] += {}->data[((n * C + ic) * H + ih) * W + iw] * {}->data[((oc * IC + ic) * KH + kh) * KW + kw];",
            output, input, weight
        ));
        self.dedent();
        self.emit("}");
        self.dedent();
        self.emit("}");
        self.dedent();
        self.emit("}");
        self.dedent();
        self.emit("}");
        self.dedent();
        self.emit("}");
        self.dedent();
        self.emit("}");
        self.dedent();
        self.emit("}");

        self.dedent();
        self.emit("}");
        self.dedent();
        self.emit("}");
        self.emit("");
    }

    fn gen_relu(&mut self, node: Node, _graph: Graph) {
        let input = Self::clean_name(&node.inputs[0]);
        let output = Self::clean_name(&node.outputs[0]);

        self.emit(&format!(
            "void {}(Tensor* {}, Tensor* {}) {{",
            Self::clean_name(&node.id),
            input,
            output
        ));
        self.indent();

        self.emit(&format!("int size = {}->size;", input));
        self.emit(&format!(
            "reshape_tensor({}, {}->ndim, {}->shape);",
            output, input, input
        ));

        self.emit("for (int i = 0; i < size; i++) {");
        self.indent();
        self.emit(&format!(
            "{}->data[i] = {}->data[i] > 0 ? {}->data[i] : 0;",
            output, input, input
        ));
        self.dedent();
        self.emit("}");

        self.dedent();
        self.emit("}");
        self.emit("");
    }

    fn gen_add(&mut self, node: Node, _graph: Graph) {
        let input1 = Self::clean_name(&node.inputs[0]);
        let input2 = Self::clean_name(&node.inputs[1]);
        let output = Self::clean_name(&node.outputs[0]);

        self.emit(&format!(
            "void {}(Tensor* {}, Tensor* {}, Tensor* {}) {{",
            Self::clean_name(&node.id),
            input1,
            input2,
            output
        ));
        self.indent();

        self.emit(&format!("int size = {}->size;", input1));
        self.emit(&format!(
            "reshape_tensor({}, {}->ndim, {}->shape);",
            output, input1, input1
        ));

        self.emit("for (int i = 0; i < size; i++) {");
        self.indent();
        self.emit(&format!(
            "{}->data[i] = {}->data[i] + {}->data[i];",
            output, input1, input2
        ));
        self.dedent();
        self.emit("}");
        self.dedent();
        self.emit("}");
    }

    fn gen_matmul(&mut self, node: Node, _graph: Graph) {
        let input1 = Self::clean_name(&node.inputs[0]);
        let input2 = Self::clean_name(&node.inputs[1]);
        let output = Self::clean_name(&node.outputs[0]);

        self.emit(&format!(
            "void {}(Tensor* {}, Tensor* {}, Tensor* {}) {{",
            Self::clean_name(&node.id),
            input1,
            input2,
            output
        ));
        self.indent();

        // Get shapes
        self.emit(&format!("int M = {}->shape[0];", input1));
        self.emit(&format!("int K = {}->shape[1];", input1));
        self.emit(&format!("int N = {}->shape[1];", input2));
        self.emit("");

        // Output Shape: [M, N]
        self.emit("int output_shape[] = {M, N};");
        self.emit(&format!("reshape_tensor({}, 2, output_shape);", output));

        self.emit("for (int i = 0; i < M; i++) {");
        self.indent();
        self.emit("for (int j = 0; j < N; j++) {");
        self.indent();
        self.emit("float sum = 0.0f;");
        self.emit("for (int k = 0; k < K; k++) {");
        self.indent();
        self.emit(&format!(
            "sum += {}->data[i * K + k] * {}->data[k * N + j];",
            input1, input2
        ));
        self.dedent();
        self.emit("}");
        self.emit(&format!("{}->data[i * N + j] = sum;", output));
        self.dedent();
        self.emit("}");
        self.dedent();
        self.emit("}");
        self.emit("");
    }

    fn gen_helpers(&mut self) {
        self.emit(Self::helper_init_tensor().as_str());
        self.emit(Self::helper_free_tensor().as_str());
        self.emit(Self::helper_reshape_tensor().as_str());
    }

    fn helper_reshape_tensor() -> String {
        let mut code = String::new();
        code.push_str(
            r###"
void reshape_tensor(Tensor* tensor, int ndim, const int* shape_values) {
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape_values[i];
    }

    // Check if reallocation is needed
    if (tensor->data != NULL && tensor->size == size) {
        // Size match, update shape just in case (e.g. [1, 100] vs [100, 1])
        if (tensor->ndim != ndim) {
             free(tensor->shape);
             tensor->shape = (int*)malloc(ndim * sizeof(int));
             tensor->ndim = ndim;
        }
        memcpy(tensor->shape, shape_values, ndim * sizeof(int));
        return;
    }

    if (tensor->data != NULL) {
        free(tensor->data);
    }
    if (tensor->shape != NULL) {
        free(tensor->shape);
    }

    tensor->ndim = ndim;
    tensor->shape = (int*)malloc(ndim * sizeof(int));
    if (tensor->shape == NULL) { exit(EXIT_FAILURE); }
    memcpy(tensor->shape, shape_values, ndim * sizeof(int));
    
    tensor->size = size;
    tensor->data = (float*)malloc(size * sizeof(float));
    if (tensor->data == NULL) { free(tensor->shape); exit(EXIT_FAILURE); }
}"###,
        );
        code
    }

    fn get_attribute_float(&self, node: &Node, name: &str, default: f32) -> f32 {
        node.attr
            .get(name)
            .and_then(|attr| {
                if let Some(crate::ir::Attribute::Float(val)) = attr {
                    Some(*val)
                } else {
                    None
                }
            })
            .unwrap_or(default)
    }

    fn get_attribute_int(&self, node: &Node, name: &str, default: i64) -> i64 {
        node.attr
            .get(name)
            .and_then(|attr| {
                if let Some(crate::ir::Attribute::Int(val)) = attr {
                    Some(*val)
                } else {
                    None
                }
            })
            .unwrap_or(default)
    }

    fn get_attribute_ints(&self, node: &Node, name: &str, default: Vec<i64>) -> Vec<i64> {
        node.attr
            .get(name)
            .and_then(|attr| {
                if let Some(crate::ir::Attribute::Ints(vals)) = attr {
                    Some(vals.clone())
                } else {
                    None
                }
            })
            .unwrap_or(default)
    }

    fn clean_name(name: &str) -> String {
        // Split '/' and take the last string,
        // Replace '.' -> '_'
        let parts: Vec<&str> = name.split('/').collect();
        let last_part = parts.last().map(|s| *s).unwrap_or(name);
        last_part.replace('.', "_").replace("onnx::", "")
    }

    fn clean_name_num(name: &str) -> String {
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
