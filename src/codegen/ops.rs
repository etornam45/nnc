use crate::{
    error::{CompileError, CompileResult},
    ir::{Attribute, Node},
};

pub struct OpGen;

impl OpGen {
    pub fn gen_add(node: &Node) -> String {
        Self::gen_binary_op(node, "+")
    }

    pub fn gen_sub(node: &Node) -> String {
        Self::gen_binary_op(node, "-")
    }

    pub fn gen_mul(node: &Node) -> String {
        Self::gen_binary_op(node, "*")
    }

    fn gen_binary_op(node: &Node, operator: &str) -> String {
        let input1 = Self::clean_name(&node.inputs[0]);
        let input2 = Self::clean_name(&node.inputs[1]);
        let output = Self::clean_name(&node.outputs[0]);
        let mut code = String::new();

        Self::emit(
            &mut code,
            &format!(
                "void {}(Tensor* {}, Tensor* {}, Tensor* {}) {{",
                Self::clean_name_num(&node.id),
                input1,
                input2,
                output
            ),
        );
        Self::emit(&mut code, &format!("int size = {}->size;", input1));
        Self::emit(
            &mut code,
            &format!(
                "reshape_tensor({}, {}->ndim, {}->shape);",
                output, input1, input1
            ),
        );
        Self::emit(&mut code, "for (int i = 0; i < size; i++) {");
        Self::emit(
            &mut code,
            &format!(
                "{}->data[i] = {}->data[i] {} {}->data[i];",
                output, input1, operator, input2
            ),
        );
        Self::emit(&mut code, "}");
        Self::emit(&mut code, "}");
        code
    }

    pub fn gen_reshape(node: &Node) -> String {
        let input = Self::clean_name(&node.inputs[0]);
        let shape_tensor = Self::clean_name(&node.inputs[1]);
        let output = Self::clean_name(&node.outputs[0]);
        let mut code = String::new();

        Self::emit(
            &mut code,
            &format!(
                "void {}(Tensor* {}, Tensor* {}, Tensor* {}) {{",
                Self::clean_name_num(&node.id),
                input,
                shape_tensor,
                output
            ),
        );
        Self::emit(&mut code, &format!("int out_ndim = {}->size;", shape_tensor));
        Self::emit(
            &mut code,
            "int* out_shape = (int*)malloc(out_ndim * sizeof(int));",
        );
        Self::emit(&mut code, "for (int i = 0; i < out_ndim; i++) {");
        Self::emit(
            &mut code,
            &format!("out_shape[i] = (int){}->data[i];", shape_tensor),
        );
        Self::emit(&mut code, "}");
        Self::emit(
            &mut code,
            &format!("reshape_tensor({}, out_ndim, out_shape);", output),
        );
        Self::emit(&mut code, &format!("for (int i = 0; i < {}->size; i++) {{", input));
        Self::emit(
            &mut code,
            &format!("{}->data[i] = {}->data[i];", output, input),
        );
        Self::emit(&mut code, "}");
        Self::emit(&mut code, "free(out_shape);");
        Self::emit(&mut code, "}");
        code
    }

    pub fn gen_transpose(node: &Node) -> CompileResult<String> {
        let input = Self::clean_name(&node.inputs[0]);
        let output = Self::clean_name(&node.outputs[0]);
        let perm = Self::get_ints_attr(node, "perm").unwrap_or_else(|| vec![1, 0]);
        if perm.len() != 2 {
            return Err(CompileError::Codegen(
                "transpose currently supports 2D tensors only".to_string(),
            ));
        }
        let mut code = String::new();
        Self::emit(
            &mut code,
            &format!(
                "void {}(Tensor* {}, Tensor* {}) {{",
                Self::clean_name_num(&node.id),
                input,
                output
            ),
        );
        Self::emit(
            &mut code,
            &format!("int output_shape[] = {{{}->shape[1], {}->shape[0]}};", input, input),
        );
        Self::emit(
            &mut code,
            &format!("reshape_tensor({}, 2, output_shape);", output),
        );
        Self::emit(&mut code, &format!("for (int i = 0; i < {}->shape[0]; i++) {{", input));
        Self::emit(&mut code, &format!("for (int j = 0; j < {}->shape[1]; j++) {{", input));
        Self::emit(
            &mut code,
            &format!("{}->data[j * {}->shape[0] + i] = {}->data[i * {}->shape[1] + j];", output, input, input, input),
        );
        Self::emit(&mut code, "}");
        Self::emit(&mut code, "}");
        Self::emit(&mut code, "}");
        Ok(code)
    }

    pub fn gen_concat(node: &Node) -> CompileResult<String> {
        let axis = Self::get_int_attr(node, "axis").unwrap_or(1);
        if axis != 1 {
            return Err(CompileError::Codegen(
                "concat currently supports axis=1 only".to_string(),
            ));
        }
        let a = Self::clean_name(&node.inputs[0]);
        let b = Self::clean_name(&node.inputs[1]);
        let output = Self::clean_name(&node.outputs[0]);
        let mut code = String::new();
        Self::emit(
            &mut code,
            &format!(
                "void {}(Tensor* {}, Tensor* {}, Tensor* {}) {{",
                Self::clean_name_num(&node.id),
                a,
                b,
                output
            ),
        );
        Self::emit(
            &mut code,
            &format!("int output_shape[] = {{{}->shape[0], {}->shape[1] + {}->shape[1]}};", a, a, b),
        );
        Self::emit(
            &mut code,
            &format!("reshape_tensor({}, 2, output_shape);", output),
        );
        Self::emit(&mut code, &format!("int cols_a = {}->shape[1];", a));
        Self::emit(&mut code, &format!("int cols_b = {}->shape[1];", b));
        Self::emit(&mut code, "for (int r = 0; r < output_shape[0]; r++) {");
        Self::emit(&mut code, "for (int c = 0; c < cols_a; c++) {");
        Self::emit(
            &mut code,
            &format!("{}->data[r * output_shape[1] + c] = {}->data[r * cols_a + c];", output, a),
        );
        Self::emit(&mut code, "}");
        Self::emit(&mut code, "for (int c = 0; c < cols_b; c++) {");
        Self::emit(
            &mut code,
            &format!("{}->data[r * output_shape[1] + cols_a + c] = {}->data[r * cols_b + c];", output, b),
        );
        Self::emit(&mut code, "}");
        Self::emit(&mut code, "}");
        Self::emit(&mut code, "}");
        Ok(code)
    }

    pub fn gen_batch_norm(node: &Node) -> CompileResult<String> {
        let x = Self::clean_name(&node.inputs[0]);
        let scale = Self::clean_name(&node.inputs[1]);
        let bias = Self::clean_name(&node.inputs[2]);
        let mean = Self::clean_name(&node.inputs[3]);
        let var = Self::clean_name(&node.inputs[4]);
        let y = Self::clean_name(&node.outputs[0]);
        let epsilon = Self::get_float_attr(node, "epsilon").unwrap_or(1e-5);
        let mut code = String::new();
        Self::emit(
            &mut code,
            &format!(
                "void {}(Tensor* {}, Tensor* {}, Tensor* {}, Tensor* {}, Tensor* {}, Tensor* {}) {{",
                Self::clean_name_num(&node.id),
                x,
                scale,
                bias,
                mean,
                var,
                y
            ),
        );
        Self::emit(
            &mut code,
            &format!("reshape_tensor({}, {}->ndim, {}->shape);", y, x, x),
        );
        Self::emit(&mut code, &format!("int N = {}->shape[0];", x));
        Self::emit(&mut code, &format!("int C = {}->shape[1];", x));
        Self::emit(
            &mut code,
            &format!("int inner = {}->size / (N * C);", x),
        );
        Self::emit(&mut code, "for (int n = 0; n < N; n++) {");
        Self::emit(&mut code, "for (int c = 0; c < C; c++) {");
        Self::emit(&mut code, "for (int i = 0; i < inner; i++) {");
        Self::emit(&mut code, "int idx = (n * C + c) * inner + i;");
        Self::emit(
            &mut code,
            &format!(
                "{}->data[idx] = (({}->data[idx] - {}->data[c]) / sqrtf({}->data[c] + {}f)) * {}->data[c] + {}->data[c];",
                y, x, mean, var, epsilon, scale, bias
            ),
        );
        Self::emit(&mut code, "}");
        Self::emit(&mut code, "}");
        Self::emit(&mut code, "}");
        Self::emit(&mut code, "}");
        Ok(code)
    }

    pub fn gen_sigmoid(node: &Node) -> String {
        Self::gen_unary_math_op(node, "1.0 / (1.0 + expf(-x))")
    }

    pub fn gen_tanh(node: &Node) -> String {
        Self::gen_unary_math_op(node, "tanhf(x)")
    }

    pub fn gen_sqrt(node: &Node) -> String {
        Self::gen_unary_math_op(node, "sqrtf(x)")
    }

    pub fn gen_exp(node: &Node) -> String {
        Self::gen_unary_math_op(node, "expf(x)")
    }

    pub fn gen_log(node: &Node) -> String {
        Self::gen_unary_math_op(node, "logf(x)")
    }

    fn gen_unary_math_op(node: &Node, formula: &str) -> String {
        let input = Self::clean_name(&node.inputs[0]);
        let output = Self::clean_name(&node.outputs[0]);
        let mut code = String::new();

        Self::emit(&mut code, &format!("void {}(Tensor* {}, Tensor* {}) {{", Self::clean_name_num(&node.id), input, output));
        Self::emit(&mut code, &format!("\tint size = {}->size;", input));
        Self::emit(&mut code, &format!("\treshape_tensor({}, {}->ndim, {}->shape);", output, input, input));
        Self::emit(&mut code, "\tfor (int i = 0; i < size; i++) {");
        Self::emit(&mut code, &format!("\t\tfloat x = {}->data[i];", input));
        Self::emit(&mut code, &format!("\t\t{}->data[i] = {};", output, formula.replace("x", "x")));
        Self::emit(&mut code, "\t}");
        Self::emit(&mut code, "}");
        code
    }

    pub fn gen_matmul(node: &Node) -> String {
        let input1 = Self::clean_name(&node.inputs[0]);
        let input2 = Self::clean_name(&node.inputs[1]);
        let output = Self::clean_name(&node.outputs[0]);
        let mut code = String::new();

        Self::emit(&mut code, &format!("void {}(Tensor* {}, Tensor* {}, Tensor* {}) {{", Self::clean_name_num(&node.id), input1, input2, output));
        Self::emit(&mut code, &format!("\tint M = {}->shape[0];", input1));
        Self::emit(&mut code, &format!("\tint K = {}->shape[1];", input1));
        Self::emit(&mut code, &format!("\tint N = {}->shape[1];", input2));
        Self::emit(&mut code, "\tint output_shape[] = {M, N};");
        Self::emit(&mut code, &format!("\treshape_tensor({}, 2, output_shape);", output));

        Self::emit(&mut code, "\tfor (int i = 0; i < M; i++) {");
        Self::emit(&mut code, "\t\tfor (int j = 0; j < N; j++) {");
        Self::emit(&mut code, "\t\t\tfloat sum = 0.0f;");
        Self::emit(&mut code, "\t\t\tfor (int k = 0; k < K; k++) {");
        Self::emit(&mut code, &format!("\t\t\t\tsum += {}->data[i * K + k] * {}->data[k * N + j];", input1, input2));
        Self::emit(&mut code, "\t\t\t}");
        Self::emit(&mut code, &format!("\t\t\t{}->data[i * N + j] = sum;", output));
        Self::emit(&mut code, "\t\t}");
        Self::emit(&mut code, "\t}");
        Self::emit(&mut code, "}");
        code
    }

    pub fn gen_relu(node: &Node) -> String {
        let input = Self::clean_name(&node.inputs[0]);
        let output = Self::clean_name(&node.outputs[0]);
        let mut code = String::new();

        Self::emit(&mut code, &format!("void {}(Tensor* {}, Tensor* {}) {{", Self::clean_name_num(&node.id), input, output));
        Self::emit(&mut code, &format!("\tint size = {}->size;", input));
        Self::emit(&mut code, &format!("\treshape_tensor({}, {}->ndim, {}->shape);", output, input, input));
        Self::emit(&mut code, "\tfor (int i = 0; i < size; i++) {");
        Self::emit(&mut code, &format!("\t\t{}->data[i] = {}->data[i] > 0 ? {}->data[i] : 0;", output, input, input));
        Self::emit(&mut code, "\t}");
        Self::emit(&mut code, "}");
        code
    }

    pub fn gen_softmax(node: &Node) -> String {
        let input = Self::clean_name(&node.inputs[0]);
        let output = Self::clean_name(&node.outputs[0]);
        let mut code = String::new();

        Self::emit(&mut code, &format!("void {}(Tensor* {}, Tensor* {}) {{", Self::clean_name_num(&node.id), input, output));
        Self::emit(&mut code, &format!("\tint size = {}->size;", input));
        Self::emit(&mut code, &format!("\treshape_tensor({}, {}->ndim, {}->shape);", output, input, input));
        Self::emit(&mut code, &format!("\tfloat max_val = {}->data[0];", input));
        Self::emit(&mut code, "\tfor (int i = 1; i < size; i++) {");
        Self::emit(&mut code, &format!("\t\tif ({}->data[i] > max_val) max_val = {}->data[i];", input, input));
        Self::emit(&mut code, "\t}");
        Self::emit(&mut code, "\tfloat sum = 0.0f;");
        Self::emit(&mut code, "\tfor (int i = 0; i < size; i++) {");
        Self::emit(&mut code, &format!("\t\t{}->data[i] = expf({}->data[i] - max_val);", output, input));
        Self::emit(&mut code, &format!("\t\tsum += {}->data[i];", output));
        Self::emit(&mut code, "\t}");
        Self::emit(&mut code, "\tfor (int i = 0; i < size; i++) {");
        Self::emit(&mut code, &format!("\t\t{}->data[i] /= sum;", output));
        Self::emit(&mut code, "\t}");
        Self::emit(&mut code, "}");
        code
    }

    pub fn gen_flatten(node: &Node) -> String {
        let input = Self::clean_name(&node.inputs[0]);
        let output = Self::clean_name(&node.outputs[0]);
        let axis = Self::get_int_attr(node, "axis").unwrap_or(1);
        let mut code = String::new();

        Self::emit(&mut code, &format!("void {}(Tensor* {}, Tensor* {}) {{", Self::clean_name_num(&node.id), input, output));
        Self::emit(&mut code, "\tint dim1 = 1;");
        Self::emit(&mut code, "\tint dim2 = 1;");
        Self::emit(&mut code, &format!("\tint axis = {};", axis));
        Self::emit(&mut code, &format!("\tif (axis < 0) axis += {}->ndim;", input));
        Self::emit(&mut code, &format!("\tfor (int i = 0; i < axis; i++) dim1 *= {}->shape[i];", input));
        Self::emit(&mut code, &format!("\tfor (int i = axis; i < {}->ndim; i++) dim2 *= {}->shape[i];", input, input));
        Self::emit(&mut code, "\tint output_shape[] = {dim1, dim2};");
        Self::emit(&mut code, &format!("\treshape_tensor({}, 2, output_shape);", output));
        Self::emit(&mut code, &format!("\tfor (int i = 0; i < {}->size; i++) {}->data[i] = {}->data[i];", input, output, input));
        Self::emit(&mut code, "}");
        code
    }

    pub fn gen_max_pool(node: &Node) -> String {
        let input = Self::clean_name(&node.inputs[0]);
        let output = Self::clean_name(&node.outputs[0]);
        let kernel_shape = Self::get_ints_attr(node, "kernel_shape").unwrap_or_else(|| vec![1, 1]);
        let strides = Self::get_ints_attr(node, "strides").unwrap_or_else(|| vec![1, 1]);
        let pads = Self::get_ints_attr(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
        let mut code = String::new();

        Self::emit(&mut code, &format!("void {}(Tensor* {}, Tensor* {}) {{", Self::clean_name_num(&node.id), input, output));
        Self::emit(&mut code, &format!("\tint N = {}->shape[0], C = {}->shape[1], H_in = {}->shape[2], W_in = {}->shape[3];", input, input, input, input));
        let k_h = kernel_shape[0]; let k_w = kernel_shape[1];
        let s_h = strides[0]; let s_w = strides[1];
        let p_h = pads[0]; let p_w = pads[1];
        Self::emit(&mut code, &format!("\tint H_out = (H_in + {} + {} - {}) / {} + 1;", p_h, pads[2], k_h, s_h));
        Self::emit(&mut code, &format!("\tint W_out = (W_in + {} + {} - {}) / {} + 1;", p_w, pads[3], k_w, s_w));
        Self::emit(&mut code, "\tint output_shape[] = {N, C, H_out, W_out};");
        Self::emit(&mut code, &format!("\treshape_tensor({}, 4, output_shape);", output));

        Self::emit(&mut code, "\tfor (int n = 0; n < N; n++) {");
        Self::emit(&mut code, "\t\tfor (int c = 0; c < C; c++) {");
        Self::emit(&mut code, "\t\t\tfor (int h = 0; h < H_out; h++) {");
        Self::emit(&mut code, "\t\t\t\tfor (int w = 0; w < W_out; w++) {");
        Self::emit(&mut code, &format!("\t\t\t\t\tint h_start = h * {} - {}, w_start = w * {} - {};", s_h, p_h, s_w, p_w));
        Self::emit(&mut code, "\t\t\t\t\tfloat max_val = -FLT_MAX;");
        Self::emit(&mut code, &format!("\t\t\t\t\tfor (int kh = 0; kh < {}; kh++) {{", k_h));
        Self::emit(&mut code, &format!("\t\t\t\t\t\tfor (int kw = 0; kw < {}; kw++) {{", k_w));
        Self::emit(&mut code, "\t\t\t\t\t\t\tint ih = h_start + kh, iw = w_start + kw;");
        Self::emit(&mut code, "\t\t\t\t\t\t\tif (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {");
        Self::emit(&mut code, &format!("\t\t\t\t\t\t\t\tint idx = ((n * C + c) * H_in + ih) * W_in + iw;"));
        Self::emit(&mut code, &format!("\t\t\t\t\t\t\t\tif ({}->data[idx] > max_val) max_val = {}->data[idx];", input, input));
        Self::emit(&mut code, "\t\t\t\t\t\t\t}");
        Self::emit(&mut code, "\t\t\t\t\t\t}");
        Self::emit(&mut code, "\t\t\t\t\t}");
        Self::emit(&mut code, &format!("\t\t\t\t\t{}->data[((n * C + c) * H_out + h) * W_out + w] = max_val;", output));
        Self::emit(&mut code, "\t\t\t\t}");
        Self::emit(&mut code, "\t\t\t}");
        Self::emit(&mut code, "\t\t}");
        Self::emit(&mut code, "\t}");
        Self::emit(&mut code, "}");
        code
    }

    pub fn gen_global_average_pool(node: &Node) -> String {
        let input = Self::clean_name(&node.inputs[0]);
        let output = Self::clean_name(&node.outputs[0]);
        let mut code = String::new();

        Self::emit(&mut code, &format!("void {}(Tensor* {}, Tensor* {}) {{", Self::clean_name_num(&node.id), input, output));
        Self::emit(&mut code, &format!("\tint N = {}->shape[0], C = {}->shape[1], H = {}->shape[2], W = {}->shape[3];", input, input, input, input));
        Self::emit(&mut code, "\tint output_shape[] = {N, C, 1, 1};");
        Self::emit(&mut code, &format!("\treshape_tensor({}, 4, output_shape);", output));
        Self::emit(&mut code, "\tfor (int n = 0; n < N; n++) {");
        Self::emit(&mut code, "\t\tfor (int c = 0; c < C; c++) {");
        Self::emit(&mut code, "\t\t\tfloat sum = 0.0f;");
        Self::emit(&mut code, "\t\t\tfor (int i = 0; i < H * W; i++) {");
        Self::emit(&mut code, &format!("\t\t\t\tsum += {}->data[(n * C + c) * H * W + i];", input));
        Self::emit(&mut code, "\t\t\t}");
        Self::emit(&mut code, &format!("\t\t\t{}->data[n * C + c] = sum / (H * W);", output));
        Self::emit(&mut code, "\t\t}");
        Self::emit(&mut code, "\t}");
        Self::emit(&mut code, "}");
        code
    }

    pub fn gen_gemm(node: &Node) -> String {
        let a = Self::clean_name(&node.inputs[0]);
        let b = Self::clean_name(&node.inputs[1]);
        let c = if node.inputs.len() > 2 { Some(Self::clean_name(&node.inputs[2])) } else { None };
        let output = Self::clean_name(&node.outputs[0]);
        let alpha = Self::get_float_attr(node, "alpha").unwrap_or(1.0);
        let beta = Self::get_float_attr(node, "beta").unwrap_or(1.0);
        let trans_a = Self::get_int_attr(node, "transA").unwrap_or(0) == 1;
        let trans_b = Self::get_int_attr(node, "transB").unwrap_or(0) == 1;
        let mut code = String::new();

        let sig = if let Some(ref c_name) = c {
            format!("void {}(Tensor* {}, Tensor* {}, Tensor* {}, Tensor* {}) {{", Self::clean_name_num(&node.id), a, b, c_name, output)
        } else {
            format!("void {}(Tensor* {}, Tensor* {}, Tensor* {}) {{", Self::clean_name_num(&node.id), a, b, output)
        };
        Self::emit(&mut code, &sig);

        if trans_a {
            Self::emit(&mut code, &format!("\tint M = {}->shape[1], K = {}->shape[0];", a, a));
        } else {
            Self::emit(&mut code, &format!("\tint M = {}->shape[0], K = {}->shape[1];", a, a));
        }
        if trans_b {
            Self::emit(&mut code, &format!("\tint N = {}->shape[0];", b));
        } else {
            Self::emit(&mut code, &format!("\tint N = {}->shape[1];", b));
        }
        Self::emit(&mut code, "\tint output_shape[] = {M, N};");
        Self::emit(&mut code, &format!("\treshape_tensor({}, 2, output_shape);", output));

        Self::emit(&mut code, "\tfor (int i = 0; i < M; i++) {");
        Self::emit(&mut code, "\t\tfor (int j = 0; j < N; j++) {");
        Self::emit(&mut code, "\t\t\tfloat sum = 0.0f;");
        Self::emit(&mut code, "\t\t\tfor (int k = 0; k < K; k++) {");
        let a_idx = if trans_a { format!("k * {}->shape[1] + i", a) } else { format!("i * K + k") };
        let b_idx = if trans_b { format!("j * {}->shape[1] + k", b) } else { format!("k * N + j") };
        Self::emit(&mut code, &format!("\t\t\t\tsum += {}->data[{}] * {}->data[{}];", a, a_idx, b, b_idx));
        Self::emit(&mut code, "\t\t\t}");
        if let Some(ref c_name) = c {
            Self::emit(&mut code, &format!("\t\t\t{}->data[i * N + j] = {} * sum + {} * {}->data[i * N + j];", output, alpha, beta, c_name));
        } else {
            Self::emit(&mut code, &format!("\t\t\t{}->data[i * N + j] = {} * sum;", output, alpha));
        }
        Self::emit(&mut code, "\t\t}");
        Self::emit(&mut code, "\t}");
        Self::emit(&mut code, "}");
        code
    }

    pub fn gen_conv_2d(node: &Node) -> String {
        let input = Self::clean_name(&node.inputs[0]);
        let weight = Self::clean_name(&node.inputs[1]);
        let bias = if node.inputs.len() > 2 { Some(Self::clean_name(&node.inputs[2])) } else { None };
        let output = Self::clean_name(&node.outputs[0]);

        let strides = Self::get_ints_attr(node, "strides").unwrap_or_else(|| vec![1, 1]);
        let pads = Self::get_ints_attr(node, "pads").unwrap_or_else(|| vec![0, 0, 0, 0]);
        let dilations = Self::get_ints_attr(node, "dilations").unwrap_or_else(|| vec![1, 1]);

        let mut code = String::new();
        let sig = if let Some(ref b) = bias {
            format!("void {}(Tensor* {}, Tensor* {}, Tensor* {}, Tensor* {}) {{", Self::clean_name_num(&node.id), input, weight, b, output)
        } else {
            format!("void {}(Tensor* {}, Tensor* {}, Tensor* {}) {{", Self::clean_name_num(&node.id), input, weight, output)
        };
        Self::emit(&mut code, &sig);

        Self::emit(&mut code, &format!("\tint N = {}->shape[0], C = {}->shape[1], H = {}->shape[2], W = {}->shape[3];", input, input, input, input));
        Self::emit(&mut code, &format!("\tint OC = {}->shape[0], IC = {}->shape[1], KH = {}->shape[2], KW = {}->shape[3];", weight, weight, weight, weight));

        let sh = strides[0]; let sw = strides[1];
        let ph = pads[0]; let pw = pads[1];
        let dh = dilations[0]; let dw = dilations[1];

        Self::emit(&mut code, &format!("\tint OH = (H + {} + {} - {} * (KH - 1) - 1) / {} + 1;", ph, pads[2], dh, sh));
        Self::emit(&mut code, &format!("\tint OW = (W + {} + {} - {} * (KW - 1) - 1) / {} + 1;", pw, pads[3], dw, sw));
        Self::emit(&mut code, "\tint output_shape[] = {N, OC, OH, OW};");
        Self::emit(&mut code, &format!("\treshape_tensor({}, 4, output_shape);", output));

        Self::emit(&mut code, "\tfor (int n = 0; n < N; n++) {");
        Self::emit(&mut code, "\t\tfor (int oc = 0; oc < OC; oc++) {");
        Self::emit(&mut code, "\t\t\tfor (int oh = 0; oh < OH; oh++) {");
        Self::emit(&mut code, "\t\t\t\tfor (int ow = 0; ow < OW; ow++) {");
        let base_idx = "((n * OC + oc) * OH + oh) * OW + ow";
        if let Some(ref b) = bias {
            Self::emit(&mut code, &format!("\t\t\t\t\t{}->data[{}] = {}->data[oc];", output, base_idx, b));
        } else {
            Self::emit(&mut code, &format!("\t\t\t\t\t{}->data[{}] = 0.0f;", output, base_idx));
        }
        Self::emit(&mut code, "\t\t\t\t}");
        Self::emit(&mut code, "\t\t\t}");
        Self::emit(&mut code, "\t\t}");
        Self::emit(&mut code, "\t}");

        Self::emit(&mut code, "\tfor (int n = 0; n < N; n++) {");
        Self::emit(&mut code, "\t\tfor (int oc = 0; oc < OC; oc++) {");
        Self::emit(&mut code, "\t\t\tfor (int ic = 0; ic < IC; ic++) {");
        Self::emit(&mut code, "\t\t\t\tfor (int kh = 0; kh < KH; kh++) {");
        Self::emit(&mut code, "\t\t\t\t\tfor (int kw = 0; kw < KW; kw++) {");
        Self::emit(&mut code, "\t\t\t\t\t\tfor (int oh = 0; oh < OH; oh++) {");
        Self::emit(&mut code, "\t\t\t\t\t\t\tfor (int ow = 0; ow < OW; ow++) {");
        Self::emit(&mut code, &format!("\t\t\t\t\t\t\t\tint ih = oh * {} - {} + kh * {};", sh, ph, dh));
        Self::emit(&mut code, &format!("\t\t\t\t\t\t\t\tint iw = ow * {} - {} + kw * {};", sw, pw, dw));
        Self::emit(&mut code, "\t\t\t\t\t\t\t\tif (ih >= 0 && ih < H && iw >= 0 && iw < W) {");
        Self::emit(&mut code, &format!("\t\t\t\t\t\t\t\t\t{}->data[((n * OC + oc) * OH + oh) * OW + ow] += {}->data[((n * C + ic) * H + ih) * W + iw] * {}->data[((oc * IC + ic) * KH + kh) * KW + kw];", output, input, weight));
        Self::emit(&mut code, "\t\t\t\t\t\t\t\t}");
        Self::emit(&mut code, "\t\t\t\t\t\t\t}");
        Self::emit(&mut code, "\t\t\t\t\t\t}");
        Self::emit(&mut code, "\t\t\t\t\t}");
        Self::emit(&mut code, "\t\t\t\t}");
        Self::emit(&mut code, "\t\t\t}");
        Self::emit(&mut code, "\t\t}");
        Self::emit(&mut code, "\t}");
        Self::emit(&mut code, "}");
        code
    }

    pub fn gen_identity(node: &Node) -> String {
        let input = Self::clean_name(&node.inputs[0]);
        let output = Self::clean_name(&node.outputs[0]);
        let mut code = String::new();

        Self::emit(&mut code, &format!("void {}(Tensor* {}, Tensor* {}) {{", Self::clean_name_num(&node.id), input, output));
        Self::emit(&mut code, &format!("\treshape_tensor({}, {}->ndim, {}->shape);", output, input, input));
        Self::emit(&mut code, &format!("\tfor (int i = 0; i < {}->size; i++) {}->data[i] = {}->data[i];", input, output, input));
        Self::emit(&mut code, "}");
        code
    }

    fn get_int_attr(node: &Node, attr_name: &str) -> Option<i64> {
        node.attr
            .get(attr_name)
            .and_then(|v| v.as_ref())
            .and_then(|v| match v {
                Attribute::Int(val) => Some(*val),
                _ => None,
            })
    }

    fn get_ints_attr(node: &Node, attr_name: &str) -> Option<Vec<i64>> {
        node.attr
            .get(attr_name)
            .and_then(|v| v.as_ref())
            .and_then(|v| match v {
                Attribute::Ints(vals) => Some(vals.clone()),
                _ => None,
            })
    }

    fn get_float_attr(node: &Node, attr_name: &str) -> Option<f32> {
        node.attr
            .get(attr_name)
            .and_then(|v| v.as_ref())
            .and_then(|v| match v {
                Attribute::Float(val) => Some(*val),
                _ => None,
            })
    }

    fn clean_name(name: &str) -> String {
        let parts: Vec<&str> = name.split('/').collect();
        let last_part = parts.last().copied().unwrap_or(name);
        last_part.replace('.', "_").replace("onnx::", "")
    }

    fn clean_name_num(name: &str) -> String {
        let clean = Self::clean_name(name);
        clean
            .chars()
            .filter(|c| !c.is_ascii_digit())
            .collect::<String>()
            .trim_end_matches('_')
            .to_string()
    }

    fn emit(code: &mut String, line: &str) {
        code.push_str(line);
        code.push('\n');
    }
}
