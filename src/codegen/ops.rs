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
