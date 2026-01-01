use crate::ir::Node;

pub struct OpGen;

impl OpGen {
    pub fn gen_add(node: Node) -> String {
        let mut code = String::new();

        let input1 = Self::clean_name(&node.inputs[0]);
        let input2 = Self::clean_name(&node.inputs[1]);
        let output = Self::clean_name(&node.outputs[0]);

        Self::emit(
					&mut code,
            &format!(
                "void {}(Tensor* {}, Tensor* {}, Tensor* {}) {{",
                Self::clean_name(&node.id),
                input1,
                input2,
                output
            ),
        );
        Self::emit(&mut code, "\t");

        Self::emit(&mut code, &format!("int size = {}->size;", input1));
        Self::emit(
            &mut code,
            &format!(
                "reshape_tensor({}, {}->ndim, {}->shape);",
                output, input1, input1
            ),
        );

        Self::emit(&mut code, "for (int i = 0; i < size; i++) {");
        Self::emit(&mut code, "\t");
        Self::emit(
            &mut code,
            &format!(
                "{}->data[i] = {}->data[i] + {}->data[i]; }}",
                output, input1, input2
            ),
        );
        // Self::emit(&mut &code, "}");
        // Self::emit(&mut code, "}");

        code
    }

    fn clean_name(name: &str) -> String {
        // Split '/' and take the last string,
        // Replace '.' -> '_'
        let parts: Vec<&str> = name.split('/').collect();
        let last_part = parts.last().map(|s| *s).unwrap_or(name);
        last_part.replace('.', "_")
    }

    fn emit(code: &mut String, line: &str) {
        // let indent = "	".repeat(self.indent_level);
        code.push_str(&format!("{}\n", line));
    }
}
