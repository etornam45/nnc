use std::collections::HashSet;

use crate::ir::Graph;

use super::c::CCodeGen;

#[derive(Debug, Clone)]
pub struct InferencePlan {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub weights: Vec<String>,
    pub intermediates: Vec<String>,
}

impl InferencePlan {
    pub fn from_graph(graph: &Graph) -> Self {
        let inputs: Vec<String> = graph.input_names.iter().map(|n| CCodeGen::clean_name(n)).collect();
        let outputs: Vec<String> = graph.output_names.iter().map(|n| CCodeGen::clean_name(n)).collect();

        let input_set: HashSet<_> = inputs.iter().cloned().collect();
        let output_set: HashSet<_> = outputs.iter().cloned().collect();

        let mut weights = Vec::new();
        let mut intermediates = HashSet::new();

        let mut sorted_tensor_names: Vec<_> = graph.tensors.keys().cloned().collect();
        sorted_tensor_names.sort();

        for name in &sorted_tensor_names {
            let clean = CCodeGen::clean_name(name);
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

        for node in &graph.nodes {
            for name in node.inputs.iter().chain(node.outputs.iter()) {
                let clean = CCodeGen::clean_name(name);
                if !input_set.contains(&clean)
                    && !output_set.contains(&clean)
                    && !weights.contains(&clean)
                    && !intermediates.contains(&clean)
                {
                    intermediates.insert(clean);
                }
            }
        }

        let mut intermediates: Vec<String> = intermediates.into_iter().collect();
        intermediates.sort();

        Self {
            inputs,
            outputs,
            weights,
            intermediates,
        }
    }
}
