use crate::ir::{Attribute, Graph, Node, Op, Shape, Tensor};
use candle_onnx::onnx::{
    tensor_shape_proto::{dimension, Dimension},
    type_proto, ModelProto, NodeProto, TensorProto, TypeProto, ValueInfoProto,
};
use prost::Message;
use std::{collections::HashMap, fs::File, io::Read};

pub struct OnnxLoader;

impl OnnxLoader {
    pub fn new() -> OnnxLoader {
        return OnnxLoader;
    }

    pub fn load(&self, path: &str) -> Result<Graph, String> {
        let mut file = File::open(path).map_err(|e| e.to_string())?;
        let mut buffer = Vec::new();

        file.read_to_end(&mut buffer).map_err(|e| e.to_string())?;

        let model = ModelProto::decode(prost::bytes::Bytes::from(buffer)).map_err(|e| e.to_string())?;

        self.convert_model(model)
    }

    fn convert_model(&self, model: ModelProto) -> Result<Graph, String> {
        let graph_proto = model.graph.ok_or("No graph in model")?;

        let mut graph = Graph {
            name: graph_proto.name.clone(),
            nodes: Vec::new(),
            tensors: HashMap::new(),
            input_names: Vec::new(),
            output_names: Vec::new(),
        };

        // Load Initializers (Weights/Biases)
        for tensor_proto in &graph_proto.initializer {
            let tensor = self.convert_tensor(tensor_proto);
            graph.tensors.insert(tensor.name.clone(), tensor);
        }

        // Load Value Info (Intermediate Shapes)
        let all_value_infos = graph_proto
            .input
            .iter()
            .chain(graph_proto.output.iter())
            .chain(graph_proto.value_info.iter());

        for value_info in all_value_infos {
            if !graph.tensors.contains_key(&value_info.name) {
                if let Some(tensor) = self.convert_value_info(value_info) {
                    graph.tensors.insert(tensor.name.clone(), tensor);
                }
            }
        }

        for node_proto in &graph_proto.node {
            let node = self.convert_node(node_proto)?;
            graph.nodes.push(node);
        }

        for input in &graph_proto.input {
            graph.input_names.push(input.name.clone());
        }
        for output in &graph_proto.output {
            graph.output_names.push(output.name.clone());
        }

        Ok(graph)
    }

    fn convert_tensor(&self, tensor: &TensorProto) -> Tensor {
        return Tensor {
            name: tensor.name.clone(),
            dtype: tensor.data_type,
            shape: Shape {
                dims: tensor.dims.clone().into_iter().map(Some).collect(),
            },
            data: Some(tensor.raw_data.clone()),
        };
    }

    fn convert_value_info(&self, value_info: &ValueInfoProto) -> Option<Tensor> {
        let type_proto = value_info.r#type.as_ref()?;
        let value = type_proto.value.as_ref()?;

        match value {
            type_proto::Value::TensorType(tensor_type) => {
                let elem_type = tensor_type.elem_type;
                let shape_proto = tensor_type.shape.as_ref()?;
                
                let dims = shape_proto.dim.iter().map(|d| {
                    match &d.value {
                        Some(dimension::Value::DimValue(v)) => Some(*v),
                        _ => None // Dynamic or Param
                    }
                }).collect();

                Some(Tensor {
                    name: value_info.name.clone(),
                    dtype: elem_type,
                    shape: Shape { dims },
                    data: None,
                })
            }
            _ => None, // Not a tensor (Sequence, Map, etc.)
        }
    }

    fn convert_node(&self, node: &NodeProto) -> Result<Node, String> {
        let op = match node.op_type.as_str() {
            "MatMul" => Op::MatMul,
            "Add" => Op::Add,
            "Conv" => Op::Conv2d,
            "Relu" => Op::Relu,
            "Softmax" => Op::Softmax,
            "Flatten" => Op::Flatten,
            "Gemm" => Op::Gemm,
            _ => return Err(format!("Unsupported Op: {}", node.op_type)),
        };

        // Attribute parsing
        let mut attr = HashMap::new();
        for attribute in &node.attribute {
            let val = match attribute.r#type {
                1 => Some(Attribute::Float(attribute.f)), // FLOAT
                2 => Some(Attribute::Int(attribute.i)), // INT
                3 => Some(Attribute::String(String::from_utf8_lossy(&attribute.s).to_string())), // STRING
                6 => Some(Attribute::Floats(attribute.floats.clone())), // FLOATS
                7 => Some(Attribute::Ints(attribute.ints.clone())), // INTS
                8 => Some(Attribute::Strings(attribute.strings.iter().map(|s| String::from_utf8_lossy(s).to_string()).collect())), // STRINGS
                _ => None,
            };

            if let Some(v) = val {
                attr.insert(attribute.name.clone(), Some(v));
            }
        }

        Ok(Node {
            id: node.name.clone(),
            op,
            inputs: node.input.clone(),
            outputs: node.output.clone(),
            attr,
        })
    }
}
