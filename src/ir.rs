use std::collections::HashMap;

#[derive(Debug, Clone)]
pub enum DataType {
    F32 = 0,
    F16 = 1,
    I32 = 2,
    I8 = 3,
}

#[derive(Debug, Clone)]
pub struct Shape {
   pub dims: Vec<Option<i64>>, // None for dynamic dimentions
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub dtype: i32,
    pub shape: Shape,
    pub data: Option<Vec<u8>>
}

#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub enum Op {
    Add,
    Mul,
    Sub,
    MatMul,
    Conv2d,
    Relu,
    Softmax,
    Flatten,
    Reshape,
    Transpose,
    Concat,
    BatchNormalization,
    Gemm,
    Identity,
    MaxPool,
    GlobalAveragePool,
    Sigmoid,
    Tanh,
    Sqrt,
    Exp,
    Log,
}



#[derive(Debug, Clone)]
pub struct Node {
    pub id: String, //gemm
    pub op: Op, // add, matmul, conv2d, ...
    pub inputs: Vec<String>,  // matmul Y = A * B + C
    pub outputs: Vec<String>,
    pub attr: HashMap<String, Option<Attribute>>,
}

#[derive(Debug, Clone)]
pub enum Attribute {
    Float(f32),
    Int(i64),
    Bool(bool),
    String(String),
    Floats(Vec<f32>),
    Ints(Vec<i64>),
    Strings(Vec<String>),
}


#[derive(Debug, Clone)]
pub struct Graph {
    pub name: String,
    pub nodes: Vec<Node>,
    pub tensors: HashMap<String, Tensor>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>
}
