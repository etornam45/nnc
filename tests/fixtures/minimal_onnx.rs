use candle_onnx::onnx::{GraphProto, ModelProto};
use prost::Message;

pub fn minimal_model_bytes() -> Vec<u8> {
    let model = ModelProto {
        graph: Some(GraphProto {
            name: "minimal_model".to_string(),
            ..Default::default()
        }),
        ..Default::default()
    };

    model.encode_to_vec()
}
