import torch
import torch.nn as nn
import torch.onnx

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Uses Conv → Relu
        self.conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1)
        self.relu = nn.ReLU()

        # This linear layer becomes MatMul + Add in ONNX
        self.fc = nn.Linear(4 * 26 * 26, 10)

        # Softmax
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)      # Conv
        x = self.relu(x)      # Relu
        x = torch.flatten(x, 1)
        x = self.fc(x)        # MatMul + Add
        x = self.softmax(x)   # Softmax
        return x


# Create the model
model = TinyNet()
model.eval()

# Dummy input (batch=1, channels=1, 28x28)
dummy = torch.randn(1, 1, 28, 28)

# Export to ONNX
torch.onnx.export(
    model,
    dummy,
    "out/tinynet.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=13,
)

print("Exported model to tinynet.onnx")