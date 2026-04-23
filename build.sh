#!/usr/bin/env bash
# define mode
MODE="file"

# Compile onnx file 
cargo run --bin nnc -- --model ./out/resnet_model.onnx --weights external --mode $MODE

# Compile model
if [ "$MODE" == "lib" ]; then
    # compile as a dynamic library
    gcc -shared -o out/nn.so out/nn.c -lm
else
    # compile as a static library
    gcc -o out/nn out/nn.c -lm

		# run 
		cd ./out
		./nn ./input.bin ./output.bin
		cd -
fi

