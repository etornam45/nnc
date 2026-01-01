#!/usr/bin/env bash

# Compile onnx file 
cargo run -- --weights external --mode dummy

# Compile model
gcc out/nn.c -o out/nn -lm

# # run 
cd ./out
./nn
cd -