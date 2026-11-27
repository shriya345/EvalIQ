#!/bin/sh
set -eu
: "${LLAMA_SOURCE:?Set LLAMA_SOURCE to the llama.cpp b5430 source directory}"
: "${LLAMA_LIB:?Set LLAMA_LIB to the built CPU library directory}"
c++ -std=c++17 -I"$LLAMA_SOURCE/include" -I"$LLAMA_SOURCE/ggml/include" "$LLAMA_SOURCE/examples/simple/simple.cpp" -L"$LLAMA_LIB" -Wl,-rpath,"$LLAMA_LIB" -lllama -lggml -lggml-base -o "${1:-/tmp/evaliq-llama-simple}"
