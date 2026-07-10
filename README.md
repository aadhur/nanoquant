# nanoquant v1
An inference and quantization engine in C++ from scratch (Training in PyTorch)

## File Structure 
1. **Libraries:**
   * `tensor.h` - Tensors struct (tensor and qtensor) and utility functions
   * `ops.h` - Operations related to inference
   * `quant.h` - Quantization functions (inc. matmul w/ quantizing)
2. **/test_files**
   * `test_tensor.cpp` - Unit tests for tensor func
   * `test_infer.cpp` - Benchmarks standard, unquantized raw inference
   * `test_quant.cpp` - Main script w/ inference + quantization
   * `train.py` - Training in PyTorch to generate the weight/bias bin files + calibration images

## Project Status & Scope

### What it does right now:
* Built for the MNIST dataset - static layout.
* Quantization happens on matrix multiplication, with bias added separately.
* Execution time dropped by **94%** compared to the traditional, unquantized C++ setup.

### What it doesn't do yet (Future stuff):
* **No Dynamic Models**: Needs ONNX integration so it doesn't rely on a hardcoded structure.
* **No SIMD**: Needs hardware acceleration like AVX2 or NEON vectorization.
* **No Production Backend**: Needs an HTTP server with request batching and multi-threaded inference.