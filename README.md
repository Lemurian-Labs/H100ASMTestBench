# H100ASMTestBench

H100ASMTestBench is a CUDA test bench for comparing:

- CUDA math/operator behavior
- custom inline PTX implementations in [complexops/custom_asm.hpp](complexops/custom_asm.hpp)
- PyTorch eager and TorchInductor reference outputs

The repository is focused on float32 operation coverage and instruction-level inspection on NVIDIA GPUs (originally targeting H100-style workflows).

## What Is In This Repo

- [complexops](complexops): Unary, binary, and ternary operation test harnesses plus PTX-backed custom implementations.
- [predicateops](predicateops): Predicate/comparison-oriented tests.
- [convert](convert): Conversion utilities and tests (for example fp32/bf16 conversions).
- [demo](demo): GPU demo executable.
- [include](include): Shared helpers (result formatting, file IO, CUDA check macros, etc.).
- [bin](bin): Python scripts to generate Torch eager/inductor reference binaries.

Key files:

- [complexops/custom_asm.hpp](complexops/custom_asm.hpp): Main custom operation implementation file.
- [complexops/unary_test.cu](complexops/unary_test.cu): Unary operation harness.
- [complexops/binary_test.cu](complexops/binary_test.cu): Binary operation harness.
- [complexops/ternary_test.cu](complexops/ternary_test.cu): Ternary operation harness.

## Prerequisites

- Linux
- NVIDIA driver compatible with your CUDA toolkit
- CUDA toolkit (repo prefers `/usr/local/cuda-13.0/bin/nvcc` when present)
- CMake 3.20+
- Ninja
- Python 3.10+ (for Torch reference scripts)

Install build tools on Ubuntu/Debian if needed:

```bash
sudo apt update
sudo apt install -y cmake ninja-build python3 python3-venv
```

## Python Venv + PyTorch (CUDA)

Create and activate a venv:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install CUDA-enabled PyTorch wheels (CUDA index URL, not ROCm):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

If your environment requires a different CUDA wheel build, switch the index URL accordingly (for example `cu121`).

Quick sanity check:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
```

## Build

From repo root:

```bash
cmake -S . -B build -G Ninja
cmake --build build -j
```

Build specific targets only:

```bash
cmake --build build --target unary_test binary_test ternary_test
```

## Running Tests

Executables are generated under [build](build), for example:

- [build/complexops/unary_test](build/complexops/unary_test)
- [build/complexops/binary_test](build/complexops/binary_test)
- [build/complexops/ternary_test](build/complexops/ternary_test)

### 1) Unary

Show options:

```bash
./build/complexops/unary_test --help
```

Run one op directly:

```bash
./build/complexops/unary_test --exp2f --verbose --quiet --color
```

Compare with Torch eager/inductor:

```bash
./build/complexops/unary_test --dump-inputs unarytest.in
python ./bin/torchunary.py --op exp2f --file unarytest.in
./build/complexops/unary_test --exp2f \
	--torcheager torcheagerexp2f.bin \
	--torchinductor torchinductorexp2f.bin \
	--verbose --quiet --color
```

### 2) Binary

Show options:

```bash
./build/complexops/binary_test --help
```

Run one op directly:

```bash
./build/complexops/binary_test --div --verbose --quiet --color
```

Compare with Torch eager/inductor:

```bash
./build/complexops/binary_test --dump-inputs binarytest.in
python ./bin/torchbinary.py --op div --file binarytest.in
./build/complexops/binary_test --div \
	--torcheager torcheagerdiv.bin \
	--torchinductor torchinductordiv.bin \
	--verbose --quiet --color
```

Supported binary ops include:

- `dim`, `div`, `pow`, `fldiv`, `cldiv`, `root`, `atan2`, `copysign`, `fmax`, `fmin`, `fmod`, `hypot`, `nextafter`, `remainder`

### 3) Ternary

Show options:

```bash
./build/complexops/ternary_test --help
```

Run one op directly:

```bash
./build/complexops/ternary_test --mac --verbose --quiet --color
```

Compare with Torch eager/inductor:

```bash
./build/complexops/ternary_test --dump-inputs ternarytest.in
python ./bin/torchternary.py --op mac --file ternarytest.in
./build/complexops/ternary_test --mac \
	--torcheager torcheagermac.bin \
	--torchinductor torchinductormac.bin \
	--verbose --quiet --color
```

Supported ternary ops:

- `mac`, `clip`, `sel`

## CSV Output Mode

All three harnesses support `--csv`, producing VALUE/HEX/DIFF rows that are useful for scripting and diffing:

```bash
./build/complexops/unary_test --roundf --csv --quiet > roundf.csv
```

## PTX / Assembly Inspection Notes

- Top-level CMake adds `--keep` for CUDA builds, so intermediate files (including PTX) are retained in [build](build).
- Example files often inspected during development:
	- [build/unary_test.ptx](build/unary_test.ptx)
	- [build/binary_test.ptx](build/binary_test.ptx)
	- [build/ternary_test.ptx](build/ternary_test.ptx)

## Troubleshooting

- If CMake picks the wrong CUDA compiler, set it explicitly:

```bash
cmake -S . -B build -G Ninja -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc
```

- If PyTorch cannot see CUDA, verify driver/toolkit compatibility and re-check:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

- The build may emit warning `#550-D` for inline-asm temporaries; this is expected for some register-only helper variables.

