"""
07_cpu_and_gpu_basics.py

Purpose
-------
This script introduces the basic idea of CPU and GPU calculations.

The CPU is the main processor in a computer and is used for most normal
Python calculations.

A GPU can perform many calculations in parallel. GPUs are especially useful
for machine learning and some scientific computing tasks.

PyTorch is used here because it can perform nearly identical calculations
on either the CPU or a compatible GPU.

This script will still run if the computer does not have a GPU.
"""

import time
import torch


# ---------------------------------------------------------------------------
# 1. CHECK WHETHER A GPU IS AVAILABLE
# ---------------------------------------------------------------------------

gpu_available = torch.cuda.is_available()

print("Is a CUDA-compatible GPU available?")
print(gpu_available)

print()


# ---------------------------------------------------------------------------
# 2. CREATE A SMALL TENSOR ON THE CPU
# ---------------------------------------------------------------------------

# A PyTorch tensor is similar to a NumPy array.

cpu_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])

print("CPU tensor:")
print(cpu_tensor)

print("Tensor location:")
print(cpu_tensor.device)

print()


# ---------------------------------------------------------------------------
# 3. PERFORM A SIMPLE CPU CALCULATION
# ---------------------------------------------------------------------------

cpu_result = cpu_tensor * 2.0

print("CPU result:")
print(cpu_result)

print()


# ---------------------------------------------------------------------------
# 4. MOVE DATA TO THE GPU, IF ONE IS AVAILABLE
# ---------------------------------------------------------------------------

if gpu_available:

    # "cuda" tells PyTorch to use an NVIDIA GPU.
    device = torch.device("cuda")

    gpu_tensor = cpu_tensor.to(device)

    print("GPU tensor:")
    print(gpu_tensor)

    print("Tensor location:")
    print(gpu_tensor.device)

    print()

    gpu_result = gpu_tensor * 2.0

    print("GPU result:")
    print(gpu_result)

    print()

    # Move the result back to the CPU.
    returned_result = gpu_result.cpu()

    print("GPU result moved back to CPU:")
    print(returned_result)

else:

    print("No CUDA-compatible GPU was detected.")
    print("The rest of the course code can still use the CPU.")

print()


# ---------------------------------------------------------------------------
# 5. CPU TIMING EXAMPLE
# ---------------------------------------------------------------------------

# Create two matrices.
#
# Matrix multiplication is a common operation in machine learning.

matrix_size = 1000

matrix_a = torch.rand((matrix_size, matrix_size))
matrix_b = torch.rand((matrix_size, matrix_size))

start_time = time.perf_counter()

cpu_product = torch.matmul(matrix_a, matrix_b)

end_time = time.perf_counter()

cpu_time = end_time - start_time

print("CPU matrix multiplication time:")
print(cpu_time, "seconds")

print()


# ---------------------------------------------------------------------------
# 6. GPU TIMING EXAMPLE
# ---------------------------------------------------------------------------

if gpu_available:

    matrix_a_gpu = matrix_a.to("cuda")
    matrix_b_gpu = matrix_b.to("cuda")

    # GPU calculations can run asynchronously.
    # synchronize() makes Python wait so that timing is meaningful.
    torch.cuda.synchronize()

    start_time = time.perf_counter()

    gpu_product = torch.matmul(matrix_a_gpu, matrix_b_gpu)

    torch.cuda.synchronize()

    end_time = time.perf_counter()

    gpu_time = end_time - start_time

    print("GPU matrix multiplication time:")
    print(gpu_time, "seconds")

    print()

    print("Important:")
    print("The GPU is not always faster for very small calculations.")
    print("Moving data between the CPU and GPU also takes time.")

else:

    print("GPU timing example skipped because no CUDA GPU was detected.")

print()


# ---------------------------------------------------------------------------
# PRACTICE EXERCISE
# ---------------------------------------------------------------------------

# Change matrix_size near the middle of this script.
#
# Try:
#
# matrix_size = 100
# matrix_size = 500
# matrix_size = 1000
# matrix_size = 2000
#
# Compare the CPU and GPU times.
#
# Questions to think about:
#
# 1. Is the GPU always faster?
# 2. Does the difference become larger for bigger matrices?
# 3. Why might copying data to the GPU matter?
