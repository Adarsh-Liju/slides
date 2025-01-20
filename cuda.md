---
class: invert
---

# CPU vs GPU: The Iron Man and Hulk Analogy

In this presentation, we’ll compare **CPU** and **GPU** using **Iron Man** and **Hulk** as analogies.

**Brain:brain: vs Brawn:muscle:**

---

# CPU (Iron Man)

- **Role**: Iron Man represents the **CPU** because he is highly skilled in thinking, strategizing, and making complex decisions.
- **Capabilities**: Iron Man can do many different tasks (like controlling the Iron Man suit, designing technology, etc.), but he does them one at a time.
- **Workload**: The CPU can handle fewer tasks but can perform each one with a lot of variety and precision.

---

# Example: Iron Man's Strategy

- Iron Man thinks deeply and plans one strategy at a time, such as targeting the enemy’s weak spot.
- His strength is in making complex decisions and executing them in sequence.

---

# GPU (Hulk)

- **Role**: Hulk represents the **GPU** because he is fast, powerful, and excels at performing many tasks simultaneously.
- **Capabilities**: Hulk can smash many targets at once, handling parallel tasks efficiently.
- **Workload**: The GPU handles lots of simple tasks in parallel, ideal for operations like graphics rendering and large data processing.

---

# Example: Hulk’s Power

- Hulk can smash many enemies at once without needing to think deeply about each one.
- His strength is in handling multiple repetitive tasks in parallel, like rendering a scene or training a machine learning model.

---

![bg left:40% 80%](./Nvidia_logo.svg)

# **CUDA**

Nvidia's brainchild and flagship technology

---

# CPU vs GPU in Action

- **Iron Man (CPU)**: Handles complex tasks like running the operating system, managing logic, and processing varied software.
- **Hulk (GPU)**: Handles tasks that involve large amounts of simple actions simultaneously, like rendering graphics or processing data.

---

# In a Game Example

- **Iron Man (CPU)**: Manages game logic, character AI, and overall physics of the game.
- **Hulk (GPU)**: Renders graphics and processes visual data, ensuring the game looks smooth and fast.

---

# Conclusion

- **Iron Man (CPU)**: Focuses on complex, sequential tasks.
- **Hulk (GPU)**: Focuses on parallel tasks with raw power.
- Together, they make a powerful team for handling diverse computing tasks!

---

![bg center contain](./CPU.png)

---

![bg center contain](./GPU.png)

---

# CPU vs GPU

| **Feature**           | **CPU**                          | **GPU**                                              |
| --------------------- | -------------------------------- | ---------------------------------------------------- |
| **Purpose**           | General-purpose computation      | Specialized computation for graphics, parallel tasks |
| **Task Handling**     | Single-threaded, complex tasks   | Highly parallel tasks (e.g., graphics rendering)     |
| **Optimization**      | Sequential processing            | Parallel processing                                  |
| **Cache Memory**      | Smaller (L1, L2, L3)             | Larger (VRAM) for high-speed data transfer           |
| **Energy Efficiency** | More efficient for general tasks | Higher power consumption for parallel tasks          |

---

# Definition

- CUDA (Compute Unified Device Architecture) is a parallel computing platform and programming model developed by NVIDIA.
- It allows developers to use NVIDIA GPUs (Graphics Processing Units) for general-purpose computing (GPGPU).
- CUDA provides a way to harness the massive parallelism of GPUs to accelerate computations in various fields, including artificial intelligence, scientific simulations, and real-time graphics.

---

# Why CUDA?

1. **High Performance**: Leverages thousands of GPU cores for parallel processing.
2. **Ease of Use**: Extends C/C++ with simple keywords and APIs.
3. **Massive Parallelism**: Executes many tasks simultaneously.
4. **Optimized Libraries**: Offers libraries like cuBLAS (linear algebra), cuDNN (deep learning), and Thrust (high-level algorithms).

---

# CUDA Programming Model

- Host (CPU): Executes the main program.
- Device (GPU): Executes parallel computations.
- Kernels: Functions executed on the GPU in parallel.
- Threads & Blocks: CUDA organizes parallel execution in a grid of blocks, and each block contains multiple threads.

---

# Example Code without CUDA

```c
#include <iostream>
#include <vector>
const int N = 1000000; // Size of the arrays
// Function to add two arrays
void add(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}
int main() {
    std::vector<int> a(N), b(N), c(N);
    // Initialize arrays
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }
    // Perform addition
    add(a, b, c);
    // Print a few results
    for (int i = 0; i < c.size(); i++) {
        std::cout << "c[" << i << "] = " << c[i] << std::endl;
    }
    return 0;
}
```

---

# Explanation of the Code

- Declare Arrays: Create three arrays a, b, and c of size N.
- Initialize Arrays:
  - `a[i] = i`
  - `b[i] = i \* 2`
- Call add Function: Pass arrays a, b, and c to the add function.
- In add Function:
  - Loop from `0 to N-1`
  - Add `a[i]` and `b[i]` and store the result in `c[i]`.
- Print Results: Display the first 10 elements of c.
- End Program: Program finishes execution.

---

# Example Code using CUDA

```c
#include <iostream>
#include <cuda_runtime.h>
const int N = 1024; // Size of the arrays
// CUDA kernel function to add two arrays
__global__ void add(int *a, int *b, int *c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}
int main() {
    int *a, *b, *c;  // Host pointers
    int *d_a, *d_b, *d_c;  // Device pointers
    size_t size = N * sizeof(int);
    // Allocate host memory
    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);
    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }
    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    // Copy data from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
    // Launch the kernel with one block of 256 threads
    add<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c);
    // Copy result from device to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
    // Print a few results
    for (int i = 0; i < 10; i++) {
        std::cout << "c[" << i << "] = " << c[i] << std::endl;
    }
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    // Free host memory
    free(a);
    free(b);
    free(c);
    return 0;
}
```

---

## **Step 1: CUDA Program Overview**

### **Objective:**

- Perform **parallel addition** of two arrays using CUDA.
- Utilize **GPU acceleration** for faster computation.

### **Key Components:**

✅ **Host (CPU) Operations** – Memory allocation, data transfer.  
✅ **Device (GPU) Execution** – Parallel computation with CUDA kernels.  
✅ **Data Transfer & Cleanup** – Copy results back to CPU and free memory.

---

## **Step 2: Host (CPU) Operations**

### **1. Memory Allocation & Initialization**

✔ Allocate memory for arrays `a`, `b`, and `c`.  
✔ Initialize arrays:

- `a[i] = i`
- `b[i] = i * 2`

### **2. Device Memory Allocation**

✔ Allocate GPU memory (`d_a`, `d_b`, `d_c`).

### **3. Copy Data to Device**

✔ Use `cudaMemcpy()` to transfer `a` and `b` from **CPU → GPU**.

---

## **Step 3: CUDA Kernel Execution (GPU)**

### **1. Kernel Launch**

🚀 **Launch configuration:**

```cpp
add<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c);
```

✔ `(N + 255) / 256` → Computes required **blocks**.  
✔ `256` → Number of **threads per block**.

---

### **2. Parallel Computation**

✔ **Each thread** executes:

```cpp
c[idx] = a[idx] + b[idx];
```

✔ Index calculation:

```cpp
int idx = threadIdx.x + blockIdx.x * blockDim.x;
```

✔ Threads operate **independently** for speedup.

---

## **Step 4: CUDA Grid & Block Structure**

### **Grid & Block Breakdown:**

- **Grid Size**: `(N + 255) / 256` blocks
- **Block Size**: `256` threads

| **Block** | **Threads**    |
| --------- | -------------- |
| Block 0   | `[0 - 255]`    |
| Block 1   | `[256 - 511]`  |
| Block 2   | `[512 - 767]`  |
| Block 3   | `[768 - 1023]` |

✔ **Scalable Design** – Works for large `N`.

---

## **Step 5: Copy Back & Cleanup**

### **1. Transfer Results to Host (CPU ← GPU)**

✔ `cudaMemcpy()` moves `c` from **GPU → CPU**.

### **2. Display Results**

✔ Print the first **10 values of `c`**.

### **3. Free Memory**

✔ Deallocate memory:

```cpp
cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
free(a); free(b); free(c);
```

---

## **Step 6: Why is CUDA Efficient?**

### **Performance Benefits:**

✅ **Massive Parallelism** – Thousands of threads run **simultaneously**.  
✅ **Optimized Memory Transfers** – Reduces CPU-GPU communication overhead.  
✅ **Scalability** – Can handle large datasets efficiently.  
✅ **Minimal Code Changes** – Works across different GPU architectures.

---

### **Summary & Takeaways**

📌 **CUDA enables parallel computing with GPU acceleration.**  
📌 **Each thread independently computes a portion of the data.**  
📌 **CUDA’s grid & block model optimizes performance.**  
📌 **Efficient memory management ensures speedup.**

🔥 **Key Learning:** GPUs can dramatically **speed up computations** by executing thousands of threads in parallel! 🚀

---

## **Compilation and Execution**

### **For CPU Version**

```bash
g++ -o cpu_addition cpu_addition.cpp -O2
./cpu_addition
```

### **For CUDA Version**

```bash
nvcc -o cuda_addition cuda_addition.cu
./cuda_addition
```

---

## **Performance Expectations**

- **CPU Version**: Runs in a single thread, slower than GPU but still optimized.
- **CUDA Version**: Uses GPU parallelism, which is significantly faster for large arrays.

---

## **Conclusion**

- The CUDA version should be **faster** due to GPU parallelism.
- The CPU version is easier to use and doesn’t require a dedicated GPU.
- CUDA requires data transfers between CPU and GPU, which can add overhead.
