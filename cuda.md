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

#
