Yes, you can run performance benchmarks on Docker containers to compare the resource consumption, speed, and efficiency of different container images (like Alpine vs Ubuntu) under similar workloads. Here’s a step-by-step guide to running some common performance benchmarks in Docker containers:

### 1. **System Information and Resource Monitoring**

First, ensure you have tools like `top`, `htop`, or `docker stats` available to monitor resource usage (CPU, memory, disk I/O).

- **Using Docker Stats**:
  You can use `docker stats` to monitor the performance of containers in real-time:

  ```bash
  docker stats
  ```

  This will show metrics for each running container, including CPU, memory, network, and disk usage.

### 2. **CPU Benchmark with `sysbench`**

To benchmark CPU performance, you can use a tool like `sysbench`. This tool can run various tests, including CPU stress tests.

#### Alpine Example:

1. **Run Alpine Container**:
   Start an Alpine container and install `sysbench`:

   ```bash
   docker run -it alpine /bin/sh
   apk add --no-cache sysbench
   ```

2. **Run CPU Benchmark**:
   Run a CPU performance benchmark (e.g., 10 seconds of CPU intensive task):

   ```bash
   sysbench --test=cpu --cpu-max-prime=20000 run
   ```

   This will run a benchmark on the CPU, calculating prime numbers up to 20,000, and print the results like execution time, events per second, and other CPU stats.

#### Ubuntu Example:

1. **Run Ubuntu Container**:
   Start an Ubuntu container and install `sysbench`:

   ```bash
   docker run -it ubuntu /bin/bash
   apt-get update
   apt-get install -y sysbench
   ```

2. **Run CPU Benchmark**:
   Run the same `sysbench` CPU test:

   ```bash
   sysbench --test=cpu --cpu-max-prime=20000 run
   ```

### 3. **Memory Benchmark with `sysbench`**

You can also test memory performance using `sysbench`. Here’s how to run a memory benchmark inside the containers:

#### Alpine Example:

1. **Run Memory Benchmark**:

   ```bash
   sysbench --test=memory --memory-total-size=10G run
   ```

   This will benchmark memory access by allocating 10 GB of memory and performing memory read/write operations.

#### Ubuntu Example:

1. **Run Memory Benchmark**:

   ```bash
   sysbench --test=memory --memory-total-size=10G run
   ```

### 4. **Disk I/O Benchmark with `fio`**

Disk I/O performance can also be benchmarked with `fio`, a flexible I/O tester and benchmark tool. You can use `fio` to test random read/write, sequential read/write, etc.

#### Alpine Example:

1. **Run Alpine Container** and Install `fio`:

   ```bash
   docker run -it alpine /bin/sh
   apk add --no-cache fio
   ```

2. **Run Disk Benchmark**:

   ```bash
   fio --name=mytest --ioengine=sync --size=1G --readwrite=randwrite --numjobs=1 --time_based --runtime=30s
   ```

   This will run a random write test with 1 job on a 1 GB file for 30 seconds.

#### Ubuntu Example:

1. **Run Ubuntu Container** and Install `fio`:

   ```bash
   docker run -it ubuntu /bin/bash
   apt-get update
   apt-get install -y fio
   ```

2. **Run Disk Benchmark**:

   ```bash
   fio --name=mytest --ioengine=sync --size=1G --readwrite=randwrite --numjobs=1 --time_based --runtime=30s
   ```

### 5. **Network Benchmark with `iperf3`**

You can benchmark network throughput using `iperf3`. This requires a server-client setup where one container acts as the server and the other as the client.

#### Alpine Example (Server):

1. **Start Alpine Container as Server**:

   ```bash
   docker run -d --name iperf-server alpine /bin/sh -c "apk add --no-cache iperf3 && iperf3 -s"
   ```

#### Ubuntu Example (Client):

1. **Run Ubuntu Container as Client**:

   ```bash
   docker run --rm ubuntu /bin/bash -c "apt-get update && apt-get install -y iperf3 && iperf3 -c iperf-server"
   ```

This will test the network speed between the two containers (client and server). You can change the `iperf3` parameters to test different aspects of network performance (e.g., bandwidth, jitter, latency).

### 6. **Comparison of Results**

After running the benchmarks, you can compare:

- **CPU performance**: Check execution time and operations per second.
- **Memory performance**: Compare read/write speeds and throughput.
- **Disk I/O performance**: Evaluate read/write speeds and I/O latency.
- **Network performance**: Look at network throughput between containers.

### Conclusion

These benchmarks can give you an idea of how different container images perform under similar workloads. You might find that Alpine is faster for smaller tasks, while Ubuntu might be better suited for certain use cases where compatibility and ease of use are more important.

However, keep in mind that benchmarks can vary based on the nature of your workload, so it’s important to test with your actual use case.
