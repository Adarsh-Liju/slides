#include <iostream>
#include <vector>
#include <chrono>
#include <sys/resource.h>

const int N = 1000000; // Size of the arrays

// Function to add two arrays on CPU
void add_cpu(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& c) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
}

// Function to print resource usage
void print_resource_usage() {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        std::cout << "User CPU time used: " << usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6 << " seconds\n";
        std::cout << "System CPU time used: " << usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6 << " seconds\n";
        std::cout << "Maximum resident set size: " << usage.ru_maxrss << " KB\n";
    } else {
        std::cerr << "Failed to get resource usage.\n";
    }
}

int main() {
    std::vector<int> a(N), b(N), c(N);

    // Initialize arrays
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Measure time
    auto start = std::chrono::high_resolution_clock::now();
    add_cpu(a, b, c);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Print results
    std::cout << "CPU Time taken for addition: " << duration.count() << " seconds\n";
    print_resource_usage();

    return 0;
}

