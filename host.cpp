#include "xcl2.hpp"
#include "event_timer.hpp"
#include <vector>
#include <iostream>
#include <cstdlib>

#define lm 4
#define ln 4
#define lp 4

#define m (1 << lm)
#define n (1 << ln)
#define p (1 << lp)

typedef unsigned int input_type;
typedef unsigned int result_type;

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }

    EventTimer et;
    std::string binaryFile = argv[1];
    size_t size_A = n * m * sizeof(input_type);
    size_t size_B = m * p * sizeof(input_type);
    size_t size_C = n * p * sizeof(result_type);

    cl_int err;
    cl::Context context;
    cl::Kernel krn_matmul;
    cl::CommandQueue q;

    // Allocate Host Memory
    et.add("Allocate Memory in Host Memory");
    std::vector<input_type, aligned_allocator<input_type>> A(n * m);
    std::vector<input_type, aligned_allocator<input_type>> B(m * p);
    std::vector<result_type, aligned_allocator<result_type>> C_hw(n * p);
    std::vector<result_type, aligned_allocator<result_type>> C_sw(n * p);
    et.finish();

    // Initialize Matrices
    et.add("Fill the matrices with random values");
    for (int i = 0; i < n * m; i++) A[i] = static_cast<input_type>(std::rand());
    for (int i = 0; i < m * p; i++) B[i] = static_cast<input_type>(std::rand());
    std::fill(C_hw.begin(), C_hw.end(), 0);

    // Compute Software Matrix Multiplication
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            result_type sum = 0;
            for (int k = 0; k < m; k++) {
                sum += A[i * m + k] * B[k * p + j];
            }
            C_sw[i * p + j] = sum;
        }
    }
    et.finish();

    // Load Binary File
    et.add("Load Binary File to Alveo U200");
    auto devices = xcl::get_xil_devices();
    auto fileBuf = xcl::read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    int valid_device = 0;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        OCL_CHECK(err, context = cl::Context(device, NULL, NULL, NULL, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        cl::Program program(context, {device}, bins, NULL, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krn_matmul = cl::Kernel(program, "mult_hw", &err));
            valid_device++;
            break;
        }
    }
    if (valid_device == 0) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }
    et.finish();

    // Allocate Buffers
    et.add("Allocate Buffer in Global Memory");
    OCL_CHECK(err, cl::Buffer buffer_A(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, size_A, A.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_B(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, size_B, B.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_C(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, size_C, C_hw.data(), &err));
    et.finish();

    // Set Kernel Arguments
    et.add("Set Kernel Arguments");
    OCL_CHECK(err, err = krn_matmul.setArg(0, buffer_A));
    OCL_CHECK(err, err = krn_matmul.setArg(1, buffer_B));
    OCL_CHECK(err, err = krn_matmul.setArg(2, buffer_C));
    et.finish();

    // Transfer Input Data
    et.add("Copy Input Data to Device");
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_A, buffer_B}, 0));
    et.finish();

    // Launch Kernel
    et.add("Launch Kernel");
    OCL_CHECK(err, err = q.enqueueTask(krn_matmul));
    et.finish();

    // Transfer Results Back
    et.add("Copy Results to Host");
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_C}, CL_MIGRATE_MEM_OBJECT_HOST));
    OCL_CHECK(err, err = q.finish());
    et.finish();

    // Validate Results
    et.add("Validate Results");
    bool match = true;
    for (int i = 0; i < n * p; i++) {
        if (C_hw[i] != C_sw[i]) {
            std::cout << "Mismatch at index " << i << ": HW=" << C_hw[i] << " SW=" << C_sw[i] << std::endl;
            match = false;
            break;
        }
    }
    et.finish();

    std::cout << "----------------- Key Execution Times -----------------" << std::endl;
    et.print();
    std::cout << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;

    return match ? EXIT_SUCCESS : EXIT_FAILURE;
}
