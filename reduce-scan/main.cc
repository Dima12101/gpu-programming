#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl2.hpp>
#endif

#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>

#include "linear-algebra.hh"
#include "reduce-scan.hh"

using clock_type = std::chrono::high_resolution_clock;
using duration = clock_type::duration;
using time_point = clock_type::time_point;

double bandwidth(int n, time_point t0, time_point t1) {
    using namespace std::chrono;
    const auto dt = duration_cast<microseconds>(t1-t0).count();
    if (dt == 0) { return 0; }
    return ((n+n+n)*sizeof(float)*1e-9)/(dt*1e-6);
}

void print(const char* name, std::array<duration,5> dt, std::array<double,2> bw) {
    using namespace std::chrono;
    std::cout << std::setw(19) << name;
    for (size_t i=0; i<5; ++i) {
        std::stringstream tmp;
        tmp << duration_cast<microseconds>(dt[i]).count() << " us";
        std::cout << std::setw(20) << tmp.str();
    }
    for (size_t i=0; i<2; ++i) {
        std::stringstream tmp;
        tmp << bw[i] << " GB/s";
        std::cout << std::setw(20) << tmp.str();
    }
    std::cout << '\n';
}

void print_column_names() {
    std::cout << std::setw(19) << "function";
    std::cout << std::setw(20) << "OpenMP";
    std::cout << std::setw(20) << "OpenCL total";
    std::cout << std::setw(20) << "OpenCL copy-in";
    std::cout << std::setw(20) << "OpenCL kernel";
    std::cout << std::setw(20) << "OpenCL copy-out";
    std::cout << std::setw(20) << "OpenMP bandwidth";
    std::cout << std::setw(20) << "OpenCL bandwidth";
    std::cout << '\n';
}

struct OpenCL {
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    cl::CommandQueue queue;
};

void profile_reduce(int n, OpenCL& opencl) {
    auto a = random_vector(n);
    int result = 0, expected_result = 0;
    opencl.queue.flush();
    cl::Kernel kernel(opencl.program, "reduce");
    auto t0 = clock_type::now();
    expected_result = reduce(a);
    auto t1 = clock_type::now();
    cl::Buffer d_a(opencl.queue, std::begin(a), std::end(a), true);
    cl::Buffer d_result(opencl.context, CL_MEM_READ_WRITE, sizeof(int));
    kernel.setArg(0, d_a);
    kernel.setArg(1, d_result);
    auto t2 = clock_type::now();
    opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n / 10), cl::NDRange(256));
    opencl.queue.flush();
    auto t3 = clock_type::now();
    cl::copy(opencl.queue, d_result, &result, &result + 1);
    auto t4 = clock_type::now();
    // verify_vector({expected_result}, {result});
    print("reduce",
          {t1-t0,t4-t1,t2-t1,t3-t2,t4-t3},
          {bandwidth(n*n+n+n, t0, t1), bandwidth(n*n+n+n, t2, t3)});
    std::cout << "Result: " << result << "; Expected result: " << expected_result << "; Abs error: " << std::abs(result - expected_result) << "\n";
}

duration kernel_time = duration::zero();

#define GROUP_SIZE 64

void rec_scan(OpenCL& opencl, cl::Buffer& d_a, int n) {
    cl::Kernel kernel_scan(opencl.program, "scan_all");
    cl::Kernel kernel_assign(opencl.program, "scan_assign");
    cl::Buffer d_blocks(opencl.context, CL_MEM_READ_WRITE, (n + GROUP_SIZE - 1) / GROUP_SIZE * sizeof(int));
    kernel_scan.setArg(0, d_a);
    kernel_scan.setArg(1, d_a);
    kernel_scan.setArg(2, d_blocks);
    kernel_assign.setArg(0, d_a);
    kernel_assign.setArg(1, d_blocks);
    cl::Event ev_scan, ev_assign;
    int SCAN_SIZE = (n < GROUP_SIZE) ? n : GROUP_SIZE;
    auto t0 = clock_type::now();
    opencl.queue.enqueueNDRangeKernel(kernel_scan, cl::NullRange, cl::NDRange(n), cl::NDRange(SCAN_SIZE), NULL, &ev_scan);
    ev_scan.wait();
    auto t1 = clock_type::now();
    kernel_time += t1 - t0;
    if (n <= GROUP_SIZE)
    return;
    rec_scan(opencl, d_blocks, n / GROUP_SIZE);
    auto t2 = clock_type::now();
    opencl.queue.enqueueNDRangeKernel(kernel_assign, cl::NullRange, cl::NDRange(n), cl::NDRange(GROUP_SIZE), NULL, &ev_assign);
    ev_assign.wait();
    auto t3 = clock_type::now();
    kernel_time += t3 - t2;
}
 
 
void profile_scan_exclusive(int n, OpenCL& opencl) {
    auto a = random_vector(n);
    Vector<int> result(a), expected_result(a);
    opencl.queue.flush();
    auto t0 = clock_type::now();
    scan_exclusive(expected_result);
    auto t1 = clock_type::now();
    cl::Buffer d_a(opencl.queue, std::begin(a), std::end(a), true);
    auto t2 = clock_type::now();
    rec_scan(opencl, d_a, n);
    auto t3 = clock_type::now();
    opencl.queue.flush();
    cl::copy(opencl.queue, d_a, begin(result), end(result));
    auto t4 = clock_type::now();
    // verify_vector(expected_result, result);
    print("recursive-scan",
          {t1-t0,t4-t1,t2-t1,kernel_time,t4-t3},
          {bandwidth(n*n+n*n+n*n, t0, t1), bandwidth(n*n+n*n+n*n, t2, t3)});
    for (int k = 1; k < n; k *= 2) {
      std::cout << result[k] << ' ' << expected_result[k] << ' ' << std::abs(result[k] - expected_result[k]) << '\n';
    }
    std::cout << result[1024*1024*10 - 1] << ' ' << expected_result[1024*1024*10 - 1] << ' ' << std::abs(result[1024*1024*10 - 1] - expected_result[1024*1024*10 -1]) << '\n';
}

void opencl_main(OpenCL& opencl) {
    using namespace std::chrono;
    print_column_names();
    profile_reduce(1024*1024*10, opencl);
    profile_scan_exclusive(1024*1024*10, opencl);
}

const std::string src = R"(
 
#define K 10
#define GROUP_SIZE_REDUCE 256
#define GROUP_SIZE_SCAN 64
 
 
kernel void reduce(global int* a,
           global int* ans) {
    const int loc_id = get_local_id(0);
    const int glob_id = get_global_id(0);
    local int precalc[GROUP_SIZE_REDUCE];
    precalc[loc_id] = 0;
    for(int i = glob_id * K; i < (glob_id + 1) * K; ++i)
    precalc[loc_id] += a[i];
 
    barrier(CLK_LOCAL_MEM_FENCE);
 
    for(int n = GROUP_SIZE_REDUCE; n > 1; n >>= 1) {
    if (2 * loc_id < n) {
        precalc[loc_id] += precalc[loc_id + n / 2];
        }
    barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    if (loc_id == 0) {
    atom_add(ans, precalc[0]);
    }
}
 
//Scan kernels:
kernel void scan_all(global int* a,
                     global int* result,
             global int* blocks) {
    const int loc_id = get_local_id(0);
    const int glob_id = get_global_id(0);
    const int group_id = get_group_id(0);
 
    local int data[GROUP_SIZE_SCAN];
    data[loc_id] = a[glob_id];
 
    int cur_val = data[loc_id];
    int sum = data[loc_id];
 
    barrier(CLK_LOCAL_MEM_FENCE);
 
    for (int offset = 1; offset < GROUP_SIZE_SCAN; offset <<= 1) {
    if (loc_id >= offset) {
        sum += data[loc_id - offset];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    data[loc_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    }
 
    result[glob_id] = data[loc_id] - cur_val;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (loc_id == 0) {
    blocks[group_id] = data[GROUP_SIZE_SCAN - 1];
    }
}
 
kernel void scan_assign(global int* result,
            global int* blocks) {
    const int glob_id = get_global_id(0);
    const int group_id = get_group_id(0);
    result[glob_id] += blocks[group_id];
}
 
)";
 
int main() {
    try {
        // find OpenCL platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            std::cerr << "Unable to find OpenCL platforms\n";
            return 1;
        }
        cl::Platform platform = platforms[0];
        std::clog << "Platform name: " << platform.getInfo<CL_PLATFORM_NAME>() << '\n';
        // create context
        cl_context_properties properties[] =
            { CL_CONTEXT_PLATFORM, (cl_context_properties)platform(), 0};
        cl::Context context(CL_DEVICE_TYPE_GPU, properties);
        // get all devices associated with the context
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        cl::Device device = devices[0];
        std::clog << "Device name: " << device.getInfo<CL_DEVICE_NAME>() << '\n';
        cl::Program program(context, src);
        // compile the programme
        try {
            program.build(devices);
        } catch (const cl::Error& err) {
            for (const auto& device : devices) {
                std::string log = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
                std::cerr << log;
            }
            throw;
        }
        cl::CommandQueue queue(context, device);
        OpenCL opencl{platform, device, context, program, queue};
        opencl_main(opencl);
    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error in " << err.what() << '(' << err.err() << ")\n";
        std::cerr << "Search cl.h file for error code (" << err.err()
            << ") to understand what it means:\n";
        std::cerr << "https://github.com/KhronosGroup/OpenCL-Headers/blob/master/CL/cl.h\n";
        return 1;
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        return 1;
    }
    return 0;
}