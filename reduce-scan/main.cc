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
    auto a = random_vector<float>(n);
    float result = 0, expected_result = 0;
    float solid_truth = 0;
    for (int k = 0; k < n; k++) {
      solid_truth += a[k];
    }

    opencl.queue.flush();
    cl::Kernel kernel(opencl.program, "reduce");
    auto t0 = clock_type::now();
    expected_result = reduce(a);
    auto t1 = clock_type::now();
    Vector<float> v_result(1);
    cl::Buffer d_a(opencl.queue, begin(a), end(a), true);

    opencl.queue.finish();

    auto t2 = clock_type::now();
    int last_vec_size = n;
    cl::Buffer current_vec = d_a;
    for (int vec_size = n/1024; vec_size > 0; vec_size /= 1024) {
      kernel.setArg(0, current_vec);
      cl::Buffer d_result(opencl.context, CL_MEM_READ_WRITE, vec_size*sizeof(float));
      kernel.setArg(1, d_result);
      kernel.setArg(2, vec_size*1024);
      kernel.setArg(3, 1024*sizeof(float), NULL);
      opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(vec_size*1024), cl::NDRange(1024));
      current_vec = d_result;
      last_vec_size = vec_size;
    }

    if (last_vec_size > 1) {
      kernel.setArg(0, current_vec);
      cl::Buffer d_result(opencl.context, CL_MEM_READ_WRITE, sizeof(float));
      kernel.setArg(1, d_result);
      kernel.setArg(2, last_vec_size);
      kernel.setArg(3, last_vec_size*sizeof(float), NULL);
      opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(last_vec_size), cl::NDRange(last_vec_size));
      current_vec = d_result;
    }

    opencl.queue.finish();

    auto t3 = clock_type::now();
    cl::copy(opencl.queue, current_vec, begin(v_result), end(v_result));
    result = v_result[0];
    auto t4 = clock_type::now();
    
    print("reduce",
          {t1-t0,t4-t1,t2-t1,t3-t2,t4-t3},
          {bandwidth(n*n+n+n, t0, t1), bandwidth(n*n+n+n, t2, t3)});
    std::cout << "Result: " << result << "; Expected result: " << expected_result << "; Abs error: " << std::abs(result - expected_result) << "\n";
}

void profile_scan_inclusive(int n, OpenCL& opencl) {
    auto a = random_vector<float>(n);
    Vector<float> result(a), expected_result(a);
    opencl.queue.flush();
    cl::Kernel kernel(opencl.program, "scan_inclusive");
    cl::Kernel kernel_fin(opencl.program, "finish_scan_inclusive");

    auto t0 = clock_type::now();
    scan_inclusive(expected_result);

    auto t1 = clock_type::now();
    cl::Buffer d_a(opencl.queue, begin(a), end(a), false);
    opencl.queue.finish();

    auto t2 = clock_type::now();
    kernel.setArg(0, d_a);
    kernel.setArg(1, 1);
    kernel.setArg(2, 1024*sizeof(float), NULL);
    opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n), cl::NDRange(1024));

    kernel.setArg(0, d_a);
    kernel.setArg(1, 1*1024);
    kernel.setArg(2, 1024*sizeof(float), NULL);
    opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n/1024), cl::NDRange(1024));

    kernel.setArg(0, d_a);
    kernel.setArg(1, 1*1024*1024);
    kernel.setArg(2, 1024*sizeof(float), NULL);
    opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(n/(1024*1024)), cl::NDRange(10));

    kernel_fin.setArg(0, d_a);
    kernel_fin.setArg(1, 1024);
    opencl.queue.enqueueNDRangeKernel(kernel_fin, cl::NullRange, cl::NDRange(n/1024 - 1), cl::NullRange);

    opencl.queue.finish();
    std::cout << "GPU FINISHED" << '\n';
    auto t3 = clock_type::now();
    cl::copy(opencl.queue, d_a, begin(result), end(result));
    auto t4 = clock_type::now();
    for (int k = 1; k < n; k *= 2) {
      std::cout << result[k] << ' ' << expected_result[k] << ' ' << std::abs(result[k] - expected_result[k]) << '\n';
    }
    std::cout << result[1024*1024*10 - 1] << ' ' << expected_result[1024*1024*10 - 1] << ' ' << std::abs(result[1024*1024*10 - 1] - expected_result[1024*1024*10 -1]) << '\n';

    print("scan-inclusive",
          {t1-t0,t4-t1,t2-t1,t3-t2,t4-t3},
          {bandwidth(n*n+n*n+n*n, t0, t1), bandwidth(n*n+n*n+n*n, t2, t3)});
}

void opencl_main(OpenCL& opencl) {
    using namespace std::chrono;
    print_column_names();
    profile_reduce(1024*1024*10, opencl);
    profile_scan_inclusive(1024*1024*10, opencl);
}

const std::string src = R"(
kernel void reduce(global float* a,
                   global float* result,
                   int n,
                   local float* group_part) {
    const int i = get_global_id(0);
    const int local_i = get_local_id(0);
    if (n < 1024) {
      if (i == 0) {
          group_part[0] = 0;
          for (int k = 0; k < n; k++) {
            group_part[0] += a[k];
          }
        }
    } else {
      group_part[local_i] = a[i];
      barrier(CLK_LOCAL_MEM_FENCE);
      for (int offset = get_local_size(0)/2; offset > 0; offset >>= 1) {
         if (local_i < offset) {
            group_part[local_i] += group_part[local_i + offset];
         }
         barrier(CLK_LOCAL_MEM_FENCE);
      }
    }
    if (local_i == 0) result[get_group_id(0)] = group_part[0];
}

kernel void scan_inclusive(global float* a,
                           int a_step,
                           local float* group_part) {
    const int i = get_global_id(0);
    const int local_i = get_local_id(0);
    const int local_size = get_local_size(0);
    group_part[local_i] = a[(a_step-1) + i*a_step];
    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = group_part[local_i];
    for (int offset=1; offset<local_size; offset *= 2) {
      if (local_i >= offset) {
        sum += group_part[local_i - offset];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      group_part[local_i] = sum;
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    a[(a_step-1) + i*a_step] = group_part[local_i];
}

kernel void finish_scan_inclusive(global float*a, int step) {
        const int i = get_global_id(0);
        for (int p = 0; p < step-1; p++) {
          a[(i+1)*step + p] += a[(i+1)*step - 1];
        }
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
