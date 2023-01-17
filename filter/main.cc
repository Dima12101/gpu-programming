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

#include "filter.hh"
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

void print(const char* name, std::array<duration,5> dt) {
    using namespace std::chrono;
    std::cout << std::setw(19) << name;
    for (size_t i=0; i<5; ++i) {
        std::stringstream tmp;
        tmp << duration_cast<microseconds>(dt[i]).count() << "us";
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
    std::cout << '\n';
}

struct OpenCL {
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    cl::CommandQueue queue;
};

void profile_filter(int n, OpenCL& opencl) {
    auto input = random_std_vector<float>(n);
    std::vector<float> result(n), expected_result;
    expected_result.reserve(n);
    opencl.queue.flush();

    cl::Kernel mask_kernel(opencl.program, "mask");
    cl::Kernel scan_kernel(opencl.program, "scan_inclusive");
    cl::Kernel finish_scan_kernel(opencl.program, "finish_scan_inclusive"); //scatter
    cl::Kernel scatter_kernel(opencl.program, "scatter");

    auto t0 = clock_type::now();
    filter(input, expected_result, [] (float x) { return x > 0; }); // filter positive numbers

    auto t1 = clock_type::now();
    cl::Buffer d_input(opencl.queue, begin(input), end(input), true);
    cl::Buffer d_mask(opencl.context, CL_MEM_READ_WRITE, n*sizeof(float));
    cl::Buffer d_out(opencl.context, CL_MEM_READ_WRITE, n*sizeof(float));
    mask_kernel.setArg(0, d_input);
    mask_kernel.setArg(1, d_mask);
    opencl.queue.finish();
    auto t2 = clock_type::now();
    opencl.queue.enqueueNDRangeKernel(mask_kernel, cl::NullRange, cl::NDRange(n), cl::NullRange);

    scan_kernel.setArg(0, d_mask);
    scan_kernel.setArg(1, d_mask);
    scan_kernel.setArg(2, 1);
    opencl.queue.enqueueNDRangeKernel(scan_kernel, cl::NullRange, cl::NDRange(n), cl::NDRange(1024));

    scan_kernel.setArg(0, d_mask);
    scan_kernel.setArg(1, d_mask);
    scan_kernel.setArg(2, 1024);
    opencl.queue.enqueueNDRangeKernel(scan_kernel, cl::NullRange, cl::NDRange(n/1024), cl::NDRange(1024));

    finish_scan_kernel.setArg(0, d_mask);
    finish_scan_kernel.setArg(1, 1024);
    opencl.queue.enqueueNDRangeKernel(finish_scan_kernel, cl::NullRange, cl::NDRange(n/1024-1), cl::NullRange);

    scatter_kernel.setArg(0, d_input);
    scatter_kernel.setArg(1, d_mask);
    scatter_kernel.setArg(2, d_out);
    opencl.queue.enqueueNDRangeKernel(scatter_kernel, cl::NullRange, cl::NDRange(n), cl::NullRange);
    opencl.queue.finish();

    auto t3 = clock_type::now();
    cl::copy(opencl.queue, d_out, begin(result), end(result));

    auto t4 = clock_type::now();
    for (int k = 0; k < n; k++) {
      if (result[k] == 0) {
        result.resize(k);
        break;
      }
    }
    verify_vector(expected_result, result);
    print("filter", {t1-t0,t4-t1,t2-t1,t3-t2,t4-t3});
}

void opencl_main(OpenCL& opencl) {
    using namespace std::chrono;
    print_column_names();
    profile_filter(1024*1024, opencl);
}

const std::string src = R"(
    
kernel void mask(global float* a,
                   global float* result) {
    const int i = get_global_id(0);
    result[i] = convert_float(a[i] > 0);
}

kernel void scan_inclusive(global float* a,
                           global float* result,
                           int a_step) {
    const int i = get_global_id(0);
    const int local_i = get_local_id(0);
    const int local_size = get_local_size(0);
    local float group_part[1024];
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
kernel void scatter(global float* a, global float* scanned, global float* out) {
     const int i = get_global_id(0);
     int current = scanned[i];
     if (current > 0) {
       if (i == 0) {
         out[current - 1] = a[i];
       } else {
         int prev = scanned[i-1];
         if (current != prev) {
             out[current - 1] = a[i];
         }
       }
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
