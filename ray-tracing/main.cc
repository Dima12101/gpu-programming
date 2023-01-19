#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl2.hpp>
#endif

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "color.hh"
#include "ray.hh"
#include "scene.hh"
#include "theora.hh"
#include "vector.hh"
#include "random.hh"

using uniform_distribution = std::uniform_real_distribution<float>;
using color = Color<float>;
using object_ptr = std::unique_ptr<Object>;
using clock_type = std::chrono::high_resolution_clock;
using duration = clock_type::duration;
using time_point = clock_type::time_point;

struct OpenCL {
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    cl::CommandQueue queue;
};

vec trace(ray r, const Object_group& objects) {
    float factor = 1;
    const int max_depth = 50;
    int depth=0;
    for (; depth<max_depth; ++depth) {
        if (Hit hit = objects.hit(r, 1e-3f, std::numeric_limits<float>::max())) {
            r = ray(hit.point, hit.normal + random_in_unit_sphere()); // scatter
            factor *= 0.5f; // diffuse 50% of light, scatter the remaining
        } else {
            break;
        }
    }
    //if (depth == max_depth) { return vec{}; }
    // nothing was hit
    // represent sky as linear gradient in Y dimension
    float t = 0.5f*(unit(r.direction())(1) + 1.0f);
    return factor*((1.0f-t)*vec(1.0f, 1.0f, 1.0f) + t*vec(0.5f, 0.7f, 1.0f));
}

void print_column_names(const char* version) {
    std::cout << std::setw(20) << "Time step";
    std::cout << std::setw(20) << "No. of steps";
    std::cout << std::setw(20) << version << " time";
    std::cout << '\n';
}

void ray_tracing_cpu() {
    using std::chrono::duration_cast;
    using std::chrono::seconds;
    using std::chrono::microseconds;
    int nx = 800, ny = 600, nrays = 100;
    
    Pixel_matrix<float> pixels(nx,ny);
    thx::screen_recorder recorder("out.ogv", nx,ny);
    Object_group objects;
    objects.add(object_ptr(new Sphere(vec(0.f,0.f,-1.f),0.5f)));
    objects.add(object_ptr(new Sphere(vec(0.f,-1000.5f,-1.f),1000.f)));
    Camera camera;
    uniform_distribution distribution(0.f,1.f);
    float gamma = 2;
    const int max_time_step = 60;
    print_column_names("OpenMP");
    duration total_time = duration::zero();
    for (int time_step=1; time_step<=max_time_step; ++time_step) {
        auto t0 = clock_type::now();
        #pragma omp parallel for collapse(2) schedule(dynamic,1)
        for (int j=0; j<ny; ++j) {
            for (int i=0; i<nx; ++i) {
                vec sum;
                for (int k=0; k<nrays; ++k) {
                    float u = (i + distribution(prng)) / nx;
                    float v = (j + distribution(prng)) / ny;
                    sum += trace(camera.make_ray(u,v),objects);
                }
                sum /= float(nrays); // antialiasing
                sum = pow(sum,1.f/gamma); // gamma correction
                pixels(i,j) = to_color(sum);
            }
        }
        auto t1 = clock_type::now();
        const auto dt = duration_cast<microseconds>(t1-t0);
        total_time += dt;
        std::clog
            << std::setw(20) << time_step
            << std::setw(20) << max_time_step
            << std::setw(20) << dt.count()
            << std::endl;
        std::ofstream out("out.ppm");
        out << pixels;
        recorder.record_frame(pixels);
        camera.move(vec{0.f,0.f,0.1f});
    }
    std::clog << "Ray-tracing time: " << duration_cast<seconds>(total_time).count()
        << "s" << std::endl;
    std::clog << "Movie time: " << max_time_step/60.f << "s" << std::endl;
}

void ray_tracing_gpu(OpenCL& opencl) {

    opencl.queue.flush();
    using std::chrono::duration_cast;
    using std::chrono::seconds;
    using std::chrono::microseconds;
    int nx = 800, ny = 600, nrays = 100;
    Pixel_matrix<float> pixels(nx,ny);
    thx::screen_recorder recorder("out.ogv", nx,ny);
    std::vector<Sphere> objects = {
        Sphere{vec(0.f,0.f,-1.f),0.5f},
        Sphere{vec(0.f,-1000.5f,-1.f),1000.f}
    };

    uniform_distribution distribution(0.f,1.f);
    float gamma = 2;
    const int max_time_step = 60;
    print_column_names("OpenCL");

    float camera_origin[4] = {13.f,2.f,3.f, 0};
    float camera_move_direction[4] = {0.f,0.f,0.1f, 0};
    float camera_ll_corner[4] = {3.02374f,-1.22628f,3.4122f, 0};
    float camera_horizontal[4] = {1.18946f,0.f,-5.15434f, 0};
    float camera_vertical[4] = {-0.509421f,3.48757f,-0.117559f, 0};

    std::vector<float> objects_vec;
    for (int i = 0; i < objects.size(); i++) {
        objects_vec.push_back(objects[i].origin()(0));
        objects_vec.push_back(objects[i].origin()(1));
        objects_vec.push_back(objects[i].origin()(2));
        objects_vec.push_back(objects[i].radius());
    }

    cl::Buffer d_camera_origin(opencl.queue, camera_origin, camera_origin + 4, false);
    cl::Buffer d_camera_move_direction(opencl.queue, camera_move_direction, camera_move_direction + 4, true);
    cl::Buffer d_camera_ll_corner(opencl.queue, camera_ll_corner, camera_ll_corner+4, true);
    cl::Buffer d_camera_horizontal(opencl.queue, camera_horizontal, camera_horizontal+4, true);
    cl::Buffer d_camera_vertical(opencl.queue, camera_vertical, camera_vertical+4, true);
    cl::Buffer d_objects(opencl.queue, std::begin(objects_vec), std::end(objects_vec), true);
    cl::Buffer d_result(opencl.context, CL_MEM_READ_WRITE, nx*ny*3*sizeof(float));

    std::normal_distribution<float> dist(0.f,1.f);
    int distr_size = 1<<24;
    std::vector<float> distr;
    //distr.reserve(distr_size);

    //#pragma omp parallel for
    for (int i = 0; i < distr_size; i++) {
        distr.push_back(dist(prng));
    }

    cl::Buffer d_distr(opencl.queue, begin(distr), end(distr), true);

    cl::Kernel kernel(opencl.program, "ray_trace");
    cl::Kernel move_camera_kernel(opencl.program, "move_camera");
    move_camera_kernel.setArg(0, d_camera_origin);
    move_camera_kernel.setArg(1, d_camera_move_direction);

    kernel.setArg(0, d_camera_origin);
    kernel.setArg(1, d_camera_ll_corner);
    kernel.setArg(2, d_camera_horizontal);
    kernel.setArg(3, d_camera_vertical);
    kernel.setArg(4, d_objects);
    kernel.setArg(5, 2); // count of objects
    kernel.setArg(6, d_distr);
    kernel.setArg(7, distr_size);
    kernel.setArg(8, d_result);
    kernel.setArg(9, ny);
    kernel.setArg(10, nx);
    kernel.setArg(11, nrays);
    kernel.setArg(12, gamma);

    opencl.queue.flush(); 

    duration total_time = duration::zero();
    for (int time_step=1; time_step<=max_time_step; ++time_step) {
        auto t0 = clock_type::now();

        opencl.queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(ny, nx), cl::NullRange);
        opencl.queue.flush();
        // move camera a bit
        opencl.queue.enqueueNDRangeKernel(move_camera_kernel, cl::NullRange, cl::NDRange(1), cl::NullRange);
        opencl.queue.flush();

        opencl.queue.enqueueReadBuffer(d_result, true, 0, 3*nx*ny*sizeof(float), (float*)(pixels.pixels().data()));
        opencl.queue.finish();
        auto t1 = clock_type::now();
        const auto dt = duration_cast<microseconds>(t1-t0);
        total_time += dt;
        std::clog
            << std::setw(20) << time_step
            << std::setw(20) << max_time_step
            << std::setw(20) << dt.count()
            << std::endl;
        std::ofstream out("out.ppm");
        out << pixels;
        recorder.record_frame(pixels);
    }
    std::clog << "Ray-tracing time: " << duration_cast<seconds>(total_time).count()
        << "s" << std::endl;
    std::clog << "Movie time: " << max_time_step/60.f << "s" << std::endl;
}

const std::string kernelsmykernels = R"(
struct Ray
	{
	float3 origin;
    float3 direction;
	};
typedef struct Ray Ray;

struct Hit {
    float t;
    float3 point;
    float3 normal;
};
typedef struct Hit Hit;

struct Object {
    float radius;
    float3 center;
};
typedef struct Object Object;

float3 random_in_unit_sphere(global float* distribution, int distr_size, int seed ) {
    int seed_l = (get_global_id(0)*10 + get_global_id(1)*10) + seed*3;

    float3 randvec = (float3)(0.f, 0.f, 0.f);
    float eta = 2.f*(M_PI)*distribution[(seed_l) % distr_size];
    // idk why, but acos is weird
    float phi = (distribution[(seed_l+1) % distr_size] - distribution[(seed_l+2) % distr_size])*(M_PI_2); //acos(2.f*distribution[(seed_l+1) % distr_size] - 1.f) - (pi/2.f); 

    randvec.x =  cos(eta)*cos(eta);
    randvec.y = cos(phi)*sin(eta);
    randvec.z = sin(phi);
    return randvec;
}

Ray make_ray(float u, float v, float3 camera_origin, float3 camera_ll_corner, float3 camera_horizontal, float3 camera_vertical) {
    Ray result;
    result.origin = camera_origin;
    result.direction = camera_ll_corner + u*camera_horizontal + v*camera_vertical - camera_origin;
    return result;
}

Hit get_hit(Ray r, float t_min, float t_max, int objects_num, global float* objects) {
    Hit result;
    result.t = -1.f;
    result.point = (float3)(0.f, 0.f, 0.f);
    result.normal = (float3)(0.f, 0.f, 0.f);
    for (int i = 0; i < objects_num; i++) {
        float3 center = (float3)(objects[i*4], objects[i*4 + 1], objects[i*4 + 2]);
        float radius = objects[i*4 + 3];
        float3 oc = r.origin - center;
        float a = dot(r.direction, r.direction);
        float b = dot(oc, r.direction);
        float c = dot(oc, oc) - radius*radius;
        float discriminant = b*b - a*c;

        if (discriminant > 0) {
            float d = sqrt(discriminant);
            float t = (-b - d)/a;
            bool success = false;
            if (t_min < t && t < t_max) {
                success = true;
            } else {
                t = (-b + d)/a;
                if (t_min < t && t < t_max) { success = true; }
            }
            if (success && (result.t <= 0 || result.t > t)) {
                result.t = t;
                result.point = r.origin + t*r.direction;
                result.normal = (result.point - center) / radius;
            }
        }
    }
    return result;
}

float3 trace(Ray r, int objects_num, global float* objects, global float* distr, float distr_size, int ray_num) {
    float factor = 1;
    const int max_depth = 50;
    int depth = 0;
    for (; depth<max_depth; ++depth) {
        Hit hit = get_hit(r, 1e-3f, FLT_MAX, objects_num, objects);
        if (hit.t > 1e-3f) {
            r.origin = hit.point;
            r.direction = hit.normal; 
            float3 rnd = random_in_unit_sphere(distr, distr_size, 100*depth + 10*ray_num); 
            //rnd = normalize(rnd);
            r.direction += rnd; // scatter
            //r.direction = normalize(r.direction);
            factor *= 0.5f; // diffuse 50% of light, scatter the remaining
        } else {
            break;
        }
    }
    //if (depth == max_depth) { return vec{}; }
    // nothing was hit
    // represent sky as linear gradient in Y dimension
    r.direction /= length(r.direction);
    float t = 0.5f*(r.direction.y + 1.0f);
    return factor*((1.0f-t)*(float3)(1.0f, 1.0f, 1.0f) + t*(float3)(0.5f, 0.7f, 1.0f));
}

kernel void ray_trace(
                                global float3* camera_origin,
                                global float3* camera_ll_corner,
                                global float3* camera_horizontal,
                                global float3* camera_vertical,

                                global float* objects,
                                int objects_num,

                                global float* distribution,
                                int distr_size,
                                global float* result,
                                int ny, int nx, int nrays, float gamma) {

        const int y = get_global_id(0);
        const int x = get_global_id(1);

        const int i = y*nx + x;


        float3 camera_origin_p = camera_origin[0];
        float3 camera_ll_corner_p = camera_ll_corner[0];
        float3 camera_horizontal_p = camera_horizontal[0];
        float3 camera_vertical_p = camera_vertical[0];

        
        float3 sum = (float3)(0.f, 0.f, 0.f);
        for (int k=0; k<nrays; ++k) {
            float u = (float)(x + distribution[(2*(i+k) + x + nrays)%distr_size]) / nx;
            float v = (float)(y + distribution[(2*(i+k) + y + 1 + nrays)%distr_size]) / ny;
            Ray ray = make_ray(u, v, camera_origin_p, camera_ll_corner_p, camera_horizontal_p, camera_vertical_p);
            sum += trace(ray, objects_num, objects, distribution, distr_size, k);
        }
        sum /= (float)(nrays); // antialiasing
        sum = pow(sum,1.f/gamma); // gamma correction

        result[3 * i + 0] = sum.x;
        result[3 * i + 1] = sum.y;
        result[3 * i + 2] = sum.z;
}

kernel void move_camera(global float3* camera_origin, global float3* camera_move_direction) {
    float3 cur_pos = camera_origin[0];
    cur_pos += camera_move_direction[0];
    camera_origin[0] = cur_pos;
}

)";

void start_gpu() {
   try {
        // find OpenCL platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            std::cerr << "Unable to find OpenCL platforms\n";
            return;
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
        cl::Program program(context, kernelsmykernels);
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
        ray_tracing_gpu(opencl);

    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error in " << err.what() << '(' << err.err() << ")\n";
        std::cerr << "Search cl.h file for error code (" << err.err()
            << ") to understand what it means:\n";
        std::cerr << "https://github.com/KhronosGroup/OpenCL-Headers/blob/master/CL/cl.h\n";
        return;
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        return;
    }
    return;
}

int main(int argc, char* argv[]) {
    enum class Version { CPU, GPU };
    Version version = Version::CPU;
    if (argc == 2) {
        std::string str(argv[1]);
        for (auto& ch : str) { ch = std::tolower(ch); }
        if (str == "gpu") { version = Version::GPU; }
    }
    switch (version) {
        case Version::CPU: ray_tracing_cpu(); break;
        case Version::GPU: start_gpu(); break;
        default: return 1;
    }
    return 0;
}
