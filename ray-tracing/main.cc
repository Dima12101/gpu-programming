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
    int nx = 600, ny = 400, nrays = 100;
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

void ray_tracing_gpu() {
    std::clog << "GPU version is not implemented!" << std::endl; std::exit(1);
    using std::chrono::duration_cast;
    using std::chrono::seconds;
    using std::chrono::microseconds;
    int nx = 600, ny = 400, nrays = 100;
    Pixel_matrix<float> pixels(nx,ny);
    thx::screen_recorder recorder("out.ogv", nx,ny);
    std::vector<Sphere> objects = {
        Sphere{vec(0.f,0.f,-1.f),0.5f},
        Sphere{vec(0.f,-1000.5f,-1.f),1000.f}
    };
    Camera camera;
    uniform_distribution distribution(0.f,1.f);
    float gamma = 2;
    const int max_time_step = 60;
    print_column_names("OpenCL");
    duration total_time = duration::zero();
    for (int time_step=1; time_step<=max_time_step; ++time_step) {
        auto t0 = clock_type::now();
        // TODO Use GPU to race sun rays!
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
        case Version::GPU: ray_tracing_gpu(); break;
        default: return 1;
    }
    return 0;
}
