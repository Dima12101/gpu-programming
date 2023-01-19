#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_TARGET_OPENCL_VERSION 120
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl2.hpp>
#endif

#include <GL/glew.h>
#include <GL/glut.h>

#include "opengl.hh"
#include "vector.hh"

using clock_type = std::chrono::high_resolution_clock;
using float_duration = std::chrono::duration<float>;
using vec2 = Vector<float,2>;

enum class Version { CPU, GPU };
Version version;

struct OpenCL {
    cl::Platform platform;
    cl::Device device;
    cl::Context context;
    cl::Program program;
    cl::CommandQueue queue;
};
OpenCL opencl;

// Original code: https://github.com/cerrno/mueller-sph
constexpr const float kernel_radius = 16;
constexpr const float particle_mass = 65;
constexpr const float poly6 = 315.f/(65.f*float(M_PI)*std::pow(kernel_radius,9));
constexpr const float spiky_grad = -45.f/(float(M_PI)*std::pow(kernel_radius,6));
constexpr const float visc_laplacian = 45.f/(float(M_PI)*std::pow(kernel_radius,6));
constexpr const float gas_const = 2000.f;
constexpr const float rest_density = 1000.f;
constexpr const float visc_const = 250.f;
constexpr const vec2 G(0.f, 12000*-9.8f);

struct Particle {

    vec2 position;
    vec2 velocity;
    vec2 force;
    float density;
    float pressure;

    Particle() = default;
    inline explicit Particle(vec2 x): position(x) {}

};


struct Particles_GPU {
    int count = 0;
    std::vector<float> positions_plain;
};

std::vector<Particle> particles;
Particles_GPU particles_GPU;

void generate_particles() {
    std::random_device dev;
    std::default_random_engine prng(dev());
    float jitter = 1;
    std::uniform_real_distribution<float> dist_x(-jitter,jitter);
    std::uniform_real_distribution<float> dist_y(-jitter,jitter);
    int ni = 15;
    int nj = 40;
    float x0 = window_width*0.25f;
    float x1 = window_width*0.75f;
    float y0 = window_height*0.20f;
    float y1 = window_height*1.00f;
    float step = 1.5f*kernel_radius;
    for (float x=x0; x<x1; x+=step) {
        for (float y=y0; y<y1; y+=step) {
            particles.emplace_back(vec2{x+dist_x(prng),y+dist_y(prng)});
        }
    }
    std::clog << "No. of particles: " << particles.size() << std::endl;
}

struct Buffers {
    cl::Buffer positions_buf;
    cl::Buffer forces_buf;
    cl::Buffer velocities_buf;
    cl::Buffer densities_buf;
    cl::Buffer pressures_buf;
};

Buffers buffers;

void generate_particles_OpenCl() {
    std::random_device dev;
    std::default_random_engine prng(dev());
    float jitter = 1;
    std::uniform_real_distribution<float> dist_x(-jitter,jitter);
    std::uniform_real_distribution<float> dist_y(-jitter,jitter);
    int ni = 15;
    int nj = 40;
    float x0 = window_width*0.25f;
    float x1 = window_width*0.75f;
    float y0 = window_height*0.20f;
    float y1 = window_height*1.00f;
    float step = 1.5f*kernel_radius;

    for (float x = x0; x < x1; x += step) {
        for (float y = y0; y < y1; y += step) {
            particles_GPU.positions_plain.push_back(x + dist_x(prng));
            particles_GPU.positions_plain.push_back(y + dist_y(prng));
            particles_GPU.count++;
        }
    }
    std::clog << "No. of particles: " << particles_GPU.count << std::endl;

    buffers.positions_buf = cl::Buffer(opencl.context, begin(particles_GPU.positions_plain), end(particles_GPU.positions_plain), true);
    int buf_size = particles_GPU.count*sizeof(cl_float);
    buffers.velocities_buf = cl::Buffer(opencl.context, CL_MEM_READ_WRITE, 2*buf_size);
    buffers.forces_buf = cl::Buffer(opencl.context, CL_MEM_READ_WRITE, 2*buf_size);
    buffers.densities_buf = cl::Buffer(opencl.context, CL_MEM_READ_WRITE, buf_size);
    buffers.pressures_buf = cl::Buffer(opencl.context, CL_MEM_READ_WRITE, buf_size);
}

void compute_density_and_pressure() {
    const auto kernel_radius_squared = kernel_radius*kernel_radius;
    #pragma omp parallel for schedule(dynamic)
    for (auto& a : particles) {
        float sum = 0;
        for (auto& b : particles) {
            auto sd = square(b.position-a.position);
            if (sd < kernel_radius_squared) {
                sum += particle_mass*poly6*std::pow(kernel_radius_squared-sd, 3);
            }
        }
        a.density = sum;
        a.pressure = gas_const*(a.density - rest_density);
    }
}

void compute_forces() {
    #pragma omp parallel for schedule(dynamic)
    for (auto& a : particles) {
        vec2 pressure_force(0.f, 0.f);
        vec2 viscosity_force(0.f, 0.f);
        for (auto& b : particles) {
            if (&a == &b) { continue; }
            auto delta = b.position - a.position;
            auto r = length(delta);
            if (r < kernel_radius) {
                pressure_force += -unit(delta)*particle_mass*(a.pressure + b.pressure)
                    / (2.f * b.density)
                    * spiky_grad*std::pow(kernel_radius-r,2.f);
                viscosity_force += visc_const*particle_mass*(b.velocity - a.velocity)
                    / b.density * visc_laplacian*(kernel_radius-r);
            }
        }
        vec2 gravity_force = G * a.density;
        a.force = pressure_force + viscosity_force + gravity_force;
    }
}

void compute_positions() {
    const float time_step = 0.0008f;
    const float eps = kernel_radius;
    const float damping = -0.5f;
    #pragma omp parallel for
    for (auto& p : particles) {
        // forward Euler integration
        p.velocity += time_step*p.force/p.density;
        p.position += time_step*p.velocity;
        // enforce boundary conditions
        if (p.position(0)-eps < 0.0f) {
            p.velocity(0) *= damping;
            p.position(0) = eps;
        }
        if (p.position(0)+eps > window_width) {
            p.velocity(0) *= damping;
            p.position(0) = window_width-eps;
        }
        if (p.position(1)-eps < 0.0f) {
            p.velocity(1) *= damping;
            p.position(1) = eps;
        }
        if (p.position(1)+eps > window_height) {
            p.velocity(1) *= damping;
            p.position(1) = window_height-eps;
        }
    }
}

void on_display() {
    if (no_screen) { glBindFramebuffer(GL_FRAMEBUFFER,fbo); }
    glClear(GL_COLOR_BUFFER_BIT);
    glLoadIdentity();
    gluOrtho2D(0, window_width, 0, window_height);
    glColor4f(0.2f, 0.6f, 1.0f, 1);
    glBegin(GL_POINTS);
    switch (version) {
        case Version::CPU: 
            for (const auto& particle : particles) {
                glVertex2f(particle.position(0), particle.position(1));
            }
            break;
        case Version::GPU:
            for (int i = 0; i < particles_GPU.count; i++) {
                glVertex2f(particles_GPU.positions_plain[i*2], particles_GPU.positions_plain[i*2+1]);
            }
            break;
        default: throw std::runtime_error("bad renderer");
    }
    glEnd();
    glutSwapBuffers();
    if (no_screen) { glReadBuffer(GL_RENDERBUFFER); }
    recorder.record_frame();
    if (no_screen) { glBindFramebuffer(GL_FRAMEBUFFER,0); }
}

void on_idle_cpu() {
    if (particles.empty()) { generate_particles(); }
    using std::chrono::duration_cast;
    using std::chrono::seconds;
    using std::chrono::microseconds;
    auto t0 = clock_type::now();
    compute_density_and_pressure();
    compute_forces();
    compute_positions();
    auto t1 = clock_type::now();
    auto dt = duration_cast<float_duration>(t1-t0).count();
    std::clog
        << std::setw(20) << dt
        << std::setw(20) << 1.f/dt
        << std::endl;
	glutPostRedisplay();
}

void on_idle_gpu() {
    if (particles_GPU.count == 0) {
         generate_particles_OpenCl(); 
    }
    using std::chrono::duration_cast;
    using std::chrono::seconds;
    using std::chrono::microseconds;

    cl::Kernel density_kernel(opencl.program, "density_pressure");
    cl::Kernel forces_kernel(opencl.program, "forces");
    cl::Kernel positions_kernel(opencl.program, "positions");

    int count = particles_GPU.count;
    auto t0 = clock_type::now();

    density_kernel.setArg(0, buffers.positions_buf);
    density_kernel.setArg(1, buffers.densities_buf);
    density_kernel.setArg(2, buffers.pressures_buf);
    opencl.queue.flush();

    opencl.queue.enqueueNDRangeKernel(density_kernel, cl::NullRange, cl::NDRange(count), cl::NullRange);
    opencl.queue.flush();

    forces_kernel.setArg(0, buffers.positions_buf);
    forces_kernel.setArg(1, buffers.forces_buf);
    forces_kernel.setArg(2, buffers.velocities_buf);
    forces_kernel.setArg(3, buffers.densities_buf);
    forces_kernel.setArg(4, buffers.pressures_buf);
    opencl.queue.flush();

    opencl.queue.enqueueNDRangeKernel(forces_kernel, cl::NullRange, cl::NDRange(count), cl::NullRange);
    opencl.queue.flush();

    positions_kernel.setArg(0, buffers.positions_buf);
    positions_kernel.setArg(1, buffers.forces_buf);
    positions_kernel.setArg(2, buffers.velocities_buf);
    positions_kernel.setArg(3, buffers.densities_buf);
    positions_kernel.setArg(4, 800); // window_width
    positions_kernel.setArg(5, 600); // window_height
    opencl.queue.flush();

    opencl.queue.enqueueNDRangeKernel(positions_kernel, cl::NullRange, cl::NDRange(count), cl::NullRange);
    opencl.queue.flush();

    auto t1 = clock_type::now();
    opencl.queue.enqueueReadBuffer(buffers.positions_buf, true, 0, 2*count*sizeof(cl_float), particles_GPU.positions_plain.data());
    auto dt = duration_cast<float_duration>(t1-t0).count();
    std::clog
        << std::setw(20) << dt
        << std::setw(20) << 1.f/dt
        << std::endl;
	glutPostRedisplay();
}

void on_keyboard(unsigned char c, int x, int y) {
    switch(c) {
        case ' ':
            generate_particles();
            break;
        case 'r':
        case 'R':
            particles.clear();
            generate_particles();
            break;
    }
}

const std::string calc_kernels = R"(
#define kernel_radius  16
#define kernel_radius_squared (float)(kernel_radius*kernel_radius)
#define particle_mass  65
#define poly6  315.f/(65.f*(float)(M_PI)*pow((float)kernel_radius,(float)9))
#define spiky_grad  -45.f/((float)(M_PI)*pow((float)kernel_radius,(float)6))
#define visc_laplacian  45.f/((float)(M_PI)*pow((float)kernel_radius,(float)6))
#define gas_const  2000.f
#define rest_density  1000.f
#define visc_const  250.f
#define G (float2)(0.f, 12000*-9.8f)

float square(float2 f) {
    return dot(f, f);
}

kernel void density_pressure(global float2* positions, global float* densities, global float* pressures) {
    int g_id = get_global_id(0);
    int size = get_global_size(0);

    float sum = 0;
    float2 p_pos = positions[g_id];

    for (int i = 0; i < size; i++) {
        float2 pos = positions[i];
        float sd = square(pos - p_pos);
        if (sd < kernel_radius_squared) {
            sum += particle_mass*poly6*pow(kernel_radius_squared-sd, 3);
        }
    }

    densities[g_id] = sum;
    pressures[g_id] = gas_const*(sum - rest_density);
}

kernel void forces( global float2* positions, global float2* forces, global float2* velocities,
    global float* densities, global float* pressures ) {

    int g_id = get_global_id(0);
    int size = get_global_size(0);

    float2 pressure_force = (float2)(0.f, 0.f);
    float2 viscosity_force = (float2)(0.f, 0.f);
    float2 cur_pos = positions[g_id];
    float cur_pres = pressures[g_id];
    float cur_dens = densities[g_id];
    float2 cur_vel = velocities[g_id];

    for (int i = 0; i < size; i++) {
        if (i == g_id) continue;

        float2 pos = positions[i];
        float2 delta = pos - cur_pos;

        float dist = length(delta);
        if (dist < kernel_radius) {
            float pres = pressures[i];
            float dens = densities[i];
            float2 vel = velocities[i];
            pressure_force += -normalize(delta) * particle_mass*(cur_pres + pres) / (2.f * dens) * spiky_grad * pow(kernel_radius - dist, 2.f);
            viscosity_force += visc_const * particle_mass*(vel - cur_vel) / dens * visc_laplacian*(kernel_radius-dist);
        }
    }
    float2 gravity_force = G * cur_dens;
    forces[g_id] = pressure_force + viscosity_force + gravity_force;
}

kernel void positions( 
    global float2* positions, global float2* forces,
    global float2* velocities, global float* densities,
    int window_width, int window_height) {

    int g_id = get_global_id(0);

    const float td = 0.0008f;
    const float damping = -0.5f;

    velocities[g_id] += td * forces[g_id] / densities[g_id];
    positions[g_id] += td * velocities[g_id];

    float2 cur_pos = positions[g_id];
    float2 cur_vel = velocities[g_id];

    if (cur_pos.x - kernel_radius < 0.0f) {
        cur_vel.x *= damping;
        cur_pos.x = kernel_radius;
    }
    if (cur_pos.x + kernel_radius > window_width) {
        cur_vel.x *= damping;
        cur_pos.x = window_width-kernel_radius;
    }
    if (cur_pos.y - kernel_radius < 0.0f) {
        cur_vel.y *= damping;
        cur_pos.y = kernel_radius;
    }
    if (cur_pos.y + kernel_radius > window_height) {
        cur_vel.y *= damping;
        cur_pos.y = window_height - kernel_radius;
    }

    positions[g_id] = cur_pos;
    velocities[g_id] = cur_vel;
}

)";

void print_column_names() {
    std::clog << std::setw(20) << "Frame duration";
    std::clog << std::setw(20) << "Frames per second";
    std::clog << '\n';
}

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
        cl::Program program(context, calc_kernels);
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
        opencl = OpenCL{platform, device, context, program, queue};

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

    version = Version::CPU;
    if (argc == 2) {
        std::string str(argv[1]);
        for (auto& ch : str) { ch = std::tolower(ch); }
        if (str == "gpu") { version = Version::GPU; }
    }
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_MULTISAMPLE);
	glutInitWindowSize(window_width, window_height);
	glutInit(&argc, argv);
	glutCreateWindow("SPH");
	glutDisplayFunc(on_display);
    glutReshapeFunc(on_reshape);
    switch (version) {
        case Version::CPU: glutIdleFunc(on_idle_cpu); break;
        case Version::GPU: start_gpu(); glutIdleFunc(on_idle_gpu); break;
        default: return 1;
    }
	glutKeyboardFunc(on_keyboard);
    glewInit();
	init_opengl(kernel_radius);
    print_column_names();
	glutMainLoop();
    return 0;
}
