#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <GL/glew.h>
#include <GL/glut.h>

#include "opengl.hh"
#include "vector.hh"

using clock_type = std::chrono::high_resolution_clock;
using float_duration = std::chrono::duration<float>;
using vec2 = Vector<float,2>;

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

std::vector<Particle> particles;

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
    for (const auto& particle : particles) {
        glVertex2f(particle.position(0), particle.position(1));
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
    std::clog << "GPU version is not implemented!" << std::endl; std::exit(1);
    using std::chrono::duration_cast;
    using std::chrono::seconds;
    using std::chrono::microseconds;
    auto t0 = clock_type::now();
    // TODO see on_idle_cpu
    auto t1 = clock_type::now();
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

void print_column_names() {
    std::clog << std::setw(20) << "Frame duration";
    std::clog << std::setw(20) << "Frames per second";
    std::clog << '\n';
}

int main(int argc, char* argv[]) {
    enum class Version { CPU, GPU };
    Version version = Version::CPU;
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
        case Version::GPU: glutIdleFunc(on_idle_gpu); break;
        default: return 1;
    }
	glutKeyboardFunc(on_keyboard);
    glewInit();
	init_opengl(kernel_radius);
    print_column_names();
	glutMainLoop();
    return 0;
}
