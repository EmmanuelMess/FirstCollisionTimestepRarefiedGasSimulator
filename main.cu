#include <iostream>
#include <chrono>
#include <array>
#include <random>
#include <numeric>

#include <curand_kernel.h>

#include <cuda/api.hpp>

#include "EasyBMP.hpp"

constexpr unsigned int ITERATIONS = 20;

constexpr unsigned int WIDTH = 500;
constexpr unsigned int HEIGHT = 500;

constexpr unsigned int PARTICLES = 5;

__device__ __constant__ const float infinity = std::numeric_limits<float>::infinity();
__device__ __constant__ const float radius = 50;
__device__ __constant__ const float dt = 10;

typedef float2 Point;
typedef float2 Velocity;
typedef float Time;

enum CollisionType {
    NONE = 0,
    IGNORE,
    PARTICLE_PARTICLE,
    PARTICLE_WALL_X,
    PARTICLE_WALL_Y
};

struct Collision {
    CollisionType type;
    unsigned int indexB;
};

__global__ void fillInitialPositions(Point* initialPositions) {
    const unsigned int i = threadIdx.x;

    curandState randomState;
    curand_init(1, i, 0, &randomState);
    initialPositions[i] = {
            radius + fmod(static_cast<float>(curand(&randomState)), WIDTH - radius*2),
            radius + fmod(static_cast<float>(curand(&randomState)), HEIGHT - radius*2)
    };

    printf("%d p(%f, %f)\n", i, initialPositions[i].x, initialPositions[i].y);
}

__global__ void fillInitialVelocity(Velocity *velocities) {
    const unsigned int i = threadIdx.x;

    const float length = 20;

    curandState randomState;
    curand_init(2, i, 0, &randomState);
    Velocity randomVelocity = {
            0.1f + fmod(static_cast<float>(curand(&randomState)), static_cast<float>(10)),
            0.1f + fmod(static_cast<float>(curand(&randomState)), static_cast<float>(10))
    };

    const float randomLength = sqrt(pow(randomVelocity.x, 2) + pow(randomVelocity.y, 2));

    velocities[i] = { randomVelocity.x / randomLength * length, randomVelocity.y / randomLength * length };

    printf("%d v(%f, %f)\n", i, velocities[i].x, velocities[i].y);
}

__global__ void fillInitialIntersectionTimes(Time * const intersectionTimes, const size_t intersectionTimesPitch) {
    const unsigned int i = threadIdx.x;
    const unsigned int j = threadIdx.y;

    auto rowIntersectionTimes = reinterpret_cast<Time*>(reinterpret_cast<char*>(intersectionTimes)
            + i * intersectionTimesPitch);
    Time &intersectionTime = rowIntersectionTimes[j];

    intersectionTime = infinity;
}


__global__ void calculateIntersectionTime(const Point *initialPositions, const Velocity *velocities,
                                          Time * const intersectionTimes, const size_t intersectionTimesPitch) {
    const unsigned int i = threadIdx.x;
    const Point pointA = initialPositions[i];
    const Velocity velocityA = velocities[i];

    const unsigned int j = threadIdx.y;
    const Point pointB = initialPositions[j];
    const Velocity velocityB = velocities[j];

    auto rowIntersectionTimes = reinterpret_cast<Time *>(reinterpret_cast<char *>(intersectionTimes) +
                                                         i * intersectionTimesPitch);

    if (i == j || i < j) {
        return;
    }

    rowIntersectionTimes[j] = infinity;

    if (sqrt(pow(pointA.x - pointB.x, 2) + pow(pointA.y - pointB.y, 2)) <= 2 * radius) {
        //TODO fix this on point generation
        printf("Overlap: %d((%f, %f), (%f, %f)) and %d((%f, %f), (%f, %f))\n", i, pointA.x, pointA.y,
               velocityA.x, velocityA.y, j, pointB.x, pointB.y, velocityB.x, velocityB.y);
        return;
    }

    const float a = pow(velocityA.x - velocityB.x, 2) + pow(velocityA.y - velocityB.y, 2);
    const float b = 2 * ((pointA.x - pointB.x) * (velocityA.x - velocityB.x) +(pointA.y - pointB.y) * (velocityA.y - velocityB.y));
    const float c = pow(pointA.x - pointB.x, 2) + pow(pointA.y - pointB.y, 2) - pow(2 * radius, 2);

    const float d = pow(b, 2) - 4 * a * c;

    if (d < 0) {
        printf("No intersect: %d((%f, %f), (%f, %f)) and %d((%f, %f), (%f, %f))\n", i, pointA.x, pointA.y,
               velocityA.x, velocityA.y, j, pointB.x, pointB.y, velocityB.x, velocityB.y);
        return;
    }
    if (b > -1e-6) {
        printf("Glancing: %d((%f, %f), (%f, %f)) and %d((%f, %f), (%f, %f))\n", i, pointA.x, pointA.y,
               velocityA.x, velocityA.y, j, pointB.x, pointB.y, velocityB.x, velocityB.y);
        return;
    }

    const Time t0 = (-b + sqrt(d)) / (2 * a);
    const Time t1 = (-b - sqrt(d)) / (2 * a);

    if (b >= 0) {
        printf("Getting farther: %d((%f, %f), (%f, %f)) and %d((%f, %f), (%f, %f))\n", i, pointA.x, pointA.y,
               velocityA.x, velocityA.y, j, pointB.x, pointB.y, velocityB.x, velocityB.y);
        return;
    }
    if (t0 < 0 && t1 > 0 && b <= -1e-6) {
        printf("No intersect: %d((%f, %f), (%f, %f)) and %d((%f, %f), (%f, %f))\n", i, pointA.x, pointA.y,
               velocityA.x, velocityA.y, j, pointB.x, pointB.y, velocityB.x, velocityB.y);
        return;
    }

    const Time t = t1;
    rowIntersectionTimes[j] = t;

    printf("Collision: %d((%f, %f), (%f, %f)) and %d((%f, %f), (%f, %f))\n", i, pointA.x, pointA.y,
           velocityA.x, velocityA.y, j, pointB.x, pointB.y, velocityB.x, velocityB.y);
}

__global__ void calculateIntersectionBorderTime(const Point *initialPositions, const Velocity *velocities,
                                                Time * const intersectionTimes, const size_t intersectionTimesPitch,
                                                Collision * const collidedParticles) {
    const unsigned int i = threadIdx.x;
    const Point pointA = initialPositions[i];
    const Velocity velocityA = velocities[i];

    auto rowIntersectionTimes = reinterpret_cast<Time *>(reinterpret_cast<char *>(intersectionTimes) +
                                                         i * intersectionTimesPitch);

    const auto collisionTimeParticleWall =
            [](const unsigned int i, const float velocity, const float point, const float wall, const char* debug) -> float { // TODO const char*?
        const float a = pow(velocity, 2);
        const float b = 2 * (point - wall) * velocity;
        const float c = (point - wall + radius) * (point - wall - radius);

        const float d = pow(b, 2) - 4 * a * c;

        if (d < 0) {
            printf("No intersect: %d((%f), (%f)) and %s\n", i, point, velocity, debug);
            return infinity;
        }
        if (b > -1e-6) {
            printf("Glancing: %d((%f), (%f)) and %s\n", i, point, velocity, debug);
            return infinity;
        }

        const Time t0 = (-b + sqrt(d)) / (2 * a);
        const Time t1 = (-b - sqrt(d)) / (2 * a);

        if (b >= 0) {
            printf("Getting farther: %d((%f), (%f)) and %s\n", i, point, velocity, debug);
            return infinity;
        }
        if (t0 < 0 && t1 > 0 && b <= -1e-6) {
            printf("No intersect: %d((%f), (%f)) and %s\n", i, point, velocity, debug);
            return infinity;
        }

        const Time t = t1; //TODO fix

        printf("Collision: %d((%f), (%f)) and %s\n", i, point, velocity, debug);
        return t;
    };

    const Time t0 = collisionTimeParticleWall(i, velocityA.x, pointA.x, 0, "Wx = 0");
    const Time t1 = collisionTimeParticleWall(i, velocityA.x, pointA.x, WIDTH, "Wx = WIDTH");
    const Time t2 = collisionTimeParticleWall(i, velocityA.y, pointA.y, 0, "Wy = 0");
    const Time t3 = collisionTimeParticleWall(i, velocityA.y, pointA.y, HEIGHT, "Wy = HEIGHT");

    rowIntersectionTimes[i] = min(min(t0, t1), min(t2, t3));

    if (min(t0, t1) < min(t2, t3)) {
        collidedParticles[i].type = PARTICLE_WALL_X;
    } else {
        collidedParticles[i].type = PARTICLE_WALL_Y;
    }
}

// TODO do a reduction as recommended by NVIDIA
__global__ void findMin(const Time *intersectionTimes, const size_t intersectionTimesPitch,
                        Collision* const collidedParticles, Time* result) {
    *result = dt;

    unsigned int indexA = 0;
    unsigned int indexB = 0;

    for (unsigned int i = 0; i < PARTICLES; i++) {
        auto rowIntersectionTimes = reinterpret_cast<const Time *>(reinterpret_cast<const char *>(intersectionTimes) +
                                                                   i * intersectionTimesPitch);

        for (unsigned int j = 0; j < PARTICLES; j++) {
            if (i < j) {
                continue;
            }
            printf("(%d, %d): %f\n", i, j, rowIntersectionTimes[j]);

            const Time &intersectionTimeA = rowIntersectionTimes[j];

            if (intersectionTimeA < *result) {
                *result = intersectionTimeA;
                indexA = i;
                indexB = j;
            }
        }
    }

    for (unsigned int i = 0; i < PARTICLES; i++) {
        if (indexA == i && *result < dt) {
            continue;
        }

        collidedParticles[i].type = NONE;
    }

    if (indexA == indexB) {
        return;
    }

    collidedParticles[indexA].type = PARTICLE_PARTICLE;
    collidedParticles[indexA].indexB = indexB;

    collidedParticles[indexB].type = IGNORE;
}

__global__ void advanceSimulation(Point * const initialPositions, Velocity * const velocities,
                                  const Collision * collidingParticles, const Time* timestep) {
    if(*timestep == 0) {
        printf("Timestep is 0, infinite loop!\n");
        return;
    }

    const unsigned int i = threadIdx.x;

    switch (collidingParticles[i].type) {
        case NONE: {
            Point& pointA = initialPositions[i];
            Velocity& velocityA = velocities[i];

            pointA.x = pointA.x + *timestep * velocityA.x;
            pointA.y = pointA.y + *timestep * velocityA.y;

            printf("%d: No collision!", i);
            return;
        }
        case IGNORE: {
            //Another particle is dealing with the collision
            printf("%d: Other particle collision!", i);
            return;
        }
        case PARTICLE_PARTICLE: {
            Point &pointA = initialPositions[i];
            Velocity &velocityA = velocities[i];
            pointA.x = pointA.x + *timestep * velocityA.x;
            pointA.y = pointA.y + *timestep * velocityA.y;


            Point &pointB = initialPositions[collidingParticles[i].indexB];
            Velocity &velocityB = velocities[collidingParticles[i].indexB];
            pointB.x = pointB.x + *timestep * velocityB.x;
            pointB.y = pointB.y + *timestep * velocityB.y;


            const float lengthA = sqrt(pow(velocityA.x, 2) + pow(velocityA.y, 2));
            const float2 normalizedVelocityA = { velocityA.x / lengthA, velocityA.y / lengthA };

            const float lengthB = sqrt(pow(velocityB.x, 2) + pow(velocityB.y, 2));
            const float2 normalizedVelocityB = { velocityB.x / lengthB, velocityB.y / lengthB };

            velocityA = { normalizedVelocityB.x * lengthA, normalizedVelocityB.y * lengthA };
            velocityB = { normalizedVelocityA.x * lengthB, normalizedVelocityA.y * lengthB };
            printf("%d: Particle collision!", i);
            return;
        }
        case PARTICLE_WALL_X: {
            Point& pointA = initialPositions[i];
            Velocity& velocityA = velocities[i];

            pointA.x = pointA.x + *timestep * velocityA.x;
            pointA.y = pointA.y + *timestep * velocityA.y;

            velocityA.x = -velocityA.x;
            printf("%d: Wall X collision!", i);
            return;
        }
        case PARTICLE_WALL_Y: {
            Point& pointA = initialPositions[i];
            Velocity& velocityA = velocities[i];

            pointA.x = pointA.x + *timestep * velocityA.x;
            pointA.y = pointA.y + *timestep * velocityA.y;

            velocityA.y = -velocityA.y;
            printf("%d: Wall Y collision!", i);
            return;
        }
        default:
            printf("Wrong particle collision type!\n");
            return;
    }
}

void simulate(Point* const initialPositions, Velocity* const velocities,
              Time* const intersectionTimes, const size_t intersectionTimesPitch,
              Collision * const collidedParticles,
              Time* const minimumTime, cuda::stream_t &stream) {
    //TODO rethink the grid and block sizes
    stream.enqueue.kernel_launch(calculateIntersectionTime,
                                 cuda::launch_config_builder().grid_dimensions(1, 1).block_dimensions(PARTICLES, PARTICLES).build(),
                                 initialPositions, velocities, intersectionTimes, intersectionTimesPitch);

    stream.enqueue.kernel_launch(calculateIntersectionBorderTime,
                                 cuda::launch_config_builder().grid_dimensions(1).block_dimensions(PARTICLES).build(),
                                 initialPositions, velocities, intersectionTimes, intersectionTimesPitch, collidedParticles);

    stream.enqueue.kernel_launch(findMin,
                                 cuda::launch_config_builder().grid_dimensions(1).block_dimensions(1).build(),
                                 intersectionTimes, intersectionTimesPitch, collidedParticles, minimumTime);

    stream.enqueue.kernel_launch(advanceSimulation,
                                 cuda::launch_config_builder().grid_dimensions(1).block_dimensions(PARTICLES).build(),
                                 initialPositions, velocities, collidedParticles, minimumTime);
    stream.synchronize();

    std::cout << *minimumTime << '\n';
}

void drawState(const unsigned int currentIteration, Point* initialPositions, const Velocity* velocities) {
    const std::string name = "results/" + std::to_string(currentIteration) + ".bmp";
    EasyBMP::Image image(WIDTH, HEIGHT, name, EasyBMP::RGBColor(0, 0, 0));

    std::minstd_rand rd; //Always generate same values
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> distrib(0, 255);

    for (unsigned int i = 0; i < PARTICLES; i++) {
        const float x = initialPositions[i].x;
        const float y = initialPositions[i].y;

        const EasyBMP::RGBColor color( distrib(gen), distrib(gen), distrib(gen));

        image.DrawCircle(std::lroundf(x), std::lroundf(y), 5, color, true);

        image.DrawCircle(std::lroundf(x), std::lroundf(y), std::lroundf(radius), EasyBMP::RGBColor(0, 0, 255), false);

        image.DrawLine(std::lroundf(x), std::lroundf(y),
                       std::lroundf(x + velocities[i].x), std::lroundf(y + velocities[i].y),
                       EasyBMP::RGBColor(255, 255, 255));
    }

    image.Write();
}

int main() {
    static_assert(PARTICLES >= 2);

    const cuda::device_t device = cuda::device::current::get();
    cuda::stream_t stream = device.create_stream(cuda::stream::implicitly_synchronizes_with_default_stream);

    auto initialPositions = cuda::memory::managed::make_unique<Point[]>(device, PARTICLES);
    stream.enqueue.kernel_launch(fillInitialPositions,
                                 cuda::launch_config_builder().grid_dimensions(1).block_size(PARTICLES).build(),
                                 initialPositions.get());

    auto velocities = cuda::memory::managed::make_unique<Velocity[]>(device, PARTICLES);
    stream.enqueue.kernel_launch(fillInitialVelocity,
                                 cuda::launch_config_builder().grid_dimensions(1).block_size(PARTICLES).build(),
                                 velocities.get());

    Time* intersectionTimes;
    size_t intersectionTimesPitch;
    cudaMallocPitch(&intersectionTimes, &intersectionTimesPitch, PARTICLES * sizeof(Time), PARTICLES);
    stream.enqueue.kernel_launch(fillInitialIntersectionTimes,
                 cuda::launch_config_builder().grid_dimensions(1, 1).block_dimensions(PARTICLES, PARTICLES).build(),
                 intersectionTimes, intersectionTimesPitch);

    auto time = cuda::memory::managed::make_unique<Time>();

    auto collidedParticles = cuda::memory::device::make_unique<Collision[]>(device, PARTICLES);

    stream.synchronize();

    /*TODO
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaGraphCreate(&graph, 0);
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    simulate(initialPositions.get(),  velocities.get(), intersectionTimes, intersectionTimesPitch, stream); //TODO rework optimization

    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
     */

    drawState(0, initialPositions.get(), velocities.get());

    std::array<float, ITERATIONS> times{};

    for (unsigned int i = 0; i < ITERATIONS; i++) {
        std::cout << i << ":" << '\n';

        auto start = std::chrono::high_resolution_clock::now();
        //TODO use graphs for perf
        /*
         * cudaGraphLaunch(instance, stream);
         * cudaStreamSynchronize(stream);
         */
        //TODO fix all the get()
        simulate(initialPositions.get(), velocities.get(), intersectionTimes, intersectionTimesPitch,
                 collidedParticles.get(), time.get(), stream);
        auto end = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<float, std::micro>(end - start).count();

        drawState(i + 1, initialPositions.get(), velocities.get());
    }

    float total = std::accumulate<>(std::begin(times), std::end(times), static_cast<float>(0), std::plus<>()) / 1000000;

    std::cout << "Average iteration time " << (total/ITERATIONS) << "ms" << '\n';
    std::cout << "Total time " << total << "ms" << '\n';

    cudaFree(intersectionTimes);
	return 0;
}
