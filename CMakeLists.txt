cmake_minimum_required(VERSION 3.22)

enable_language(CXX)
enable_language(CUDA)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(FirstCollisionTimestepRarefiedGasSimulator CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_compile_options(-G)

include(FetchContent)
FetchContent_Declare(cuda-api-wrappers
        GIT_REPOSITORY https://github.com/eyalroz/cuda-api-wrappers.git
        GIT_TAG v0.5.2
        )
FetchContent_MakeAvailable(cuda-api-wrappers)

add_executable(FirstCollisionTimestepRarefiedGasSimulator main.cu EasyBMP.hpp)

target_link_libraries(FirstCollisionTimestepRarefiedGasSimulator runtime-and-driver)

set_target_properties(FirstCollisionTimestepRarefiedGasSimulator PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
