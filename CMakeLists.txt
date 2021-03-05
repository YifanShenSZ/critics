cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(v0)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Release)

# OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# adiabatz
find_package(adiabatz REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${adiabatz_CXX_FLAGS}")

# Foptim
set(CMAKE_PREFIX_PATH ~/Library/Foptim)
find_package(Foptim REQUIRED)

# Torch-Chemistry
if(NOT tchem_FOUND)
    set(CMAKE_PREFIX_PATH ~/Library/Torch-Chemistry)
    find_package(tchem REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${tchem_CXX_FLAGS}")
endif()

include_directories(include)

add_executable(critics.exe
    source/int2cart.cpp
    source/global.cpp
    source/minimum.cpp
    source/main.cpp
)

target_link_libraries(critics.exe
    "${adiabatz_LIBRARIES}"
    "${Foptim_LIBRARIES}" "${tchem_LIBRARIES}"
)