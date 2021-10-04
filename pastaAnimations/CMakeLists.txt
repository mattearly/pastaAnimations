cmake_minimum_required (VERSION 3.8)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)

find_path(STB_INCLUDE_DIRS "stb.h")

# Add source to this project's executable.
add_executable (pastaAnimations "pastaAnimations.cpp" "pastaAnimations.h")

target_include_directories(pastaAnimations PRIVATE ${STB_INCLUDE_DIRS})

# link dem libs
target_link_libraries(pastaAnimations PRIVATE glad::glad)
target_link_libraries(pastaAnimations PRIVATE glfw)
target_link_libraries(pastaAnimations PRIVATE glm::glm)
target_link_libraries(pastaAnimations PRIVATE assimp::assimp)

configure_file(../frag.glsl ${CMAKE_BINARY_DIR}/pastaAnimations/frag.glsl COPYONLY)
configure_file(../vert.glsl ${CMAKE_BINARY_DIR}/pastaAnimations/vert.glsl COPYONLY)

# TODO: Add tests and install targets if needed.
