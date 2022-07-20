cmake_minimum_required (VERSION 3.15)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)

find_path(STB_INCLUDE_DIRS "stb_c_lexer.h")

# Add source to this project's executable.
add_executable (pastaAnimations "pastaAnimations.cpp" "pastaAnimations.h")

target_include_directories(pastaAnimations PRIVATE ${STB_INCLUDE_DIRS})

# link required libs
target_link_libraries(pastaAnimations
  PRIVATE
   glad::glad
   glfw
   glm::glm
   assimp::assimp
)

configure_file(../frag.glsl ${CMAKE_BINARY_DIR}/pastaAnimations/frag.glsl COPYONLY)
configure_file(../vert.glsl ${CMAKE_BINARY_DIR}/pastaAnimations/vert.glsl COPYONLY)

# TODO: Add tests and install targets if needed.
