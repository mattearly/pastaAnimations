cmake_minimum_required (VERSION 3.15)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(assimp CONFIG REQUIRED)
find_package(unofficial-nativefiledialog CONFIG REQUIRED)
find_path(STB_INCLUDE_DIRS "stb_c_lexer.h")

# Add source to this project's executable.
set(EXE_NAME "pastaAnimations")
add_executable (${EXE_NAME} "pastaAnimations.cpp" "pastaAnimations.h")

target_include_directories(pastaAnimations PRIVATE ${STB_INCLUDE_DIRS})

# link required libs
target_link_libraries(pastaAnimations
  PRIVATE
   glad::glad
   glfw
   glm::glm
   assimp::assimp
   unofficial::nativefiledialog::nfd
)

file(COPY "${PROJECT_SOURCE_DIR}/shaders/" DESTINATION "${CMAKE_BINARY_DIR}/${EXE_NAME}/shaders/")
# file(COPY "${PROJECT_SOURCE_DIR}/models/" DESTINATION "${CMAKE_BINARY_DIR}/${EXE_NAME}/models/")


# TODO: Add tests and install targets if needed.
