
cmake_minimum_required(VERSION 3.22.1)

project("descriptoresapp")

# Configuración de OpenCV (asegúrate de que OpenCV esté bien configurado)
set(OpenCV_STATIC on)
set(OpenCV_DIR /home/atlas/aplicaciones/OpenCV-android-sdk/sdk/native/jni)
find_package(OpenCV REQUIRED)

# Incluye la carpeta 'include' de OpenCV
include_directories(/home/atlas/aplicaciones/OpenCV-android-sdk/sdk/native/jni/include)

# Buscar la biblioteca log
find_library(
        log-lib
        log
)

add_library(${CMAKE_PROJECT_NAME} SHARED
        # List C/C++ source files with relative paths to this CMakeLists.txt.
        native-lib.cpp)

target_link_libraries(${CMAKE_PROJECT_NAME}
        ${OpenCV_LIBS}
        ${jnigraphics-lib}
        ${log-lib}
        android)
