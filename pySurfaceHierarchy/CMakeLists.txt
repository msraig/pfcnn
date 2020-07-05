cmake_minimum_required (VERSION 3.8)

project(pySurfaceHierarchy)

set(BINARIES_RELATIVE_PATH /binaries/)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}${BINARIES_RELATIVE_PATH})

include_directories(include)
file(GLOB HEADERS "include/*.hpp" "src/TangentParam/tangent_param.h" "include/DT2D/delabella.cpp")
file(GLOB SOURCES "src/*.cpp" "src/TangentParam/tangent_param.cpp" "include/DT2D/delabella.h")

#include Geex third party
include_directories(include/Geex/third_party)

#include openmesh directories
set(OPENMESH_PATH "D:/lib/OpenMesh 8.0")
include_directories(${OPENMESH_PATH}/include/)
link_directories(${OPENMESH_PATH}/lib/)
add_definitions(-D_USE_MATH_DEFINES)
add_definitions(-DOM_STATIC_BUILD)
add_definitions(-DNOMINMAX)

#include eigen
set(EIGEN_PATH "D:/lib/eigen323")
include_directories(${EIGEN_PATH})

#include GeometricTools
set(GEOMETRIC_TOOLS_PATH "D:/lib/GeometricTools/WildMagic5/SDK")
include_directories(${GEOMETRIC_TOOLS_PATH}/Include/)
link_directories(${GEOMETRIC_TOOLS_PATH}/Library/v141/x64/Debug)
link_directories(${GEOMETRIC_TOOLS_PATH}/Library/v141/x64/Release)

#include ANN lib
SET(ANNKDTREE_PATH "D:/lib/ann_1.1.2_built")
include_directories(${ANNKDTREE_PATH}/include/)
link_directories(${ANNKDTREE_PATH}/lib/)

#include pybind
SET(PYBIND11_PATH "D:/lib/pybind11")
include_directories(${PYBIND11_PATH}/include/)

#include python path
SET(PYTHON_PATH "C:/Python/Python35")
include_directories(${PYTHON_PATH}/include/)
link_directories(${PYTHON_PATH}/libs/)


#executable
add_library(pySurfaceHierarchy SHARED "src/pySurfaceHierarchy.cpp" ${SOURCES} ${HEADERS})
set_target_properties(pySurfaceHierarchy PROPERTIES SUFFIX ".pyd")

#link openmesh lib
target_link_libraries(pySurfaceHierarchy 
	debug OpenMeshCored.lib optimized OpenMeshCore.lib
	debug OpenMeshToolsd.lib optimized OpenMeshTools.lib)

#link geometric tools lib
target_link_libraries(pySurfaceHierarchy 
	optimized Wm5Core.lib optimized Wm5Mathematics.lib
	debug Wm5CoreD.lib debug Wm5MathematicsD.lib)
	
#link ann kdtree lib
target_link_libraries(pySurfaceHierarchy ANN.lib)

#link python lib
target_link_libraries(pySurfaceHierarchy python3.lib python35.lib)
