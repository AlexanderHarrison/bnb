cmake_minimum_required(VERSION 3.9.4)
project(bnb)

set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

include(ExternalProject)
ExternalProject_Add(HiGHS
  GIT_REPOSITORY https://github.com/ERGO-Code/HiGHS.git
  GIT_TAG v1.5.3
  PREFIX ${CMAKE_BINARY_DIR}/external/HiGHS
  CMAKE_ARGS -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
  INSTALL_COMMAND ""  # Prevent installation
)

ExternalProject_Get_Property(HiGHS SOURCE_DIR)
set(HIGHS_INCLUDE_DIR ${SOURCE_DIR}/src)

ExternalProject_Get_Property(HiGHS BINARY_DIR)
set(HIGHS_LIBRARY_DIR ${BINARY_DIR}/lib)

add_executable(bnb c/bnb.cpp)

target_include_directories(bnb PRIVATE ${HIGHS_INCLUDE_DIR} ${BINARY_DIR})

target_link_directories(bnb PRIVATE ${HIGHS_LIBRARY_DIR})

target_link_libraries(bnb PRIVATE ${HIGHS_LIBRARY_DIR}/libhighs.a)

add_dependencies(bnb HiGHS)
set_target_properties(bnb PROPERTIES BUILD_RPATH ${HIGHS_LIBRARY_DIR})

include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED)
if(IPO_SUPPORTED)
  set_target_properties(HiGHS PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  set_target_properties(bnb PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
