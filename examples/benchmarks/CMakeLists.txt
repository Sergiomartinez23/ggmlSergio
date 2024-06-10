
# correcteness

set(TEST_TARGET benchmark_correcteness)
add_executable(${TEST_TARGET} benchmark_correcteness.c)
target_link_libraries(${TEST_TARGET} PRIVATE ggml )

# time

set(TEST_TARGET benchmark_time_f32)
include_directories(${TEST_TARGET} /afs/atc.unican.es/u/m/martinezas/papi/src/install/include)
link_directories(${TEST_TARGET} /afs/atc.unican.es/u/m/martinezas/papi/src/install/lib)
add_executable(${TEST_TARGET} benchmark_time_f32.c)
target_link_libraries(${TEST_TARGET} PRIVATE ggml papi) 

# time

set(TEST_TARGET benchmark_time_f32_iterations)
include_directories(${TEST_TARGET} /afs/atc.unican.es/u/m/martinezas/papi/src/install/include)
link_directories(${TEST_TARGET} /afs/atc.unican.es/u/m/martinezas/papi/src/install/lib)
add_executable(${TEST_TARGET} benchmark_time_f32_iterations.c)
target_link_libraries(${TEST_TARGET} PRIVATE ggml papi)
