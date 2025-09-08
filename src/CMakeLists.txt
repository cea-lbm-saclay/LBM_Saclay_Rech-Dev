# config file reader / ConfigMap / monitoring timer
add_subdirectory(utils)


  
add_executable(${PROJECT_NAME} "")

target_sources(${PROJECT_NAME}
    PRIVATE
    LBMParams.cpp
    LBMRunFactory.cpp
    LBM_Base_Functor.cpp
    io/io_common.cpp
    io/save_vtk.cpp
    io/IO_ReadWrite.cpp
    main.cpp
    )
#${BACKWARD_ENABLE}


foreach(entry IN LISTS PROBLEMDIR)
file(GLOB PROBLEM_SRC 
    CONFIGURE_DEPENDS 
    "${PROJECT_SOURCE_DIR}/src/kernels/${entry}/*.cpp")
message("${PROBLEM_SRC}")
target_sources(${PROJECT_NAME}
    PRIVATE
    ${PROBLEM_SRC}
)
endforeach()

if(USE_HDF5)
    target_sources(${PROJECT_NAME}
        PRIVATE 
        io/IO_HDF5.cpp
    )
endif(USE_HDF5)

#add_backward(${PROJECT_NAME})

# make sure LBM_saclay_version.h is found
target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
)
    

if(USE_HDF5)
    target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${HDF5_INCLUDE_DIRS})
endif(USE_HDF5)

target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC
    kokkos
    LBM_saclay::config
    LBM_saclay::monitoring
    LBM_saclay::output_selector
)
target_include_directories(${PROJECT_NAME}
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/output_selector
)
if(USE_HDF5)
    target_link_libraries(${PROJECT_NAME}
        PUBLIC
        ${HDF5_LIBRARIES}
    )
endif(USE_HDF5)

if (USE_MPI)
    target_link_libraries(${PROJECT_NAME}
        PUBLIC
        mpiUtils)
endif(USE_MPI)

# track time taken for build
add_custom_target(TIMER_${PROJECT_NAME}
                   COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/times
                   COMMAND date +%s%3N > ${CMAKE_CURRENT_BINARY_DIR}/times/start.log
                   BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/times/start.log
)

add_dependencies(${PROJECT_NAME} TIMER_${PROJECT_NAME})

add_custom_command(TARGET ${PROJECT_NAME}
                   POST_BUILD
                   COMMAND date +%s%3N > ${CMAKE_CURRENT_BINARY_DIR}/times/end.log 
                   COMMAND bash ${PROJECT_SOURCE_DIR}/cmake/show_build_time.sh ${CMAKE_CURRENT_BINARY_DIR}/times/start.log ${CMAKE_CURRENT_BINARY_DIR}/times/end.log "${PROJECT_NAME}"
                   BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/times/end.log 
                   VERBATIM)
