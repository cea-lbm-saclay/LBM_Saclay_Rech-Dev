add_library(config "")

target_sources(config
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/ConfigMap.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/inih/ini.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/inih/INIReader.cpp
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/ConfigMap.h
  ${CMAKE_CURRENT_SOURCE_DIR}/inih/ini.h
  ${CMAKE_CURRENT_SOURCE_DIR}/inih/INIReader.h
  )

target_include_directories(config
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/inih 
 ) 



if (USE_MPI)
  target_link_libraries(config PUBLIC MPI::MPI_CXX)
endif(USE_MPI)


# track time taken for build
add_custom_target(TIMER_config
                   COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/times
                   COMMAND date +%s%3N > ${CMAKE_CURRENT_BINARY_DIR}/times/start.log
                   BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/times/start.log
)

add_dependencies(config TIMER_config)

add_custom_command(TARGET config
                   POST_BUILD
                   COMMAND date +%s%3N > ${CMAKE_CURRENT_BINARY_DIR}/times/end.log 
                   COMMAND bash ${PROJECT_SOURCE_DIR}/cmake/show_build_time.sh ${CMAKE_CURRENT_BINARY_DIR}/times/start.log ${CMAKE_CURRENT_BINARY_DIR}/times/end.log "config"
                   BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/times/end.log
                   VERBATIM
)
                   
                   
#Add an alias so that library can be used inside the build tree
add_library(LBM_saclay::config ALIAS config)
