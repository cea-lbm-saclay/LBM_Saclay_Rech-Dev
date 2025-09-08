add_library(output_selector "")

target_sources(output_selector
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Selectors.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/OutputSelector.cpp
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/Selectors.h
    ${CMAKE_CURRENT_SOURCE_DIR}/OutputSelector.h
  )


target_link_libraries(output_selector PRIVATE config)

target_include_directories(output_selector
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/src/utils/config
 ) 
 
# track time taken for build
add_custom_target(TIMER_output_selector
                   COMMAND mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/times
                   COMMAND date +%s%3N > ${CMAKE_CURRENT_BINARY_DIR}/times/start.log
                   BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/times/start.log
)

add_dependencies(output_selector TIMER_output_selector)

# manually add dependencies of output_selector to its timer, to get correct timing
add_dependencies(TIMER_output_selector config)

add_custom_command(TARGET output_selector
                   POST_BUILD
                   COMMAND date +%s%3N > ${CMAKE_CURRENT_BINARY_DIR}/times/end.log 
                   COMMAND bash ${PROJECT_SOURCE_DIR}/cmake/show_build_time.sh ${CMAKE_CURRENT_BINARY_DIR}/times/start.log ${CMAKE_CURRENT_BINARY_DIR}/times/end.log "output_selector"
                   BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/times/end.log 
                   VERBATIM
)
                   
                   
#Add an alias so that library can be used inside the build tree
add_library(LBM_saclay::output_selector ALIAS output_selector)
# "$(date +%s)"
