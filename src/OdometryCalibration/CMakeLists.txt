add_executable(OdometryCalibration OdometryCalibration.cpp utility.hpp)
target_link_libraries(OdometryCalibration
    ${G2O_LIBS}
    ${CHOLMOD_LIBRARIES}
)
