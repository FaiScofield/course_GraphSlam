#ifndef ODOMETRY_CALIBRATION_UTILITY_HPP
#define ODOMETRY_CALIBRATION_UTILITY_HPP

#include <fstream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <g2o/types/slam2d/types_slam2d.h>

// Eigen::Isometry3d / Isometry2d
Eigen::Matrix3d VectorToMatrix(const Eigen::Vector3d& pose)
{
    const double c = cos(pose(2)), s = sin(pose(2));
    Eigen::Matrix3d T;
    T << c, -s, pose(0), s, c, pose(1), 0, 0, 1;
    return T;
}

Eigen::Matrix3d SE2ToMatrix(const g2o::SE2& pose)
{
    return VectorToMatrix(pose.toVector());
}

Eigen::Vector3d MatrixToVector(const Eigen::Matrix3d& T)
{
    Eigen::Vector3d pose;
    pose[0] = T(0, 2);
    pose[1] = T(1, 2);
    pose[2] = std::atan2(T(1, 0), T(0, 0));

    return pose;
}

g2o::SE2 MatrixToSE2(const Eigen::Matrix3d& T)
{
    return g2o::SE2(MatrixToVector(T));
}

bool plotTrajectory(const std::string& plotFile, const std::string& inputFile)
{
    std::ofstream ofs(plotFile);
    if (!ofs.is_open()) {
        std::cerr << "Error in openning the output file: " << plotFile << std::endl;
        return false;
    }
    // set term
    ofs << "set term png size 900,900" << std::endl;
    ofs << "set output \"" << plotFile << ".png\"" << std::endl;
    ofs << "set size ratio -1" << std::endl;
    ofs << "set xrange [-60:100]" << std::endl;
    ofs << "set yrange [-80:40]" << std::endl;
    ofs << "set xlabel \"x [m]\"" << std::endl;
    ofs << "set ylabel \"y [m]\"" << std::endl;
    ofs << "plot \"" << inputFile
        << "\" using 1:2 with points lc rgb \"#000000\" pt 7 lw 2 ps 0.5 title 'Measurement',";
    ofs << "\"" << inputFile
        << "\" using 4:5 with points lc rgb \"#FF0000\" pt 7 lw 2 ps 0.5 title 'Ground Truth'" << std::endl;

    ofs.close();

    std::string command = "gnuplot " + plotFile;
    system(command.c_str());
    return true;
}

#endif
