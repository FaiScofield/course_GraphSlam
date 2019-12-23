
//#include <g2o/core/optimization_algorithm_gauss_newton.h>
//#include <g2o/core/optimization_algorithm_levenberg.h>
//#include <g2o/core/sparse_optimizer.h>
//#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
//#include <g2o/solvers/dense/linear_solver_dense.h>
//#include <g2o/core/block_solver.h>
// #include <g2o/types/slam2d/types_slam2d.h>
#include "utility.hpp"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

const string g_calibData = "/home/vance/ddu_ws/course_GraphSlam/GraphSlamCpp/data/calib.dat";


bool readData(const string& dataFile, vector<Eigen::Vector3d>& vdOdom1, vector<Eigen::Vector3d>& vdOdom2)
{
    ifstream ifs(dataFile);
    if (!ifs.is_open()) {
        cerr << "Error on openning data file, please check it: " << dataFile << endl;
        return false;
    }

    vector<Eigen::Vector3d> data1, data2;
    data1.reserve(610);
    data2.reserve(610);
    while (ifs.peek() != EOF) {
        string line;
        getline(ifs, line);
        stringstream lineStream(line);
        Eigen::Vector3d o1, o2;
        lineStream >> o1[0] >> o1[1] >> o1[2] >> o2[0] >> o2[1] >> o2[2];
        data1.push_back(o1);
        data2.push_back(o2);
    }
    vdOdom1 = data1;
    vdOdom2 = data2;

    return (!data1.empty()) && (!data2.empty());
}

bool analyseData(const vector<Eigen::Vector3d>& vdOdom, vector<Eigen::Vector3d>& vOdomData)
{
    size_t N = vdOdom.size();

    vOdomData.clear();
    vOdomData.resize(N + 1, Eigen::Vector3d::Zero());

    vector<Eigen::Matrix3d> tmpPose(N + 1);
    tmpPose[0].setIdentity();
    for (size_t i = 0; i < N; ++i) {
        tmpPose[i + 1] = tmpPose[i] * VectorToMatrix(vdOdom[i]);
        vOdomData[i + 1] = MatrixToVector(tmpPose[i + 1]);
    }

    return vOdomData.size() == vdOdom.size() + 1;
}

bool saveData(const string& outFile, const vector<Eigen::Vector3d>& vOdomEsitimate,
              const vector<Eigen::Vector3d>& vGroundTruth,
              const Eigen::Matrix3d& A = Eigen::Matrix3d::Identity())
{
    size_t N = vGroundTruth.size();
    ofstream ofs(outFile);
    if (!ofs.is_open()) {
        cerr << "Error on openning data file, please check it: " << outFile << endl;
        return false;
    }

    for (size_t i = 0; i < N; ++i) {
        const Eigen::Matrix3d oem = A * VectorToMatrix(vOdomEsitimate[i]);
        const Eigen::Vector3d oev = MatrixToVector(oem);
        ofs << oev.transpose() << " " << vGroundTruth[i].transpose() << endl;
    }

    ofs.close();
    cout << "Save trajectory to the file: " << outFile << endl;

    return true;
}


// void solveWithG2O(const vector<Eigen::Vector3d>& vOdomEsitimate, const vector<Eigen::Vector3d>& vGroundTruth)
//{
//    assert(!vOdomEsitimate.empty() && !vGroundTruth.empty());

//    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType;
//    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

//    g2o::SparseOptimizer optimizer;
//    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
//      g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
//    optimizer.setAlgorithm(solver);
//    optimizer.setVerbose(true);

//    for (size_t i = 0, iend = vOdomEsitimate.size(); i < iend; ++i) {
//        g2o::VertexSE2* v = new g2o::VertexSE2();
//        v->setId(i);
//        v->setEstimate(g2o::SE2(vOdomEsitimate[i]));
//        optimizer.addVertex(v);
//    }


//}

/**
 * @brief solveWithSVD
 * @param vOdomEsitimate
 * @param vGroundTruth
 * @description
 *
 */
Eigen::Matrix3d
solveWithSVD(const vector<Eigen::Vector3d>& vOdomEsitimate, const vector<Eigen::Vector3d>& vGroundTruth)
{
    assert(vOdomEsitimate.size() == vGroundTruth.size());
    size_t N = vOdomEsitimate.size();

    // initial guess
    Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d dA = Eigen::Matrix3d::Zero();

    // iteration
    double lastSum = 9999999.;
    int it = 0;
    for (; it < 10; ++it) {
        Eigen::Matrix<double, 9, 9> H;
        Eigen::Matrix<double, 9, 1> b;
        H.setZero();
        b.setZero();
        double sum = 0;
        for (size_t i = 0; i < N; ++i) {
            Eigen::Vector3d e = vGroundTruth[i] - A * vOdomEsitimate[i];
            sum += e.norm();

            Eigen::Matrix<double, 3, 9> J;
            J.setZero();
            J.block<1, 3>(0, 0) = -vOdomEsitimate[i].transpose();
            J.block<1, 3>(1, 3) = -vOdomEsitimate[i].transpose();
            J.block<1, 3>(2, 6) = -vOdomEsitimate[i].transpose();
            H += J.transpose() * J;
            b += J.transpose() * e;
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        const Eigen::MatrixXd& U = svd.matrixU();
        const Eigen::MatrixXd& V = svd.matrixV();
        const Eigen::MatrixXd& S = svd.singularValues();
        Eigen::MatrixXd S_inv(V.cols(), U.cols());
        S_inv.setZero();
        for (int j = 0; j < S.size(); ++j) {
            if (S(j, 0) > 0)
                S_inv(j, j) = 1 / S(j, 0);
        }

        Eigen::Matrix<double, 9, 9> H_inv = V * S_inv * U.transpose();
        Eigen::Matrix<double, 9, 1> delta_x = -H_inv * b;
        dA.block<1, 3>(0, 0) = delta_x.block<3, 1>(0, 0);
        dA.block<1, 3>(1, 0) = delta_x.block<3, 1>(3, 0);
        dA.block<1, 3>(2, 0) = delta_x.block<3, 1>(6, 0);
        A += dA;

        cout << " it = " << it << ", sum = " << sum << ", A = " << endl << A << endl;
        if (lastSum - sum < 1e-6 && it > 0)
            break;
        lastSum = sum;
    }

    return A;
}


int main(int argc, char** argv)
{
    vector<Eigen::Vector3d> vdOdom1, vdOdom2;
    bool ok = readData(g_calibData, vdOdom1, vdOdom2);
    if (!ok) {
        cerr << "No odom datas in the file: " << g_calibData << endl;
        exit(-1);
    } else {
        cout << "Read " << vdOdom1.size() << " datas in the file." << endl;
    }

    vector<Eigen::Vector3d> vOdomEsitimate, vGroundTruth;
    bool b1 = analyseData(vdOdom1, vOdomEsitimate);
    bool b2 = analyseData(vdOdom2, vGroundTruth);
    if (b1 && b2)
        saveData("calib_before.txt", vOdomEsitimate, vGroundTruth);

    Eigen::Matrix3d A = solveWithSVD(vOdomEsitimate, vGroundTruth);
    saveData("calib_after.txt", vOdomEsitimate, vGroundTruth, A);

    plotTrajectory("traj_before.gp", "./calib_before.txt");
    plotTrajectory("traj_after.gp", "./calib_after.txt");

    return 0;
}
