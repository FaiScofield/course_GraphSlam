
#define USE_G2O 0
#define TRANS_DIMENSION_6 1

#include "utility.hpp"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if USE_G2O
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/slam2d/types_slam2d.h>

namespace g2o
{
typedef Eigen::Matrix<double, 9, 1> Vector9D;
class VertexSE2Trans : public BaseVertex<9, Vector9D>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexSE2Trans() {};

    void setToOriginImpl() override {
        _estimate << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    }

    void oplusImpl(const double* update) override
    {
        Eigen::Map<const Vector9D> v(update);
        _estimate += v;
    }

    bool read(std::istream& is) override { return false; }
    bool write(std::ostream& os) const override { return false; }
};

class EdgeSE2Trans : public BaseUnaryEdge<3, Eigen::Vector3d, VertexSE2Trans>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeSE2Trans() {};

    void setGT(const Eigen::Vector3d& gt) {
        _gt = gt;
    }

    void computeError() override
    {
//        Eigen::Matrix3d E = _measurement * _measurement.transpose();
//        Eigen::Matrix3d Ae = _gt * _measurement.transpose() * E.inverse();
//        Eigen::Map<Vector9D> A(Ae.data());
//        _measurement = A;

        const VertexSE2Trans* v1 = dynamic_cast<const VertexSE2Trans*>(_vertices[0]);
        Vector9D Ae = v1->estimate();
        Eigen::Map<const Eigen::Matrix3d> A(Ae.data());
        _error = _gt - A * _measurement;
    }

    void linearizeOplus() override
    {
        _jacobianOplusXi.setZero();
        _jacobianOplusXi.block<1, 3>(0, 0) = -_measurement.transpose();
        _jacobianOplusXi.block<1, 3>(1, 3) = -_measurement.transpose();
        _jacobianOplusXi.block<1, 3>(2, 6) = -_measurement.transpose();
    }

    bool read(std::istream& is) override { return false; }
    bool write(std::ostream& os) const override { return false; }

    Eigen::Vector3d _gt;
};

}  // namespace g2o

#endif

using namespace std;

const string g_calibData = "/home/vance/ddu_ws/course_GraphSlam/GraphSlamCpp/data/calib.dat";


bool readData(const string& dataFile, vector<Eigen::Vector3d>& vdOdom1, vector<Eigen::Vector3d>& vdOdom2)
{
    ifstream ifs(dataFile);
    if (!ifs.is_open()) {
        std::cerr << "Error on openning data file, please check it: " << dataFile << std::endl;
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

#if USE_G2O
    vector<g2o::SE2> tmpPose(N + 1);
    tmpPose[0] = g2o::SE2(0, 0, 0);
    for (size_t i = 0; i < N; ++i) {
        tmpPose[i + 1] = tmpPose[i] * g2o::SE2(vdOdom[i]);
        vOdomData[i + 1] = tmpPose[i + 1].toVector();
    }
#else
    vector<Eigen::Matrix3d> tmpPose(N + 1);
    tmpPose[0].setIdentity();
    for (size_t i = 0; i < N; ++i) {
        tmpPose[i + 1] = tmpPose[i] * VectorToMatrix(vdOdom[i]);
        vOdomData[i + 1] = MatrixToVector(tmpPose[i + 1]);
    }
#endif
    return vOdomData.size() == vdOdom.size() + 1;
}

bool saveData(const string& outFile, const vector<Eigen::Vector3d>& vOdomEsitimate,
              const vector<Eigen::Vector3d>& vGroundTruth,
              const Eigen::Matrix3d& A = Eigen::Matrix3d::Identity())
{
    const size_t N = vGroundTruth.size();
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

Eigen::Matrix3d solveWithSVD(const vector<Eigen::Vector3d>& vOdomEsitimate, const vector<Eigen::Vector3d>& vGroundTruth)
{
    assert(vOdomEsitimate.size() == vGroundTruth.size());
    size_t N = vOdomEsitimate.size();

    // initial guess
    Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d dA = Eigen::Matrix3d::Zero();

    // iteration
    double lastSum = 9999999.;
    int iter = 0;
    for (; iter < 10; ++iter) {
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
#if TRANS_DIMENSION_6
#else
            J.block<1, 3>(2, 6) = -vOdomEsitimate[i].transpose();
#endif
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

        cout << " iter = " << iter << ", sum error = " << sum << ", A = " << endl << A << endl;
        if (lastSum - sum < 1e-6 && iter > 0)
            break;
        lastSum = sum;
    }

    return A;
}

Eigen::Matrix3d solveWithSVD_SE2(const vector<Eigen::Vector3d>& vOdomEsitimate, const vector<Eigen::Vector3d>& vGroundTruth)
{
    assert(vOdomEsitimate.size() == vGroundTruth.size());
    size_t N = vOdomEsitimate.size();

    // initial guess
    Eigen::Matrix3d A = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d dA = Eigen::Matrix3d::Identity();

    // iteration
    double lastSum = 9999999.;
    int iter = 0;
    for (; iter < 100; ++iter) {
        Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
        Eigen::Vector3d b = Eigen::Vector3d::Zero();
        double sum = 0;
        for (size_t i = 0; i < N; ++i) {
            const Eigen::Matrix2d R = A.block<2, 2>(0, 0);
            const double angle = atan2(R(1, 0), R(0, 0));
            const Eigen::Vector2d t1 = vGroundTruth[i].head<2>();
            const Eigen::Vector2d t2 = vOdomEsitimate[i].head<2>();
            Eigen::Vector2d dt = t1 - R * t2 - A.block<2, 1>(0, 2);
            Eigen::Vector3d e;
            e.head<2>() = dt;
            e(2) = vGroundTruth[i](2) - vOdomEsitimate[i](2) - angle;
            sum += e.norm();

            const double x1 = vOdomEsitimate[i](0);
            const double y1 = vOdomEsitimate[i](1);
            Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
            J.block<2, 2>(0, 0) = -Eigen::Matrix2d::Identity();
            J(0, 2) = sin(angle) * x1 + cos(angle) * y1;
            J(1, 2) = sin(angle) * y1 - cos(angle) * x1;
            J(2, 2) = -1;

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

        Eigen::Matrix3d H_inv = V * S_inv * U.transpose();
        Eigen::Vector3d delta_x = -H_inv * b;

        Eigen::Rotation2Dd dR(delta_x(2));
        dA.block<2, 2>(0, 0) = dR.toRotationMatrix();
        dA.block<2, 1>(0, 2) = delta_x.block<2, 1>(0, 0);
        A = dA * A;

        cout << " iter = " << iter << ", sum error = " << sum << ", A = " << endl << A << endl;
        if (lastSum - sum < 1e-6 && iter > 0)
            break;
        lastSum = sum;
    }

    return A;
}

#if USE_G2O
Eigen::Matrix3d solveWithG2O(const vector<Eigen::Vector3d>& vOdomEsitimate, const vector<Eigen::Vector3d>& vGroundTruth)
{
    assert(!vOdomEsitimate.empty() && !vGroundTruth.empty());
    size_t N = vOdomEsitimate.size();

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 3>> BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    g2o::SparseOptimizer optimizer;
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    // vertices
    g2o::VertexSE2Trans* v0 = new g2o::VertexSE2Trans();
    v0->setId(0);
    v0->setFixed(false);
    g2o::Vector9D est;
    est << 1, 0, 0, 0, 1, 0, 0, 0, 1;
    v0->setEstimate(est);
//    g2o::VertexSE2* v0 = new g2o::VertexSE2();
//    v0->setId(0);
//    v0->setFixed(false);
//    v0->setEstimate(g2o::SE2(0, 0, 0));
    optimizer.addVertex(v0);
//    for (size_t i = 0; i < N; ++i) {
//        g2o::VertexSE2* v = new g2o::VertexSE2();
//        v->setId(i);
//        v->setFixed(false);
//        v->setEstimate(g2o::SE2(vOdomEsitimate[i]));
//        optimizer.addVertex(v);
//    }

    // edges
    for (size_t i = 1; i < N; ++i) {
//        g2o::EdgeSE2Prior* e = new g2o::EdgeSE2Prior();
//        e->setVertex(0, v0);
//        e->setMeasurement(g2o::SE2(vDeltaGT[i]));
//        e->setInformation(Eigen::Matrix3d::Identity());
//        optimizer.addEdge(e);

//        auto v1 = dynamic_cast<g2o::VertexSE2*>(optimizer.vertex(i - 1));
//        auto v2 = dynamic_cast<g2o::VertexSE2*>(optimizer.vertex(i));
//        g2o::EdgeSE2* e = new g2o::EdgeSE2();
//        e->setVertex(0, v1);
//        e->setVertex(1, v2);
//        e->setMeasurement(g2o::SE2(vDeltaGT[i]));
//        e->setInformation(Eigen::Matrix3d::Identity());
//        optimizer.addEdge(e);
        g2o::EdgeSE2Trans* e = new g2o::EdgeSE2Trans();
        e->setVertex(0, v0);
        e->setGT(vGroundTruth[i]);
        e->setMeasurement(vOdomEsitimate[i]);
        optimizer.addEdge(e);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    g2o::Vector9D Ae = v0->estimate();
    Eigen::Map<const Eigen::Matrix3d> A(Ae.data());
    return A;
}
#endif

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

#if USE_G2O
    Eigen::Matrix3d A = solveWithG2O(vOdomEsitimate, vGroundTruth);
#else
    //Eigen::Matrix3d A = solveWithSVD(vOdomEsitimate, vGroundTruth);
    Eigen::Matrix3d A = solveWithSVD_SE2(vOdomEsitimate, vGroundTruth);
#endif

    saveData("calib_after.txt", vOdomEsitimate, vGroundTruth, A);

    plotTrajectory("traj_before.gp", "./calib_before.txt");
    plotTrajectory("traj_after.gp", "./calib_after.txt");

    return 0;
}
