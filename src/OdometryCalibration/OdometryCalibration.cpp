
#define USE_G2O 0

#include "utility.hpp"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/SVD>

using namespace std;
using namespace Eigen;

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

class VertexSE2Trans : public BaseVertex<3, Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VertexSE2Trans() {}

    void setToOriginImpl() override {
        _estimate << 0, 0, 0;
    }

    void oplusImpl(const double* update) override
    {
        Eigen::Map<const Vector3d> v(update);
        Matrix2d R;
        R << cos(_estimate[2]), -sin(_estimate[2]), sin(_estimate[2]), cos(_estimate[2]);
        _estimate.head<2>() += R * v.head<2>();
        _estimate[2] += v[2];
        _estimate[2] = normalize_theta(_estimate[2]);
    }

    bool read(std::istream& is) override { return false; }
    bool write(std::ostream& os) const override { return false; }
};

class EdgeSE2Trans : public BaseUnaryEdge<3, Eigen::Vector3d, VertexSE2Trans>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE2Trans() {}

    void setGT(const Eigen::Vector3d& gt) {
        _gt = gt;
    }

    void computeError() override
    {
        const VertexSE2Trans* v1 = dynamic_cast<const VertexSE2Trans*>(_vertices[0]);
        const Vector3d Te = v1->estimate();
        Matrix2d Re;
        Re << cos(Te[2]), -sin(Te[2]), sin(Te[2]), cos(Te[2]);

        _error.head<2>() = _gt.head<2>() - Re * _measurement.head<2>() - Te.head<2>();
        _error[2] = _gt[2] - _measurement[2] - Te[2];
        _error[2] = normalize_theta(_error[2]);
    }

    // -[c -s]' = -[-s -c] = [s  c]
    //  [s  c]     [c  -s]   [-c s]
//    void linearizeOplus() override
//    {
//        const VertexSE2Trans* v1 = dynamic_cast<const VertexSE2Trans*>(_vertices[0]);
//        const Vector3d Te = v1->estimate();
//        _jacobianOplusXi.setZero();
//        _jacobianOplusXi << -1, 0, sin(Te[2])*_measurement[0] + cos(Te[2])*_measurement[1],
//                0, -1, -cos(Te[2])*_measurement[0] + sin(Te[2])*_measurement[1], 0, 0, -1;
//    }

    bool read(std::istream& is) override { return false; }
    bool write(std::ostream& os) const override { return false; }

    Eigen::Vector3d _gt;
};

}  // namespace g2o

#endif


const string g_calibData = "/home/vance/ddu_ws/course_GraphSlam/GraphSlamCpp/data/calib.dat";


bool readData(const string& dataFile, vector<Eigen::Vector3d>& vdOdom1, vector<Eigen::Vector3d>& vdOdom2)
{
    ifstream ifs(dataFile);
    if (!ifs.is_open()) {
        std::cerr << "Error on openning data file, please check it: " << dataFile << std::endl;
        return false;
    }

    vector<Vector3d> data1, data2;
    data1.reserve(610);
    data2.reserve(610);
    while (ifs.peek() != EOF) {
        string line;
        getline(ifs, line);
        stringstream lineStream(line);
        Vector3d o1, o2;
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
    vOdomData.resize(N, Vector3d::Zero());

#if USE_G2O
    vector<g2o::SE2> tmpPose(N + 1);
    tmpPose[0] = g2o::SE2(0, 0, 0);
    for (size_t i = 0; i < N; ++i) {
        tmpPose[i + 1] = tmpPose[i] * g2o::SE2(vdOdom[i]);
        vOdomData[i + 1] = tmpPose[i + 1].toVector();
    }
#else
    Matrix3d tmpPose = Matrix3d::Identity();
    for (size_t i = 0; i < N; ++i) {
        tmpPose *= VectorToMatrix(vdOdom[i]);
        vOdomData[i] = MatrixToVector(tmpPose);
    }
#endif
    return vOdomData.size() == vdOdom.size();
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
        const Matrix3d oem = A * VectorToMatrix(vOdomEsitimate[i]);
        const Vector3d oev = MatrixToVector(oem);
        ofs << oev.transpose() << " " << vGroundTruth[i].transpose() << endl;
    }

    ofs.close();
    cout << "Save trajectory to the file: " << outFile << endl;

    return true;
}

Eigen::Matrix3d solveWithDLT(const vector<Eigen::Vector3d>& vOdomEsitimate, const vector<Eigen::Vector3d>& vGroundTruth)
{
    assert(vOdomEsitimate.size() == vGroundTruth.size());
    const size_t N = vOdomEsitimate.size();

    // initial guess
    Matrix3d A = Matrix3d::Identity();
    Matrix3d dA = Matrix3d::Zero();

    // iteration
    double lastCost = 9999999.;
    int iter = 0;
    for (; iter < 100; ++iter) {
        Matrix<double, 9, 9> H;
        Matrix<double, 9, 1> b;
        H.setZero();
        b.setZero();
        double currCost = 0;
        for (size_t i = 0; i < N; ++i) {
            Vector3d e = vGroundTruth[i] - A * vOdomEsitimate[i];
            currCost += e.norm();

            Matrix<double, 3, 9> J;
            J.setZero();
            J.block<1, 3>(0, 0) = -vOdomEsitimate[i].transpose();
            J.block<1, 3>(1, 3) = -vOdomEsitimate[i].transpose();
            J.block<1, 3>(2, 6) = -vOdomEsitimate[i].transpose();

            H += J.transpose() * J;
            b += J.transpose() * e;
        }

        // Matrix<double, 9, 1> delta_x = H.ldlt().solve(-b);
        Matrix<double, 9, 1> delta_x = -H.inverse() * b;
        if (isnan(delta_x[0])) {
            cerr << "result is nan!" << endl;
            break;
        }
        if (iter > 0 && currCost > lastCost) {
            cerr << "result increase!" << endl;
            break;
        }

        dA.block<1, 3>(0, 0) = delta_x.head(3).transpose();
        dA.block<1, 3>(1, 0) = delta_x.segment(3, 3).transpose();
        dA.block<1, 3>(2, 0) = delta_x.tail(3).transpose();
        A += dA;

        cout << " iter = " << iter << ", sum error = " << currCost << ", A = " << endl << A << endl;
        if (iter > 0 && lastCost - currCost < 1e-6)
            break;
        lastCost = currCost;
    }

    return A;
}

Eigen::Matrix3d solveWithSVD_SE2(const vector<Eigen::Vector3d>& vOdomEsitimate, const vector<Eigen::Vector3d>& vGroundTruth)
{
    assert(vOdomEsitimate.size() == vGroundTruth.size());
    const size_t N = vOdomEsitimate.size();

    // initial guess
    Vector3d A = Vector3d::Zero();

    // iteration
    double currCost = 0;
    double lastCost = 9999999.;
    int iter = 0;
    for (; iter < 100; ++iter) {
        const Matrix2d R = Rotation2Dd(A(2)).toRotationMatrix();

        Matrix3d H = Matrix3d::Zero();
        Vector3d b = Vector3d::Zero();
        currCost = 0;
        for (size_t i = 0; i < N; ++i) {
            const Vector2d t1 = vGroundTruth[i].head(2);
            const Vector2d t2 = vOdomEsitimate[i].head(2);
            Vector3d e;
            e.head(2) = t1 - R * t2 - A.head(2);
            e(2) = g2o::normalize_theta(vGroundTruth[i](2) - vOdomEsitimate[i](2) - A(2));
            currCost += e.norm();

            Matrix3d J = Matrix3d::Zero();
            J.block(0, 0, 2, 2) = -Matrix2d::Identity();
            J(0, 2) = sin(A(2)) * t2(0) + cos(A(2)) * t2(1);
            J(1, 2) = sin(A(2)) * t2(1) - cos(A(2)) * t2(0);
            J(2, 2) = -1;

            H += J.transpose() * J;
            b += J.transpose() * e;
        }

        Vector3d dA = H.ldlt().solve(-b);
        if (isnan(dA[0])) {
            cerr << "result is nan!" << endl;
            break;
        }
        if (iter > 0 && currCost > lastCost) {
            cerr << "result increase!" << endl;
            break;
        }

        A.head(2) += R * dA.head(2);
        A(2) += dA(2);
        A(2) = g2o::normalize_theta(A(2));

        cout << " iter = " << iter << ", sum error = " << currCost << ", A = " << A.transpose() << endl;
        if (iter > 0 && lastCost - currCost < 1e-6)
            break;
        lastCost = currCost;
    }

    return VectorToMatrix(A);
}

Eigen::Matrix3d solveWithICP(const vector<Eigen::Vector3d>& vOdomEsitimate, const vector<Eigen::Vector3d>& vGroundTruth)
{
    assert(vOdomEsitimate.size() == vGroundTruth.size());
    const size_t N = vOdomEsitimate.size();

    Vector2d gtCenter, esCenter;
    gtCenter.setZero();
    esCenter.setZero();
    for (size_t i = 0; i < N; ++i) {
        gtCenter += vGroundTruth[i].head<2>();
        esCenter += vOdomEsitimate[i].head<2>();
    }
    gtCenter /= N;
    esCenter /= N;

    Matrix2d W = Matrix2d::Zero();
    vector<Vector2d> vGts(N), vEst(N);
    for (size_t i = 0; i < N; ++i) {
        vGts[i] = vGroundTruth[i].head<2>() - gtCenter;
        vEst[i] = vOdomEsitimate[i].head<2>() - esCenter;
        W += vGts[i] * vEst[i].transpose();
    }
    cout << "W = " << endl << W << endl;

    JacobiSVD<Matrix2d> svd(W, ComputeFullU | ComputeFullV);
    const Matrix2d U = svd.matrixU();
    const Matrix2d V = svd.matrixV();

    Matrix2d R = U * V.transpose();
    if (R.determinant() < 0)
        R = -R;
    Vector2d t = gtCenter - R * esCenter;
    Matrix3d T = Matrix3d::Identity();
    T.block<2, 2>(0, 0) = R;
    T.block<2, 1>(0, 2) = t;
    return T;
}

#if USE_G2O
Eigen::Matrix3d solveWithG2O(const vector<Eigen::Vector3d>& vOdomEsitimate, const vector<Eigen::Vector3d>& vGroundTruth)
{
    assert(!vOdomEsitimate.empty() && !vGroundTruth.empty());
    const size_t N = vOdomEsitimate.size();

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 3>> BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

    g2o::SparseOptimizer optimizer;
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    // vertex
    g2o::VertexSE2Trans* v0 = new g2o::VertexSE2Trans();
    v0->setId(0);
    v0->setFixed(false);
    v0->setEstimate(Vector3d(0, 0, 0));
    optimizer.addVertex(v0);

    // edges
    for (size_t i = 1; i < N; ++i) {
        g2o::EdgeSE2Trans* e = new g2o::EdgeSE2Trans();
        e->setVertex(0, v0);
        e->setGT(vGroundTruth[i]);
        e->setMeasurement(vOdomEsitimate[i]);
        optimizer.addEdge(e);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(10);

    Vector3d Te = v0->estimate();
    Matrix3d Trans;
    Trans << cos(Te[2]), -sin(Te[2]), Te[0], sin(Te[2]), cos(Te[2]), Te[1], 0, 0, 1;

    return Trans;
}
#endif

int main(int argc, char** argv)
{
    vector<Vector3d> vdOdom1, vdOdom2;
    bool ok = readData(g_calibData, vdOdom1, vdOdom2);
    if (!ok) {
        cerr << "No odom datas in the file: " << g_calibData << endl;
        exit(-1);
    } else {
        cout << "Read " << vdOdom1.size() << " datas in the file." << endl;
    }

    vector<Vector3d> vOdomEsitimate, vGroundTruth;
    bool b1 = analyseData(vdOdom2, vOdomEsitimate);
    bool b2 = analyseData(vdOdom1, vGroundTruth);
    if (b1 && b2)
        saveData("calib_before.txt", vOdomEsitimate, vGroundTruth);

#if USE_G2O
    Matrix3d A = solveWithG2O(vOdomEsitimate, vGroundTruth);
#else
//    Matrix3d A = solveWithDLT(vOdomEsitimate, vGroundTruth);
    Matrix3d A = solveWithSVD_SE2(vOdomEsitimate, vGroundTruth);
//    Matrix3d A = solveWithICP(vOdomEsitimate, vGroundTruth);
#endif

    cout << "A = " << endl << A << endl;

    saveData("calib_after.txt", vOdomEsitimate, vGroundTruth, A);

    plotTrajectory("traj_before.gp", "./calib_before.txt");
    plotTrajectory("traj_after.gp", "./calib_after.txt");

    return 0;
}
