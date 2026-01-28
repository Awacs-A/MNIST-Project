#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <fstream>
#include <algorithm>
#include <random>
#include <Eigen/Dense>
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255; c2 = (i >> 8) & 255; c3 = (i >> 16) & 255; c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
class NeuralNetwork {
public:
    Matrix W1, W2, W3;
    Vector b1, b2, b3;
    Vector z1, a1, z2, a2, z3, a3;
    // Accumulators for Batch Averaging
    Matrix dW1, dW2, dW3;
    Vector db1, db2, db3;
    NeuralNetwork(int inputsize, int h1, int h2, int outputsize) {
        W1 = Matrix::Random(h1, inputsize) * std::sqrt(2.0 / inputsize);
        b1 = Vector::Zero(h1);

        W2 = Matrix::Random(h2, h1) * std::sqrt(2.0 / h1);
        b2 = Vector::Zero(h2);

        W3 = Matrix::Random(outputsize, h2) * std::sqrt(2.0 / h2);
        b3 = Vector::Zero(outputsize);
        dW1 = Matrix::Zero(h1, inputsize);
        db1 = Vector::Zero(h1);
        dW2 = Matrix::Zero(h2, h1);
        db2 = Vector::Zero(h2);
        dW3 = Matrix::Zero(outputsize, h2);
        db3 = Vector::Zero(outputsize);
    }
    Vector forward(const Vector& x) {
        z1 = (W1 * x) + b1;
        a1 = z1.unaryExpr([](double v) { return std::max(0.0, v); });// ReLU activation
        z2 = (W2 * a1) + b2;
        a2 = z2.unaryExpr([](double v) { return std::max(0.0, v); });// ReLU activation

        z3 = (W3 * a2) + b3;
        a3 = z3.unaryExpr([](double v) { return 1.0 / (1.0 + std::exp(-v)); });// Sigmoid Activation
        return a3;// The networks final guess of the number
    }
    Vector SigmoidDerivative(const Vector& x) {
        return x.cwiseProduct(Vector::Ones(x.size()) - x);
    }
    Vector ReLUDerivative(const Vector& x) {
        return (x.array() > 0).cast<double>().matrix();
    }

    void backward(const Vector& x, const Vector& target) {
        Vector delta3 = (a3 - target).cwiseProduct(SigmoidDerivative(a3));
        Vector delta2 = (W3.transpose() * delta3).cwiseProduct(ReLUDerivative(z2));
        Vector delta1 = (W2.transpose() * delta2).cwiseProduct(ReLUDerivative(z1));
        
        dW3 += (delta3 * a2.transpose());
        db3 += delta3;
        dW2 += (delta2 * a1.transpose());
        db2 += delta2;
        dW1 += (delta1 * x.transpose());
        db1 += delta1;
    }
    void applyGradients(double learningRate, int batchSize) {
        double factor = learningRate / batchSize;
        W1 -= factor * dW1; b1 -= factor * db1;
        W2 -= factor * dW2; b2 -= factor * db2;
        W3 -= factor * dW3; b3 -= factor * db3;

        dW1.setZero(); db1.setZero();
        dW2.setZero(); db2.setZero();
        dW3.setZero(); db3.setZero();
    }
    void save(const std::string& filename) {
        std::ofstream out(filename, std::ios::binary);
        if (out.is_open()) {
            auto writeMatrix = [&](const Matrix& m) {
                long rows = m.rows(), cols = m.cols();
                out.write((char*)&rows, sizeof(long));
                out.write((char*)&cols, sizeof(long));
                out.write((char*)m.data(), rows * cols * sizeof(double));
            };

            writeMatrix(W1); writeMatrix(b1);
            writeMatrix(W2); writeMatrix(b2);
            writeMatrix(W3); writeMatrix(b3);
            out.close();
        }
    }
    void load(const std::string& filename) {
        std::ifstream in(filename, std::ios::binary);
        if (in.is_open()) {
            auto readMatrix = [&](auto& m) {
                long rows, cols;
                in.read((char*)&rows, sizeof(long));
                in.read((char*)&cols, sizeof(long));
                m.resize(rows, cols);
                in.read((char*)m.data(), rows * cols * sizeof(double));
            };

            readMatrix(W1); readMatrix(b1);
            readMatrix(W2); readMatrix(b2);
            readMatrix(W3); readMatrix(b3);
            in.close();
        }
    }
};
std::vector<Vector> loadImages(std::string path) {
    std::ifstream file(path, std::ios::binary);
    int magic = 0, n = 0, r = 0, c = 0;
    file.read((char*)&magic, 4);
    file.read((char*)&n, 4);
    file.read((char*)&r, 4);
    file.read((char*)&c, 4);
    n = reverseInt(n);
    std::vector<Vector> imgs(n, Vector(784));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 784; j++) {
            unsigned char pixel = 0;
            file.read((char*)&pixel, 1);
            imgs[i](j) = pixel / 255.0;
        }
    }
    return imgs;
}
//Load the answers to the training data
std::vector<Vector> loadLabels(std::string path) {
    std::ifstream file(path, std::ios::binary);
    int magic = 0, n = 0;
    file.read((char*)&magic, 4); file.read((char*)&n, 4);
    n = reverseInt(n);
    std::vector<Vector> lbls(n, Vector::Zero(10));
    for (int i = 0; i < n; i++) {
        unsigned char label = 0;
        file.read((char*)&label, 1);
        lbls[i]((int)label) = 1.0;
    }
    return lbls;
}
int main(){
    std::vector<Vector> train_images = loadImages("train-images-idx3-ubyte");
    std::vector<Vector> train_labels = loadLabels("train-labels-idx1-ubyte");
    int batchsize = 32;
    double learningrate = 0.05;
    int epochs = 5;
    NeuralNetwork nn(784,32,16,10);
    for (int e{}; e < epochs; e++) {
        for (int i{}; i < (int)train_images.size(); i += batchsize) {
            int current_batch = std::min(batchsize, (int)train_images.size() - i);
            
            for (int b{}; b < current_batch; b++) {
                nn.forward(train_images[i + b]);
                nn.backward(train_images[i + b], train_labels[i + b]);
            }
            nn.applyGradients(learningrate, current_batch);
        }
    }
    nn.save("mnist_model.bin");
    return 0;
}
