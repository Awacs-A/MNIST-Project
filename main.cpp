#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <random>
#include <numeric>
#include <Eigen/Dense>
#include <vector>
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Index = Eigen::Index;
//98.9%
int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255; c2 = (i >> 8) & 255; c3 = (i >> 16) & 255; c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
class NeuralNetwork {
public:
    std::vector<Matrix> W, mW, vW;
    std::vector<Vector> b, mb, vb;


    // Values stored during forward pass for backpropagation
    std::vector<Vector> z, a;
    int BatchSize;
    // Gradient accumulators
    std::vector<Matrix> dW;
    std::vector<Vector> db;
    int t{};
    NeuralNetwork(const std::vector<int>& layers, int bt) {
        for (size_t i = 0; i < layers.size() - 1; ++i) {
            int in = layers[i], out = layers[i + 1];

            W.push_back(Matrix::Random(out, in) * std::sqrt(2.0 / in));
            b.push_back(Vector::Zero(out));

            // Initialize Adam moments to zero
            mW.push_back(Matrix::Zero(out, in));
            vW.push_back(Matrix::Zero(out, in));
            mb.push_back(Vector::Zero(out));
            vb.push_back(Vector::Zero(out));

            dW.push_back(Matrix::Zero(out, in));
            db.push_back(Vector::Zero(out));
        }
        z.resize(W.size());
        a.resize(W.size());
        BatchSize = bt;
    }
    Vector forward(const Vector& x) {
        for (size_t i = 0; i < W.size(); ++i) {
            const Vector& prev_a = (i == 0) ? x : a[i - 1];
            z[i] = (W[i] * prev_a) + b[i];

            if (i == W.size() - 1) {
                // Softmax for the output layer
                double max_val = z[i].maxCoeff();
                Vector exp_z = (z[i].array() - max_val).exp();
                a[i] = exp_z / exp_z.sum();
            }
            else {
                a[i] = z[i].unaryExpr([](double v){return (v > 0) ? v : 0.01 * v;});
            }
        }
        return a.back();
    }
    Vector ReLUDerivative(const Vector& v) {
        return (v.array() > 0).select(Vector::Ones(v.size()), Vector::Constant(v.size(), 0.01));
    }
    void backward(const Vector& x, const Vector& target) {
        Vector delta = a.back() - target;

        for (int i = W.size() - 1; i >= 0; --i) {
            const Vector& prev_a = (i == 0) ? x : a[i - 1];
            // Calculate gradients
            dW[i] += (delta * prev_a.transpose()) / BatchSize;
            db[i] += delta / BatchSize;

            if (i > 0) {
                delta = (W[i].transpose() * delta).cwiseProduct(ReLUDerivative(z[i - 1]));
            }
        }
    }
    void applyGradients(double learningRate) {
        t++;
        const double beta1 = 0.9;
        const double beta2 = 0.999;
        const double eps = 1e-8;     // Standard Epsilon
        const double lambda = 0.0001; // L2 Regularization

        for (size_t i = 0; i < W.size(); ++i) {
            // Update moments
            mW[i] = beta1 * mW[i] + (1.0 - beta1) * dW[i];
            vW[i] = beta2 * vW[i] + (1.0 - beta2) * dW[i].cwiseProduct(dW[i]);

            mb[i] = beta1 * mb[i] + (1.0 - beta1) * db[i];
            vb[i] = beta2 * vb[i] + (1.0 - beta2) * db[i].cwiseProduct(db[i]);

            // Bias Correction
            double m_corr = 1.0 / (1.0 - std::pow(beta1, t));
            double v_corr = 1.0 / (1.0 - std::pow(beta2, t));

            W[i].array() -= learningRate * (
                (mW[i].array() * m_corr) / ((vW[i].array() * v_corr).sqrt() + eps) +
                (lambda * W[i].array())
                );

            b[i].array() -= learningRate * (
                (mb[i].array() * m_corr) / ((vb[i].array() * v_corr).sqrt() + eps)
                );

            // Reset Gradients
            dW[i].setZero();
            db[i].setZero();
        }
    }
    void save(const std::string& filename) {
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs.is_open()) return;

        for (size_t i = 0; i < W.size(); ++i) {
            // Save raw weight data
            ofs.write((char*)W[i].data(), W[i].size() * sizeof(double));
            // Save raw bias data
            ofs.write((char*)b[i].data(), b[i].size() * sizeof(double));

            ofs.write((char*)mW[i].data(), mW[i].size() * sizeof(double));
            ofs.write((char*)vW[i].data(), vW[i].size() * sizeof(double));
            ofs.write((char*)mb[i].data(), mb[i].size() * sizeof(double));
            ofs.write((char*)vb[i].data(), vb[i].size() * sizeof(double));
        }
        ofs.write((char*)&t, sizeof(int)); // Save current Adam timestep
        ofs.close();
    }

    void load(const std::string& filename) {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs.is_open()) return;

        for (size_t i = 0; i < W.size(); ++i) {
            // Read raw data directly into the existing allocated memory
            ifs.read((char*)W[i].data(), W[i].size() * sizeof(double));
            ifs.read((char*)b[i].data(), b[i].size() * sizeof(double));

            // Load Adam moments
            ifs.read((char*)mW[i].data(), mW[i].size() * sizeof(double));
            ifs.read((char*)vW[i].data(), vW[i].size() * sizeof(double));
            ifs.read((char*)mb[i].data(), mb[i].size() * sizeof(double));
            ifs.read((char*)vb[i].data(), vb[i].size() * sizeof(double));
        }
        ifs.read((char*)&t, sizeof(int)); // Restore Adam timestep
        ifs.close();
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
            imgs[i](j) = (pixel / 255.0) - 0.5;
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
Vector augment(const Vector& img) {
    static std::mt19937 gen(std::random_device{}());
    static std::uniform_int_distribution<> shift(-1, 1);

    int dx = shift(gen);
    int dy = shift(gen);

    if (dx == 0 && dy == 0) return img;

    Vector shifted = Vector::Constant(784, -0.5); // Fill with "background" color (-0.5)

    for (int r = 0; r < 28; ++r) {
        for (int c = 0; c < 28; ++c) {
            int new_r = r + dy;
            int new_c = c + dx;

            if (new_r >= 0 && new_r < 28 && new_c >= 0 && new_c < 28) {
                shifted(new_r * 28 + new_c) = img(r * 28 + c);
            }
        }
    }
    return shifted;
}
int main() {
    int n; std::cin >> n;
    std::vector<Vector> train_images = loadImages("train-images.idx3-ubyte");
    std::vector<Vector> train_labels = loadLabels("train-labels.idx1-ubyte");

    std::vector<Vector> test_images = loadImages("t10k-images.idx3-ubyte");
    std::vector<Vector> test_labels = loadLabels("t10k-labels.idx1-ubyte");
    int batchsize = 128;
    double learningrate = 0.001;
    int epochs = 25;
    NeuralNetwork nn({ 784, 512, 128, 10 }, batchsize);
    if (!n) {
        std::vector<int> indices(train_images.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::random_device rng;
        std::mt19937 g(rng());

        std::cout << "Starting Training..." << '\n';
        for (int e = 0; e < epochs; ++e) {
            std::shuffle(indices.begin(), indices.end(), g);

            for (int i = 0; i < (int)train_images.size(); i += batchsize) {
                int current_batch = std::min(batchsize, (int)train_images.size() - i);

                for (int b = 0; b < current_batch; ++b) {
                    int idx = indices[i + b];

                    // Augmentation to improve accuraccy
                    Vector augmented_img = augment(train_images[idx]);

                    nn.forward(augmented_img);
                    nn.backward(augmented_img, train_labels[idx]);
                }
                nn.applyGradients(learningrate);
            }

            int correct = 0;
            for (int i = 0; i < (int)test_images.size(); ++i) {
                Vector output = nn.forward(test_images[i]);
                int prediction, actual;
                output.maxCoeff(&prediction);
                test_labels[i].maxCoeff(&actual);
                if (prediction == actual) correct++;
            }

            double acc = (double)correct / test_images.size() * 100.0;
            std::cout << "Epoch " << e + 1 << " Accuracy: " << acc << "%" << std::endl;

            // Cut learning rate in half every 5 epochs to fine-tune
            if ((e + 1) % 5 == 0) learningrate /= 2;
        }

        nn.save("mnist_model.bin");
        std::cout << "Training complete. Current model saved" << '\n';
    }
    else {
        nn.load("mnist_model.bin");

        std::cout << "Testing loaded model..." << '\n';
        std::vector<Vector> test_images = loadImages("t10k-images.idx3-ubyte");
        std::vector<Vector> test_labels = loadLabels("t10k-labels.idx1-ubyte");

        int correct = 0;
        for (int i = 0; i < (int)test_images.size(); i++) {
            Vector output = nn.forward(test_images[i]);
            int prediction, actual;
            output.maxCoeff(&prediction);
            test_labels[i].maxCoeff(&actual);
            if (prediction == actual) correct++;
        }
        std::cout << "Loaded Model Accuracy: " << (double)correct / test_images.size() * 100.0 << "%" << '\n';
    }
    return 0;
}
