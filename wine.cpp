#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>

using namespace std;

// Constants
const double C = 1.0; // Regularization parameter
const double tol = 1e-3; // Tolerance for errors
const double eps = 1e-3; // Small value for numerical comparisons

// Global variables
vector<vector<double>> points; // Training points
vector<int> target; // Labels (+1, -1)
vector<double> alpha; // Lagrange multipliers
vector<double> w; // Weight vector (for linear SVM)
double b = 0.0; // Bias
int numChanged; // Number of changes in each iteration

// Linear kernel (dot product)
double kernel(const vector<double>& x1, const vector<double>& x2) {
    double result = 0.0;
    for (size_t i = 0; i < x1.size(); i++)
        result += x1[i] * x2[i];
    return result;
}

// SVM output for a given point
double SVMOutput(int i) {
    double result = -b;
    for (size_t j = 0; j < points.size(); j++)
        result += alpha[j] * target[j] * kernel(points[i], points[j]);
    return result;
}

// Function to update the weight vector
void updateWeights(int i1, int i2, double oldAlpha1, double oldAlpha2) {
    double deltaAlpha1 = alpha[i1] - oldAlpha1;
    double deltaAlpha2 = alpha[i2] - oldAlpha2;

    for (size_t i = 0; i < w.size(); i++) {
        w[i] += target[i1] * deltaAlpha1 * points[i1][i] +
                target[i2] * deltaAlpha2 * points[i2][i];
    }
}

bool takeStep(int i1, int i2) {
    if (i1 == i2) return false;

    double alpha1 = alpha[i1], alpha2 = alpha[i2];
    int y1 = target[i1], y2 = target[i2];
    double E1 = SVMOutput(i1) - y1;
    double E2 = SVMOutput(i2) - y2;
    double s = y1 * y2;

    double L, H;
    if (y1 != y2) {
        L = max(0.0, alpha2 - alpha1);
        H = min(C, C + alpha2 - alpha1);
    } else {
        L = max(0.0, alpha2 + alpha1 - C);
        H = min(C, alpha2 + alpha1);
    }
    if (L == H) return false;

    double k11 = kernel(points[i1], points[i1]);
    double k12 = kernel(points[i1], points[i2]);
    double k22 = kernel(points[i2], points[i2]);
    double eta = k11 + k22 - 2 * k12;

    double a2;
    if (eta > 0) {
        a2 = alpha2 + y2 * (E1 - E2) / eta;
        if (a2 < L) a2 = L;
        else if (a2 > H) a2 = H;
    } else {
        return false;
    }

    if (abs(a2 - alpha2) < eps * (a2 + alpha2 + eps)) return false;

    double a1 = alpha1 + s * (alpha2 - a2);

    double oldAlpha1 = alpha1;
    double oldAlpha2 = alpha2;

    double b1 = b - E1 - y1 * (a1 - alpha1) * k11 - y2 * (a2 - alpha2) * k12;
    double b2 = b - E2 - y1 * (a1 - alpha1) * k12 - y2 * (a2 - alpha2) * k22;
    if (0 < a1 && a1 < C) b = b1;
    else if (0 < a2 && a2 < C) b = b2;
    else b = (b1 + b2) / 2;

    alpha[i1] = a1;
    alpha[i2] = a2;

    updateWeights(i1, i2, oldAlpha1, oldAlpha2);

    return true;
}

void SMO() {
    numChanged = 0;
    bool examineAll = true;

    while (numChanged > 0 || examineAll) {
        numChanged = 0;
        if (examineAll) {
            for (size_t i = 0; i < points.size(); i++)
                numChanged += takeStep(rand() % points.size(), i);
        } else {
            for (size_t i = 0; i < points.size(); i++)
                if (alpha[i] > 0 && alpha[i] < C)
                    numChanged += takeStep(rand() % points.size(), i);
        }

        if (examineAll) examineAll = false;
        else if (numChanged == 0) examineAll = true;
    }
}

void loadCSV(const string& filename, vector<vector<double>>& points, vector<int>& target) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }

    string line;

    getline(file, line);

    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        vector<double> point;
        int label;

        for (int i = 0; i < 11; i++) {
            getline(ss, value, ';');
            point.push_back(stod(value));
        }

        getline(ss, value, ';');
        int quality = stoi(value);

        // Transform quality into binary labels (+1 for good, -1 for not good)
        label = (quality >= 6) ? 1 : -1;

        points.push_back(point);
        target.push_back(label);
    }

    file.close();
}

int main() {
    loadCSV("winequality-red.csv", points, target);

    alpha = vector<double>(points.size(), 0.0);
    w = vector<double>(points[0].size(), 0.0);

    SMO();

    cout << "Final alpha values:" << endl;
    for (double a : alpha) cout << a << " ";
    cout << endl;

    cout << "Bias (b): " << b << endl;

    cout << "Weight vector (w): ";
    for (double wi : w) cout << wi << " ";
    cout << endl;

    return 0;
}

