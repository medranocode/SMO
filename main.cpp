//#include <core/lattice/lat-hal.h>
//#include <pke/openfhe.h>
#include <iostream>
#include <getopt.h>
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>
#include <math/dftransform.h>
#include "PSA-cryptocontext.h"
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

// Linear kernel
double kernel(const vector<double>& x1, const vector<double>& x2) {
    double result = 0.0;
    for (size_t i = 0; i < x1.size(); i++)
        result += x1[i] * x2[i];
    return result;
}

void handler(int sig) {
    void *array[10];
    size_t size;

    // get void*'s for all entries on the stack
    size = backtrace(array, 10);

    // print out all the frames to stderr
    fprintf(stderr, "Error: signal %d:\n", sig);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    exit(1);
}

double slap(int argc, char **argv, std::vector<double>& u1, std::vector<int>& u2, std::vector<double>& u3) {
        signal(SIGSEGV, handler);
        std::cout << "Hello, World! " << std::endl;
        //DCRTPoly a = DCRTPoly();
        unsigned int plain_bits = 15; //log t
        unsigned int num_users = 3; //n
        unsigned int iters = 1; //i
        unsigned int k_prime = 1; //k
        Scheme scheme1 = NS;

        unsigned int N = 1; //N

        int c;
          while((c = getopt(argc, argv, "t:n:i:k:N:")) != -1){
            switch(c){
            case 't':{
                plain_bits = atoi(optarg);
                break;
            }
        case 'n':{
                num_users = atoi(optarg);
                break;
            }
        case 'i':{
                iters = atoi(optarg);
                break;
            }
        case 'k':{
                k_prime = atoi(optarg);
                break;
            }
        case 'N':{
                N = atoi(optarg);
                break;
            }
        default:{
            std::cout << "Invalid argument: " << c;
            if(optarg != nullptr){
                std::cout << ' ' << optarg;
            }
            std::cout << std::endl;
            return 1;
        }
            }
          }

        if(!plain_bits){
            throw std::runtime_error("Must have nonempty plaintext space");
        }  
        if(!num_users){
            throw std::runtime_error("Must have at least some users");
        }
        if(!iters){
            throw std::runtime_error("Must have at least some iterations");
        }

        unsigned int MAX_CTEXTS_DEFAULT = 20;

        //temp();

        //Code for testing SLAP, which isn't what this paper is about

        /**
        PSACryptocontext p = PSACryptocontext(plain_bits, num_users, iters, scheme1);
        std::vector<double> noise_times;
        std::vector<double> enc_times;
        std::vector<double> dec_times;
        p.TestEncryption(iters, false, noise_times, enc_times);

        p.TestDecryption(iters,dec_times);

        for(const double d : noise_times){
            std::cout << "noise_times " << d << '\n';
        }
        for(const double d : enc_times){
            std::cout << "enc_times " << d << '\n';
        }
        for(const double d : dec_times){
            std::cout << "dec_times " << d << '\n';
        }
         **/


        PSACryptocontext pp = PSACryptocontext(plain_bits, num_users, iters, scheme1);

        std::vector<double> poly_noise_times;
        std::vector<double> poly_enc_times;

        //pp.TestPolynomialEncryption(true, iters, poly_noise_times, poly_enc_times);
        // pp.TestPolynomialEncryption(1, MAX_CTEXTS_DEFAULT, poly_noise_times, poly_enc_times);

        pp.PolynomialEnvSetup(poly_noise_times, poly_enc_times);

        for (int i = 0; i < num_users; i++) {
            std::vector<double> inputvec = u1;
            std::vector<double> expvec(u1.size(), 1);

            //PUT ANOTHER TIME std::cout << i << " input: " << inputvec << std::endl;

            pp.PolynomialEncryption(inputvec, expvec, i, poly_noise_times, poly_enc_times);
        }


        std::vector<double> decrypt_times;

        std::vector<double> constants(pp.aggregator.plaintextParams.GetRingDimension()/2,2);
        std::vector<double> outputvec = pp.PolynomialDecryption(constants, 1, decrypt_times);

        std::cout << "Final output: " << outputvec << std::endl;


        for(const double d : poly_noise_times){
            //std::cout << "poly_noise_times " << d << '\n';
        }
        for(const double d : poly_enc_times){
            //std::cout << "poly_enc_times " << d << '\n';
        }
        for(const double d : decrypt_times){
            //std::cout << "decrypt_times " << d << '\n';
        }

	double sum = std::accumulate(outputvec.begin(), outputvec.end(), 0);

        return sum;
    }

// SVM output for a given point
double SVMOutput(int argc, char **argv, int i) {
    //double result = -b;
    //for (size_t j = 0; j < points.size(); j++)
        //result += alpha[j] * target[j] * kernel(points[i], points[j]);
    //return result;
    std::vector<double> kernel_points;
    for (size_t j = 0; j < points.size(); j++){
       kernel_points.push_back(kernel(points[i], points[j]));
    }
    double result = slap(argc, argv, alpha, target, kernel_points);
    result -= b;

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

// TakeStep method
bool takeStep(int argc, char **argv, int i1, int i2) {
    if (i1 == i2) return false;

    double alpha1 = alpha[i1], alpha2 = alpha[i2];
    int y1 = target[i1], y2 = target[i2];
    double E1 = SVMOutput(argc, argv, i1) - y1;
    double E2 = SVMOutput(argc, argv, i2) - y2;
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

// SMO method
void SMO(int argc, char **argv) {
    numChanged = 0;
    bool examineAll = true;

    while (numChanged > 0 || examineAll) {
        numChanged = 0;
        if (examineAll) {
            for (size_t i = 0; i < points.size(); i++)
                numChanged += takeStep(argc, argv, rand() % points.size(), i);
        } else {
            for (size_t i = 0; i < points.size(); i++)
                if (alpha[i] > 0 && alpha[i] < C)
                    numChanged += takeStep(argc, argv, rand() % points.size(), i);
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

    // Skip the header row
    getline(file, line);

    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        vector<double> point;
        int label;

        // Read features (semicolon-delimited)
        for (int i = 0; i < 11; i++) { // First 11 columns are features
            getline(ss, value, ';'); // Use semicolon as the delimiter
            point.push_back(stod(value)); // Convert to double
        }

        // Read label (quality)
        getline(ss, value, ';'); // Last column
        int quality = stoi(value); // Convert to integer

        // Transform quality into binary labels (+1 for good, -1 for not good)
        label = (quality >= 6) ? 1 : -1;

        points.push_back(point);
        target.push_back(label);
    }

    file.close();
}

int main(int argc, char **argv){
    // Load the dataset
    loadCSV("winequality-red.csv", points, target);

    // Initialize alpha and weight vectors
    alpha = vector<double>(points.size(), 0.0);
    w = vector<double>(points[0].size(), 0.0);

    // Run the SMO algorithm
    SMO(argc, argv);

    // Display results
    cout << "Final alpha values:" << endl;
    for (double a : alpha) cout << a << " ";
    cout << endl;

    cout << "Bias (b): " << b << endl;

    cout << "Weight vector (w): ";
    for (double wi : w) cout << wi << " ";
    cout << endl;

    return 0;
}
