#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

class NeuralNetwork {
public:
    NeuralNetwork(int numInputs, int numOutputs, int numHiddenLayers, int numNeuronsPerHiddenLayer) {
        // Initialize the network architecture
        this->numInputs = numInputs;
        this->numOutputs = numOutputs;
        this->numHiddenLayers = numHiddenLayers;
        this->numNeuronsPerHiddenLayer = numNeuronsPerHiddenLayer;

        // Initialize the weights and biases
        weights.resize(numHiddenLayers + 1);
        biases.resize(numHiddenLayers + 1);
        for (int i = 0; i < numHiddenLayers + 1; i++) {
            weights[i].resize(numNeuronsPerHiddenLayer);
            biases[i].resize(numNeuronsPerHiddenLayer);
            for (int j = 0; j < numNeuronsPerHiddenLayer; j++) {
                weights[i][j].resize(i == 0 ? numInputs : numNeuronsPerHiddenLayer);
                biases[i][j].resize(i == 0 ? numInputs : numNeuronsPerHiddenLayer);
                for (int k = 0; k < (i == 0 ? numInputs : numNeuronsPerHiddenLayer); k++) {
                    weights[i][j][k] = 0.0;
                    biases[i][j][k] = 0.0;
                }
            }
        }
    }

    void train(vector<vector<double>> inputs, vector<vector<double>> outputs, int numEpochs, double learningRate) {
        // Train the network using backpropagation
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            for (int i = 0; i < inputs.size(); i++) {
                // Forward pass
                vector<vector<double>> activations = forwardPass(inputs[i]);

                // Compute the error
                vector<double> error = computeError(activations, outputs[i]);

                // Backpropagate the error
                backpropagateError(activations, error, learningRate);
            }
        }
    }

    vector<double> predict(vector<double> input) {
        // Perform a forward pass to make a prediction
        vector<vector<double>> activations = forwardPass(input);

        // Return the output layer activations
        return activations[activations.size() - 1];
    }

private:
    int numInputs;
    int numOutputs;
    int numHiddenLayers;
    int numNeuronsPerHiddenLayer;

    vector<vector<vector<double>>> weights;
    vector<vector<vector<double>>> biases;

    vector<vector<double>> forwardPass(vector<double> input) {
        // Initialize the activations
        vector<vector<double>> activations;
        activations.push_back(input);

        // Perform forward pass through the hidden layers
        for (int i = 0; i < numHiddenLayers; i++) {
            vector<double> layerActivations;
            for (int j = 0; j < numNeuronsPerHiddenLayer; j++) {
                double weightedSum = 0.0;
                for (int k = 0; k < (i == 0 ? numInputs : numNeuronsPerHiddenLayer); k++) {
                    weightedSum += weights[i][j][k] * activations[i][k];
                }
                weightedSum += biases[i][j][0];
                layerActivations.push_back(sigmoid(weightedSum));
            }
            activations.push_back(layerActivations);
        }

        // Perform forward pass through the output layer
        vector<double> outputActivations;
        for (int j = 0; j < numOutputs; j++) {
            double weightedSum = 0.0;
            for (int k = 0; k < numNeuronsPerHiddenLayer; k++) {
                weightedSum += weights[numHiddenLayers][j][k] * activations[numHiddenLayers][k];
            }
            weightedSum += biases[numHiddenLayers][j][0];
            outputActivations.push_back(sigmoid(weightedSum));
        }
        activations.push_back(outputActivations);

        return activations;
    }

    vector<double> computeError(vector<vector<double>> activations, vector<double> expectedOutput) {
        // Compute the error for the output layer
        vector<double> error;
        for (int i = 0; i < numOutputs; i++) {
            error.push_back(expectedOutput[i] - activations[activations.size() - 1][i]);
        }

        // Propagate the error back through the hidden layers
        for (int i = numHiddenLayers - 1; i >= 0; i--) {
            vector<double> layerError;
            for (int j = 0; j < numNeuronsPerHiddenLayer; j++) {
                double weightedError = 0.0;
                for (int k = 0; k < numOutputs; k++) {
                    weightedError += weights[i + 1][k][j] * error[k];
                }
                layerError.push_back(weightedError * sigmoidPrime(activations[i][j]));
            }
            error = layerError;
        }

        return error;
    }

    void backpropagateError(vector<vector<double>> activations, vector<double> error, double learningRate) {
        // Update the weights and biases of the output layer
        for (int j = 0; j < numOutputs; j++) {
            for (int k = 0; k < numNeuronsPerHiddenLayer; k++) {
                weights[numHiddenLayers][j][k] += learningRate * error[j] * activations[numHiddenLayers][k];
                biases[numHiddenLayers][j][0] += learningRate * error[j];
            }
        }

        // Update the weights and biases of the hidden layers
        for (int i = numHiddenLayers - 1; i >= 0; i--) {
            for (int j = 0; j < numNeuronsPerHiddenLayer; j++) {
                for (int k = 0; k < (i == 0 ? numInputs : numNeuronsPerHiddenLayer); k++) {
                    weights[i][j][k] += learningRate * error[j] * activations[i][k];
                    biases[i][j][0] += learningRate * error[j];
                }
            }
        }
    }


    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    double sigmoidPrime(double x) {
        return x * (1.0 - x);
    }
};

int main() {
    // Create the training data
    vector<vector<double>> inputs = { {1.0}, {2.0}, {3.0}, {4.0}, {5.0} };
    vector<vector<double>> outputs = { {0.8}, {0.9}, {0.98}, {0.99}, {1.0} };

    // Create the neural network
    NeuralNetwork network(1, 1, 2, 10);

    // Train the neural network
    network.train(inputs, outputs, 1000, 0.1);

    // Make a prediction
    double input = 6.0;
    vector<double> prediction = network.predict({ input });

    // Print the prediction
    cout << "Predicted output: " << prediction[0] << endl;

    return 0;
}