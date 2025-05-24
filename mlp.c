// Here's how learning emerges from C, loops and floats

// MNIST
// MLP 784 (28*28 black/white pixels) -> 128 (hidden layer) -> 10 (digits)

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define LINE_BUF_SIZE 3137 // 1+784 * 4
#define INPUT_DIM 784      // 28 x 28
#define HIDDEN_LAYER_DIM 128
#define OUTPUT_DIM 10
#define MAX_TRAINING_EXAMPLES 3 // * 1000

#define RELU(i) (i > 0 ? i : 0)

// skipping the first line with headers, because I decide to hardcode the file
// structure
void skip_csv_headers(FILE *f) {
  int c;
  while ((c = fgetc(f)) != EOF && c != '\n')
    ;
}

// Random double number from range [0;1]
float get_random_number() { return (double)rand() / (double)RAND_MAX; }

// He initialization
void initialize_weights(float *weights, int weights_length, int layer_size) {
  float scale_factor = sqrtf(2.0 / layer_size);
  srand(time(NULL));
  for (int i = 0; i < weights_length; i++) {
    weights[i] = get_random_number() * scale_factor;
  }
}

int main() {
  char line[LINE_BUF_SIZE];
  FILE *f = fopen("data/mnist_train.csv", "r");

  skip_csv_headers(f);

  float *weights_1 = malloc(INPUT_DIM * HIDDEN_LAYER_DIM * sizeof(float));
  float *weights_2 = malloc(HIDDEN_LAYER_DIM * OUTPUT_DIM * sizeof(float));

  float *biases_1 = calloc(HIDDEN_LAYER_DIM, sizeof(float));
  float *biases_2 = calloc(OUTPUT_DIM, sizeof(float));

  initialize_weights(weights_1, INPUT_DIM * HIDDEN_LAYER_DIM, INPUT_DIM);
  initialize_weights(weights_2, HIDDEN_LAYER_DIM * OUTPUT_DIM,
                     HIDDEN_LAYER_DIM);

  float *activations_1 = calloc(HIDDEN_LAYER_DIM, sizeof(float));
  float *activations_2 = calloc(OUTPUT_DIM, sizeof(float));

  int samples = MAX_TRAINING_EXAMPLES;
  while (samples > 0) {
    fgets(line, LINE_BUF_SIZE, f);
    char *token = strtok(line, ",\n"); // TODO: why ",\n" instead of "\n"?

    char label = token[0];
    float inputs[INPUT_DIM];

    int i = 0;
    while (i < INPUT_DIM) {
      token = strtok(NULL, ",\n");
      inputs[i] = atoi(token) / 255.0f;
      i++;
    }

    for (int i = 0; i < INPUT_DIM; i++) {
      for (int j = 0; j < HIDDEN_LAYER_DIM; j++) {
        activations_1[j] += weights_1[i * HIDDEN_LAYER_DIM + j];
      }
    }

    for (int i = 0; i < HIDDEN_LAYER_DIM; i++) {
      activations_1[i] += biases_1[i];
      activations_1[i] =
          RELU(activations_1[i]); // should I store pre-activations and
                                  // activations separately for backprop?
    }

    for (int i = 0; i < HIDDEN_LAYER_DIM; i++) {
      for (int j = 0; j < OUTPUT_DIM; j++) {
        activations_2[j] += weights_2[i * HIDDEN_LAYER_DIM + j];
      }
    }

    for (int i = 0; i < OUTPUT_DIM; i++) {
      activations_2[i] += biases_2[i];
    }


    // softmax
    float denominator = 0;

    float exponents[OUTPUT_DIM];

    for (int i = 0; i < OUTPUT_DIM; i++) {
      exponents[i] = exp(activations_2[i]);
      denominator += exponents[i];
    }

    float outputs[OUTPUT_DIM];
    for (int i = 0; i < OUTPUT_DIM; i++) {
      outputs[i] = exponents[i] / denominator;
    }

    // TODO: backward pass

    // TODO: loss and gradient descent

    // printf("%s", line);
    samples--;
  }

  // TODO: store the weights and biases and make it loadable to the model
}

// much later TODO:
// batching (SGD)
// optimize
// learning rate scheduler

// much much later TODO:
// hyper-optimize for a chosen chip architecture that I have access to