// Here's how learning emerges from C, loops and floats

// MNIST
// MLP 784 (28*28 black/white pixels) -> 128 (hidden layer) -> 10 (digits)

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define LINE_BUF_SIZE 10000 // 1+784 * 4
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

void load_weights_and_biases(float *weights_1, float *weights_2,
                             float *biases_1, float *biases_2) {
  FILE *f = fopen("data/model_weights.csv", "r");
  if (!f) {
    printf("Error: Could not open model_weights.csv\n");
    return;
  }

  printf("Loading weights...\n");
  for (int i = 0; i < INPUT_DIM * HIDDEN_LAYER_DIM; i++) {
    if (fscanf(f, "%f", &weights_1[i]) != 1) {
      printf("Error reading weights at index %d\n", i);
      fclose(f);
      return;
    }
    if (i < 10) {
      printf("Loaded weights_1[%d] = %f\n", i, weights_1[i]);
    }
  }

  for (int i = 0; i < HIDDEN_LAYER_DIM; i++) {
    if (fscanf(f, "%f", &biases_1[i]) != 1) {
      printf("Error reading biases_1 at index %d\n", i);
      fclose(f);
      return;
    }

    if (i < 10) {
      printf("Loaded biases_1[%d] = %f\n", i, biases_1[i]);
    }
  }

  for (int i = 0; i < HIDDEN_LAYER_DIM * OUTPUT_DIM; i++) {
    if (fscanf(f, "%f", &weights_2[i]) != 1) {
      printf("Error reading weights_2 at index %d\n", i);
      fclose(f);
      return;
    }

    if (i < 10) {
      printf("Loaded weights_2[%d] = %f\n", i, weights_2[i]);
    }
  }

  for (int i = 0; i < OUTPUT_DIM; i++) {
    if (fscanf(f, "%f", &biases_2[i]) != 1) {
      printf("Error reading biases_2 at index %d\n", i);
      fclose(f);
      return;
    }

    if (i < 10) {
      printf("Loaded biases_2[%d] = %f\n", i, biases_2[i]);
    }
  }

  printf("Weights and biases loaded successfully \n");

  fclose(f);
}

void forward(float *inputs, float *weights_1, float *weights_2, float *biases_1,
             float *biases_2, float *activations_1, float *activations_2) {
  memset(activations_1, 0, HIDDEN_LAYER_DIM * sizeof(float));
  memset(activations_2, 0, OUTPUT_DIM * sizeof(float));

  // Debug print first few weights_1:
  // for (int i = 0; i < 10; i++) {
  //   printf("%f ", weights_1[i]);
  // }
  // printf("\n");

  printf("Inputs: \n ");
  for (int i = 0; i < 784; i++) {
    printf("%f ", inputs[i]);
  }

  for (int j = 0; j < HIDDEN_LAYER_DIM; j++) {
    for (int i = 0; i < INPUT_DIM; i++) {
      activations_1[j] += weights_1[j + i * HIDDEN_LAYER_DIM] * inputs[i];
    }
  }

  printf("\nAfter first linear: \n ");
  for (int i = 0; i < 5; i++) {
    printf("%f ", activations_1[i]);
  }
  for (int i = 0; i < HIDDEN_LAYER_DIM; i++) {
    activations_1[i] += biases_1[i];
  }

  printf("\nAfter first bias: \n ");
  for (int i = 0; i < 5; i++) {
    printf("%f ", activations_1[i]);
  }

  for (int i = 0; i < HIDDEN_LAYER_DIM; i++) {
    activations_1[i] =
        RELU(activations_1[i]); // should I store pre-activations and
                                // activations separately for backprop?
  }
  printf("\nAfter relu: \n ");
  for (int i = 0; i < 5; i++) {
    printf("%f ", activations_1[i]);
  }

  for (int j = 0; j < OUTPUT_DIM; j++) {
    for (int i = 0; i < HIDDEN_LAYER_DIM; i++) {
      activations_2[j] +=
          weights_2[j + i * OUTPUT_DIM] * activations_1[i];
    }
  }

  printf("\nAfter second linear: \n ");
  for (int i = 0; i < 5; i++) {
    printf("%f ", activations_2[i]);
  }

  for (int i = 0; i < OUTPUT_DIM; i++) {
    activations_2[i] += biases_2[i];
  }

  printf("\nAfter second bias: \n ");
  for (int i = 0; i < 10; i++) {
    printf("%f ", activations_2[i]);
  }

  // softmax
  float denominator = 0;

  float exponents[OUTPUT_DIM];
  float max_output = activations_2[0];
  int max_output_id = 0;

  for (int i = 0; i < OUTPUT_DIM; i++) {
    if (activations_2[i] > max_output) {
      max_output = activations_2[i];
      max_output_id = i;
    }

    exponents[i] = exp(activations_2[i]);
    denominator += exponents[i];
  }

  float outputs[OUTPUT_DIM];
  for (int i = 0; i < OUTPUT_DIM; i++) {
    outputs[i] = exponents[i] / denominator;
  }

  printf("\nPredicted: %i \n\n", max_output_id);
}

void train(float *weights_1, float *weights_2, float *biases_1, float *biases_2,
           float *activations_1, float *activations_2) {
  char line[LINE_BUF_SIZE];
  FILE *f = fopen("data/mnist_train.csv", "r");

  skip_csv_headers(f);

  initialize_weights(weights_1, INPUT_DIM * HIDDEN_LAYER_DIM, INPUT_DIM);
  initialize_weights(weights_2, HIDDEN_LAYER_DIM * OUTPUT_DIM,
                     HIDDEN_LAYER_DIM);

  int samples = MAX_TRAINING_EXAMPLES;
  while (samples > 0) {
    fgets(line, LINE_BUF_SIZE, f);
    char *token = strtok(line, ",\n"); // TODO: why ",\n" instead of "\n"?

    char label = token[0];
    float inputs[INPUT_DIM];

    int i = 1;
    while (i < INPUT_DIM) {
      token = strtok(NULL, ",\n");
      inputs[i] = atof(token) / 255.0f; // there might be a bug here, it's
                                        // possible I load label as first input
      i++;
    }

    forward(inputs, weights_1, weights_2, biases_1, biases_2, activations_1,
            activations_2); // TODO get outputs

    // TODO: backward pass

    // TODO: loss and gradient descent

    // printf("%s", line);
    samples--;
  }

  fclose(f);
}

int main(int argc, char *argv[]) {
  float *weights_1 = malloc(INPUT_DIM * HIDDEN_LAYER_DIM * sizeof(float));
  float *weights_2 = malloc(HIDDEN_LAYER_DIM * OUTPUT_DIM * sizeof(float));

  float *biases_1 = calloc(HIDDEN_LAYER_DIM, sizeof(float));
  float *biases_2 = calloc(OUTPUT_DIM, sizeof(float));

  float *activations_1 = calloc(HIDDEN_LAYER_DIM, sizeof(float));
  float *activations_2 = calloc(OUTPUT_DIM, sizeof(float));

  float *inputs = calloc(INPUT_DIM, sizeof(float));

  if (argc == 2) {
    if (strcmp(argv[1], "train") == 0) {
      train(weights_1, weights_2, biases_1, biases_2, activations_1,
            activations_2);
    }
    if (strcmp(argv[1], "run") == 0) {
      load_weights_and_biases(weights_1, weights_2, biases_1, biases_2);
      char line[LINE_BUF_SIZE];

      FILE *f = fopen("data/mnist_train.csv", "r");

      skip_csv_headers(f);

      int index = 0;
      int start_idx = 3000;
      int end_idx = 3010;

      while (index < end_idx) {
        fgets(line, LINE_BUF_SIZE, f);
        char *token = strtok(line, ",\n"); // TODO: why ",\n" instead of "\n"?

        if (index < start_idx) {
          index++;
          continue;
        }

        // fgets(line, LINE_BUF_SIZE, f);
        // char *token = strtok(line, ",\n"); // TODO: why ",\n" instead of
        // "\n"?

        char label = token[0];
        float inputs[INPUT_DIM];
        printf("label: %c, ", label);

        int i = 1;
        while (i < INPUT_DIM + 1) {
          token = strtok(NULL, ",\n");
          inputs[i - 1] =
              atof(token) / 255.0f; // there might be a bug here, it's
                                    // possible I load label as first input
          i++;
        }

        printf("First few input pixels: \n");
        for (int i = 0; i < 784; i += 28) {
          for (int j = 0; j < 28; j++) {
            if (inputs[i + j] == 0) {
              printf("|");
            } else {
              printf("-");
            }
          }
          printf("\n");
          // printf("%f ", inputs[i]);
        }
        printf("\n");

        forward(inputs, weights_1, weights_2, biases_1, biases_2, activations_1,
                activations_2); // TODO get outputs

        // TODO: backward pass

        // TODO: loss and gradient descent

        // printf("%s", line);
        index++;
      }

      fclose(f);
      // forward(inputs, weights_1, weights_2, biases_1, biases_2,
      // activations_1, activations_2);
    }
  }

  free(weights_1);
  free(weights_2);
  free(biases_1);
  free(biases_2);
  free(activations_1);
  free(activations_2);
  free(inputs);

  return 0;
}

// much later TODO:
// batching (SGD)
// optimize
// learning rate scheduler

// much much later TODO:
// hyper-optimize for a chosen chip architecture that I have access to