#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Here's how learning emerges from C, loops and floats

// MNIST
// MLP 784 (28*28 black/white pixels) -> 128 (hidden layer) -> 10 (digits)

#define LINE_BUF_SIZE 4096
#define INPUT_DIM 784
#define HIDDEN_LAYER_DIM 128
#define OUTPUT_DIM 10
#define MAX_TRAINING_EXAMPLES 3

#define RELU(i) (i > 0 ? i : 0)

// skipping the first line with headers, because I decide to hardcode the file
// structure
void skip_csv_headers(FILE *f) {
  int c;
  while ((c = fgetc(f)) != EOF && c != '\n')
    ;
}

int main() {
  char line[LINE_BUF_SIZE]; // on Stack
  FILE *f = fopen("data/mnist_train.csv", "r");

  skip_csv_headers(f);

  float *weights_1 = malloc(INPUT_DIM * HIDDEN_LAYER_DIM * sizeof(float)); // on Heap
  float *weights_2 = malloc(HIDDEN_LAYER_DIM * OUTPUT_DIM * sizeof(float));

  float *biases_1 = malloc(HIDDEN_LAYER_DIM * sizeof(float));
  float *biases_2 = malloc(OUTPUT_DIM * sizeof(float));

  int remaining_training_examples = MAX_TRAINING_EXAMPLES;
  while (remaining_training_examples > 0) {
    fgets(line, LINE_BUF_SIZE, f);
    char *token = strtok(line, ",\n");

    char label = token[0];
    float input[INPUT_DIM];

    // printf("%c \n", label);
    int i = 0;
    while (i < INPUT_DIM) {
      //   printf("%s \n", token);
      token = strtok(NULL, ",\n");
      input[i] = atoi(token) / 255.0f; // a room for performance improvements
                                       //   printf("%f \n", input[i]);
      i++;
    }

    // printf("%s", line);
    remaining_training_examples--;
  }
}