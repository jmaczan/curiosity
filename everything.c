#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct Tensor
{
    int *data;
    int size;
};

struct Tensor tensor(int size)
{
    struct Tensor t;
    t.size = size;
    t.data = (int *)calloc(size, sizeof(int));
    if (!t.data)
    {
        fprintf(stderr, "Failed to allocate memory for a tensor\n");
        exit(1);
    }
    return t;
}

void tensor_free(struct Tensor *t)
{
    free(t->data);
    t->data = NULL;
    t->size = 0;
}

struct Tensor add(struct Tensor a, struct Tensor b)
{
    if (a.size != b.size)
    {
        printf("Tensors should have the same size. a: %d, b: %d", a.size, b.size);
        exit(1);
    }

    struct Tensor output;
    output.data = (int *)malloc(sizeof(int) * a.size);

    for (int i = 0; i < a.size; i++)
    {
        output.data[i] = a.data[i] + b.data[i];
    }

    return output;
}

int main(int argc, char *argv[])
{
    int t1_size = 10;
    struct Tensor t1 = tensor(t1_size);
    t1.data[0] = 1;
    t1.data[1] = 2;
    t1.data[2] = 3;
    t1.data[3] = 4;
    t1.data[4] = 5;
    t1.data[5] = 6;
    t1.data[6] = 7;
    t1.data[7] = 8;
    t1.data[8] = 9;
    t1.data[9] = 10;

    printf("t1: %d\n", t1.data[0]);
    printf("t1: %d\n", t1.data[1]);
    printf("t1: %d\n", t1.data[2]);
    printf("t1: %d\n", t1.data[3]);
    printf("t1: %d\n", t1.data[4]);
    printf("t1: %d\n", t1.data[5]);
    printf("t1: %d\n", t1.data[6]);
    printf("t1: %d\n", t1.data[7]);
    printf("t1: %d\n", t1.data[8]);
    printf("t1: %d\n", t1.data[9]);

    t1 = add(t1, t1);

    printf("t1: %d\n", t1.data[0]);
    printf("t1: %d\n", t1.data[1]);
    printf("t1: %d\n", t1.data[2]);
    printf("t1: %d\n", t1.data[3]);
    printf("t1: %d\n", t1.data[4]);
    printf("t1: %d\n", t1.data[5]);
    printf("t1: %d\n", t1.data[6]);
    printf("t1: %d\n", t1.data[7]);
    printf("t1: %d\n", t1.data[8]);
    printf("t1: %d\n", t1.data[9]);

    tensor_free(&t1);
    return 0;
}