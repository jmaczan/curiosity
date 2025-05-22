#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct Tensor
{
    int *data;
    int size;
    int *shape;
    int ndim; // number of dimensions aka size of shape
};

struct Tensor tensor(int *shape, int ndim)
{
    struct Tensor t;
    int size = 1;
    for (int i = 0; i < ndim; i++)
    {
        size *= shape[i];
    }
    t.size = size;
    t.shape = shape;
    t.ndim = ndim;
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

struct Tensor view(struct Tensor t, int *view_shape, int view_shape_dim)
{
    if (t.ndim != view_shape_dim)
    {
        printf("Shape of view should have the same number of dimensions as tensor");
        exit(1);
    }

    // example of tensor with shape [2, 2, 3] - so a 3 dimensional tensor: B, T, C
    // [query]: [output] -- {index} 
    // assuming 0-indexed queries and indices
    // [a, b, c]
    // [0, 0, 1]: 6 -- {1}: c
    // [0, 1, 2]: 4 -- {5}: a*B + b*T + C

    // [0, 0]: [3, 6, 2] -- {0...2}: a*B + b*T ... a*B+b*T+C (but C is 1-indexed, so perhaps C-1 here or decrease be 1 all items in shape?)
    // [0, 1]: [6, 8, 4] -- {4...6}: b*C

    // [1, 0]: [1, 2, 3] -- {7..9}
    // [1, 1]: [2, 4, 5] -- {10..12} b*C*T


    // [1]: [3, 6, 2, 6, 8, 4] or [[3, 6, 2], [6, 8, 4]] -- {1...6}

    // now, I want to be able to retrieve nth element or row or column

    // data is stored contiguously as an array, so I need a way to compute indices of items from element/row/column I want to retrieve
    // [3, 6, 2, 6, 8, 4, 1, 2, 3, 2, 4, 5]

    

    // so these ^ are rather straighforward. how about more difficult scenarios, like getting for all elements first row and second column? so something like [all, 1, 2]?

}

int main(int argc, char *argv[])
{
    int shape[] = {10};
    struct Tensor t1 = tensor(shape, 1);
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