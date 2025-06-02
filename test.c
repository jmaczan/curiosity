#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    printf("argc: %i\n", argc);
    for (int i = 1; i < argc; i++) {
        float f = strtof(argv[i], NULL) / 255.0f;
        printf("arg[%d] as float: %f\n", i, f);
    }
}
