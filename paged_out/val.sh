#!/bin/bash
gcc -O3 -o mlp https://raw.githubusercontent.com/jmaczan/curiosity/refs/heads/main/mlp.c -lm && curl -s https://raw.githubusercontent.com/jmaczan/curiosity/refs/heads/main/paged_out/floats.txt | while IFS= read -r line; do
    label=${line%%,*}
    rest=${line#*,}
    echo -n "Label: $label, "
    ./mlp $(echo $rest | tr ',' ' ')
done
