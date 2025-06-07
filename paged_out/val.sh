#!/bin/bash
curl -s https://raw.githubusercontent.com/jmaczan/curiosity/refs/heads/main/paged_out/pagedout_mlp.c -o mlp.c
gcc -O3 -o mlp mlp.c -lm
curl -s https://raw.githubusercontent.com/jmaczan/curiosity/refs/heads/main/paged_out/floats.txt | while IFS= read -r line; do
    label=${line%%,*}
    rest=${line#*,}
    echo -n "Label: $label, "
    ./mlp $(echo $rest | tr ',' ' ')
done
