#!/bin/bash

# Compile the C program
gcc -O3 -o mlp mlp.c -lm || { echo "Compilation failed"; exit 1; }

# Read each line from floats.txt
while IFS= read -r line; do
    # Convert commas to spaces (optional, safe to include)
    line=$(echo "$line" | tr ',' ' ')
    
    # Extract the first element and the rest
    label=${line%% *}
    rest=${line#* }

    # Print the label
    echo "label: $label"

    # Run the program with the rest of the line as arguments
    ./mlp $rest
done < floats.txt

# Clean up
rm ./mlp
