# gcc -O3 -o test test.c -lm && ./test $(curl -s http://example.com/floats.txt | tr ',' ' ') && rm ./test
gcc -O3 -o mlp mlp.c -lm && ./mlp $(tr ',' ' ' < floats.txt) && rm ./mlp