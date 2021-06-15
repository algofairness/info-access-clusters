g++ ./main.cpp -o main -std=c++11
#./main

echo 'Please enter the path to input file'
read input

echo 'Please enter the path to output file'
read output

echo 'Please enter the alpha value?'
read alpha
value=${alpha#*.}

echo 'Please enter the number of repetitions?'
read nr

echo "Run multi source simulation (MIT): y or n?"
read multi

echo "Run simulations for all seeds: y or n?"
read response

./main $input $output $alpha $nr $response $multi

#./main "../output_files/dblp_edgelist_MIT.txt" "../output_files/dblp_probs_MIT/dblp_vectors_i${value}_${nr}.txt" $alpha $nr $response $multi

#./main "../output_files/dblp_edgelist.txt" "../output_files/dblp_vectors/dblp_vectors_i${value}_${nr}.txt" $alpha $nr $response
