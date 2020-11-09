#############################################################################################
# This pipeline is the same as run.sh except that it clusters and runs analysis based only
# on the largest connected component of the dblp network (which contains 2190 nodes) and
# only based on a select set of seed nodes.
#############################################################################################
python3 clustering_pipeline.py before_vectors_seed_cc 35
cd C++\ code/
g++ main.cpp -o main -std=c++11
echo "../data/dblp/cc_degree_seed_edgelist.txt" "../data/dblp/cc_degree_seed_vectors.txt" $1 $2 "n"| ./main
cd ..
python3 clustering_pipeline.py after_vectors_seed_cc
python3 clustering_pipeline.py find_p_after_vectors 35
