#############################################################################################
# This pipeline is the same as run.sh except that it clusters and runs analysis based only
# on the largest connected component of the dblp network (which contains 2190 nodes).
#############################################################################################
# for i in 1 2 3
# do
#   echo "hi $i times"
# done
#


for i in 5 10 15 20 25 30 35 40 45 50
do
  python3 clustering_pipeline.py before_vectors_seed_cc $i
  cd C++\ code/
  g++ main.cpp -o main -std=c++11
  echo "../data/dblp/cc_pagerank_seed_edgelist.txt" "../data/dblp/pagerank_vectors_cc_$i.txt" $1 $2 "n"| ./main
  cd ..
  python3 clustering_pipeline.py find_p_after_vectors $i >> data/dblp/pagerank_p_ari.txt
done
