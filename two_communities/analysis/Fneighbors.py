'''p_i = function of p_i values of all neighbors'''
import networkx as nx
import csv 
import statistics
import ast

def main():
    Pfunc_neighbors("statistics.mean(values_neighbors)","twocommunities_vectors.txt","twocommunities_neighborlist.txt",1.0e-6,2000)

def Pfunc_neighbors(function,vector_file,neighbor_file,error_tolerance,N):
    vectors = {}
    neighbors = {}
    with open("../" + vector_file) as vectorFile:
        vectorData = csv.reader(vectorFile, delimiter='\t')
        #dict with node as key and prob of receiving info from C++ Code Orig as value 
        for line in vectorData:
            vectors[int(line[0])] = float(line[1]) 
    with open("../" + neighbor_file) as neighborFile:
        neighborList = csv.reader(neighborFile, delimiter='\t')
        #dict with node as key and list of its neighbors as value
        for line in neighborList:
            neighbornodes = ast.literal_eval(line[1])
            neighbors[int(line[0])] = list(neighbornodes)
    print("Trying to converge...")
    recursion(function,vectors,neighbors,error_tolerance,N)
    

def recursion(function,vectors,neighbors,error_tolerance,N):
    converge = 0    
    for key in neighbors:
        values_neighbors = []
        for node in list(neighbors.get(key)):
            values_neighbors.append(vectors.get(node))
            #print("node",key,"has neighbors:",values_neighbors)
        newvalue = eval(function)
        if abs(newvalue - vectors.get(key)) <= error_tolerance:
            converge += 1
        vectors[key] = newvalue
    if converge != N:
        #print("Has not converged. Trying again.")
        recursion(function,vectors,neighbors,error_tolerance,N)
    else:
        print("Converged! New values in 'func_neighbors.txt' file.")
        #create file with list of nodes and their new value
        with open("func_neighbors.txt", 'w') as f:
            for key,value in vectors.items():
                f.write('%s\t%s\n' %(key, value))       


if __name__ == "__main__":
    main()