import random
import csv

def main():
    edges = dictEdges("network.dat")
    node_removal_number(edges,0,100)
    produce_seed_edgelist(edges,"twocommunities_seed_edgelist.txt",2000,0.3,0.3)

def dictEdges(edgeList):
    with open(edgeList) as file:
        edgeFile = csv.reader(file, delimiter='\t')
        edges = {}
        for line in edgeFile:
            edges[int(line[0])] = int(line[1])
    return edges


def LFRnode_removal_number(edges,numberA, numberB):
    global Aremoved
    Aremoved = 0
    global Bremoved
    Bremoved = 0
    indexA = random.sample(range(0, 1000), numberA)
    indexB = random.sample(range(1000,2000), numberB)
    copyedges = edges.copy()
    for index in indexA:
        for key,value in copyedges.items():
            if key == index or value==index:
                #print("key:",key,"value:",value)
                edges.pop(key,None)
                Aremoved += 1
    for index in indexB:
        for key,value in copyedges.items():
            if key == index or value==index:
                #print("key:",key,"value:",value)
                edges.pop(key,None)
                Bremoved += 1
    return edges

def LFRproduce_nodelist(edges,inNodesFile):
    with open(inNodesFile, 'w') as txt_file:
        for key in edges.items():
            if key not in edges:
                txt_file.write("{}\n".format(key))
    return 

def LFRproduce_edgelist(edges,inEdgesFile):
    with open(inEdgesFile,'w') as txt_file:
        for key,value in edges.items():
            txt_file.write("{}\t{}\n".format(key,value))
    return 

def LFRproduce_neighborlist(edges,neighborsFile):
    with open(neighborsFile,'w') as txt_file:
        neighbors = []
        node = 0
        for key,value in edges.items():
            if key == node:
                neighbors.append(value) 
            else:
                txt_file.write("{}\t{}\n".format(node,neighbors))
                node = key
                neighbors = value
    return 

def LFRproduce_seed_edgelist(edges,inSeedEdgesFile,number_nodes,PsA,PsB):
    with open(inSeedEdgesFile, 'w') as txt_file:
        global seeds 
        seeds = []
        directed = 0
        txt_file.write("{}\t{}\n".format(len(edges), directed))
        for key,value in edges.items():
            txt_file.write("{}\t{}\n".format(key, value))
        txt_file.write ("s\t")
        for key in range(1000):
            dice = random.random()
            if dice <= PsA:
                txt_file.write("{}\t".format(key))
                seeds.append(key)
        for key in range(1000,2000):
            dice = random.random() 
            if dice <= PsB:
                txt_file.write("{}\t".format(key))
                seeds.append(key)
        txt_file.write("\n")
    return 

def LFRproduce_seed_edgelist_number(edges,inSeedEdgesFile,number_nodes,numSeedA, numSeedB):
    numSeedA = int(numSeedA)
    numSeedB = int(numSeedB) 
    with open(inSeedEdgesFile,'w') as txt_file:
        global seeds
        directed = 0
        txt_file.write("{}\t{}\n".format(len(edges), directed))
        for key,value in edges.items():
            txt_file.write("{}\t{}\n".format(key, value))
        txt_file.write ("s\t")
        indexA = random.sample(range(0, 1000 - Aremoved), numSeedA)
        indexB = random.sample(range(1000 - Aremoved,2000 - Aremoved - Bremoved), numSeedB)                    
        for index in indexA:
            txt_file.write("{}\t".format(index))
        for index in indexB:
            txt_file.write("{}\t".format(index))
        seeds = indexA + indexB
    return 

if __name__ == "__main__":
    main()