import sys

graph = {}
in_comp = {}

# Format: the file should contain list of edges:
# each edge in one line - node ids separated by space
def main():
    filename = sys.argv[1]
    load_graph(filename)
    comp = lscc()
    out_put(filename, comp)


# Load the "Directed" graph
def load_graph(file):

    nodes = 0
    edges = 0
    f = open(file, "r")
    for line in f:
        edges += 1
        s = int(line.split(None, 2)[0])
        t = int(line.split(None, 2)[1])
        if s not in graph:
            graph[s] = [t]
            nodes += 1
        else:
            if t not in graph[s]:
                graph[s].append(t)
        if t not in graph:
            graph[t] = []
            nodes += 1

    print("number of nodes: " + str(nodes) + " number of edges: " + str(edges))

    f.close()


# Outputs the largest strongly connected component
def out_put(filename, largest):
    out = open(filename[:-4] + "_lscc.txt", "w")
    
    index = {}
    nodes = 0
    for node in graph:
        if in_comp[node] == largest:
            index[node] = nodes
            nodes += 1
    
    edges = 0
    for node in graph:
        if in_comp[node] == largest:
            for neighbor in graph[node]:
                if in_comp[neighbor] == largest:
                    edges += 1
                    out.write(str(index[node]) + "\t" + str(index[neighbor]) + "\n")
    
    print("number of nodes: " + str(nodes) + " number of edges: " + str(edges))
    out.close()

# Labels the nodes by their strongly connected components' id
# Returns the id of the largest component
def lscc():
    maxim = 0
    largest = 0

    visit = {}
    root = {}
    index = {}
    node_id = 0
    comp = 0
    path = []

    for key in graph:
        in_comp[key] = 0
        visit[key] = False

    #For every Spanning Tree:
    for key in graph:
        if not visit[key]:
            stack = [key]

            #Start
            while stack:
                #print(node_id+1)
                #print(stack)
                top_s = stack.pop()
                if in_comp[top_s] != 0:
                    continue
                stack.append(top_s)

                if not visit[top_s]:
                    node_id += 1
                    index[top_s] = node_id
                    visit[top_s] = True
                    root[top_s] = node_id
                    path.append(top_s)

                if len(path) == 0:
                    print(top_s)
                count = 0
                for neighbor in graph[top_s]:
                    if visit[neighbor]:
                        continue
                    stack.append(neighbor)
                    count += 1

                #Next Step
                if count != 0:
                    continue
                #print("now: ", top_s)
                stack.pop()
                for neighbor in graph[top_s]:
                    if in_comp[neighbor] != 0:
                        continue
                    root[top_s] = min(root[top_s], root[neighbor])
                #print(root)
                #print("check", top_s, index[top_s])

                if root[top_s] != index[top_s]:
                    continue
                #print("pass")
                count = 0
                comp += 1
                while path:
                    vertex = path.pop()
                    count += 1
                    in_comp[vertex] = comp
                    if vertex == top_s:
                        break
                if maxim < count:
                    maxim = count
                    largest = comp
                if maxim > len(graph) / 2:
                    return largest
                    
    #print("root", root)
    #print("index", index)
    #print("incomp", in_comp)
    print("Largest component id: ", largest)
    return largest

if __name__ == "__main__":
    main()
