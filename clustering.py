import networkx

class Clustering:

    EPSILON = 0.001  # Weight to use on matching graph instead of 0.

    def __init__(self, alpha, clustering_lol):
        self.alpha = alpha
        self.clustering = clustering_lol
        self.k = len(self.clustering)
        self.labeling = [ i for i in range(0, self.k) ]

    def set_labeling(self, label_list):
        self.labeling = label_list

    def get_labeling(self):
        return self.labeling

    def get_dict_id_labels(self):
        id_labels = {}
        for cluster, label in zip(self.clustering, self.labeling):
            for point in cluster:
                id_labels[point] = label
        return id_labels

    def node_label(cat, num):
        return str(cat) + "." + str(num)

    def get_index(node_label):
        return int(node_label.split(".")[-1])

    def get_intersection_weight(clustlist1, clustlist2):
        intersection = set(clustlist1).intersection(clustlist2)
        weight = len(intersection)
        return weight

    def get_percent_weight(clustlist1, clustlist2):
        print("    clust1 size:" + str(len(clustlist1)) + "  clust2 size:" + str(len(clustlist2)))
        intersection = set(clustlist1).intersection(clustlist2)
        weight = len(intersection)
        print("    intersection size:" + str(weight))
        if weight == 0:
            return 0
        percent = weight / len(clustlist1)
        print("    percent:" + str(percent))
        return percent

    def make_bipartite_graph(clustering1, clustering2):
        k = clustering1.k
        assert(k == clustering2.k)

        graph = networkx.Graph()
        for i in range(0, k):
            n1 = Clustering.node_label(1, i)
            graph.add_node(n1, bipartite=0)
        for i in range(0, k):
            n2 = Clustering.node_label(2, i)
            graph.add_node(n2, bipartite=1)

        for i in range(0, k):
            for j in range(0, k):
                clust1 = clustering1.clustering[i]
                clust2 = clustering2.clustering[j]
                n1 = Clustering.node_label(1, i)
                n2 = Clustering.node_label(2, j)
                weight = Clustering.get_intersection_weight(clust1, clust2)
                print("n1:" + n1 + " n2:" + n2 + "  weight:" + str(weight))
                if weight == 0:
                    weight = Clustering.EPSILON
                graph.add_edge(n1, n2, weight = weight)

        return graph

    def matching_to_dict(matching_dict_tuples):
        matching = {}
        for start, end in matching_dict_tuples:
            matching[start] = end
            matching[end] = start
        return matching

    def set_labeling_maxmatching(clustering_start, clustering_end):
        k = clustering_start.k
        graph = Clustering.make_bipartite_graph(clustering_start, clustering_end)
        # matching = networkx.bipartite.maximum_matching(graph)
        matching_tuples = networkx.algorithms.max_weight_matching(
                              graph, maxcardinality=False, weight='weight')
        matching = Clustering.matching_to_dict(matching_tuples)
        print(matching)
        startlabeling = clustering_start.get_labeling()
        print("start labels:" + str(startlabeling))
        endlabeling = []
        for i in range(0, k):
            startnode = Clustering.node_label(1, i)
            nodematch = matching[startnode]
            j = Clustering.get_index(nodematch)
            startlabel = startlabeling[j]
            endlabeling.append(startlabel)
        clustering_end.set_labeling(endlabeling)

# c1 = Clustering(0.5, [[1,2],[2,3],[1,3]])
# c2 = Clustering(0.5, [[1,2],[1,3],[2,3]])
# Clustering.set_labeling_maxmatching(c1, c2)

