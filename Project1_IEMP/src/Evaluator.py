import time
import numpy as np
import networkx

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--network", "-n", type=str, required=True)
parser.add_argument("--initial-seed", "-i", type=str, required=True)
parser.add_argument("--balanced-seed", "-b", type=str, required=True)
parser.add_argument("--budget", "-k", type=int, required=True)
parser.add_argument("--output", "-o", type=str, required=True)
args = parser.parse_args()


def read_dataset(network_path, initial_seed_path, balanced_seed_path) -> (networkx.DiGraph, set, set):
    with open(network_path, "r") as f:
        n, m = map(int, f.readline().split())
        graph = networkx.DiGraph()
        for _ in range(m):
            u, v, p1, p2 = map(float, f.readline().split())
            graph.add_edge(int(u), int(v), p1=p1, p2=p2)

    with open(initial_seed_path, "r") as f:
        k1, k2 = map(int, f.readline().split())
        i1 = [int(f.readline().strip()) for _ in range(k1)]
        i2 = [int(f.readline().strip()) for _ in range(k2)]

    with open(balanced_seed_path, "r") as f:
        k1, k2 = map(int, f.readline().split())
        b1 = [int(f.readline().strip()) for _ in range(k1)]
        b2 = [int(f.readline().strip()) for _ in range(k2)]

    return graph, set(i1 + b1), set(i2 + b2)


def evaluate(graph, set1, set2, simtimes) -> float:
    expected = 0
    node_num = graph.number_of_nodes()
    rand_list1 = np.random.rand(simtimes, graph.number_of_edges())
    rand_list2 = np.random.rand(simtimes, graph.number_of_edges())
    edges = {node: graph.edges(node, data=True) for node in graph.nodes}

    for i in range(simtimes):
        # find compaign 1's infected
        j = 0
        infected1 = set1.copy()
        activated = set1.copy()
        active = set1.copy()
        while active:
            new = set()
            for node in active:
                for v, n, p in edges[node]:
                    if n not in activated:
                        infected1.add(n)
                        if rand_list1[i][j] < p['p1']:
                            new.add(n)
                        j += 1
            activated.update(new)
            active = new

        # find compaign 2's infected
        j = 0
        infected2 = set2.copy()
        activated = set2.copy()
        active = set2.copy()
        while active:
            new = set()
            for node in active:
                for v, n, p in edges[node]:
                    if n not in activated:
                        infected2.add(n)
                        if rand_list2[i][j] < p['p2']:
                            new.add(n)
                        j += 1
            activated.update(new)
            active = new

        expected += node_num - len(set.symmetric_difference(infected1, infected2))

    return expected / simtimes


if __name__ == "__main__":
    start = time.time()
    graph, set1, set2 = read_dataset(args.network, args.initial_seed, args.balanced_seed)
    ans = evaluate(graph, set1, set2, 500)
    # print(f"Execution time: {time.time() - start:.2f}s")
    # print(ans)

    with open(args.output, "w") as f:
        f.write(str(ans))
