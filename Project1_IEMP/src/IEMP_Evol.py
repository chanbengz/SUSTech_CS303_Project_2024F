import time
import numpy as np
import networkx

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--network", "-n", type=str, required=True)
parser.add_argument("--initial-seed", "-i", type=str, required=True)
parser.add_argument("--balanced-seed", "-b", type=str, required=True)
parser.add_argument("--budget", "-k", type=int, required=True)
args = parser.parse_args()


def read_dataset(network_path, initial_seed_path) -> (networkx.DiGraph, set, set):
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

    return graph, set(i1), set(i2)


if __name__ == "__main__":
    start = time.time()
    graph, set1, set2 = read_dataset(args.network, args.initial_seed, args.balanced_seed)
    print(f"Execution time: {time.time() - start:.2f}s")

    with open(args.balanced_seed, "w") as f:
        f.write()
