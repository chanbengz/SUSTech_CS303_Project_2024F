import time
import numpy as np
from functools import reduce
from collections import defaultdict

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--network", "-n", type=str, required=True)
parser.add_argument("--initial-seed", "-i", type=str, required=True)
parser.add_argument("--balanced-seed", "-b", type=str, required=True)
parser.add_argument("--budget", "-k", type=int, required=True)
args = parser.parse_args()


def heur_search(graph1, graph2, neighbor, i1, i2, n, k, simtimes) -> (set, set):
    blncd1 = defaultdict(set)
    blncd2 = defaultdict(set)
    for i in range(simtimes):
        for j in graph1.keys():
            current = set()
            current.add(j)
            activated = current.copy()
            active = current.copy()
            while active:
                new = set()
                for node in active:
                    current = current | set(neighbor[node])
                    if node in graph1.keys():
                        for n, p1 in graph1[node]:
                            if n not in activated and np.random.random() < p1:
                                new.add(n)
                activated.update(new)
                active = new
            blncd1[j] = i == 0 and current or blncd1[j] & current

        for j in graph2.keys():
            current = set()
            current.add(j)
            activated = current.copy()
            active = current.copy()
            while active:
                new = set()
                for node in active:
                    current = current | set(neighbor[node])
                    if node in graph2.keys():
                        for n, p2 in graph2[node]:
                            if n not in activated and np.random.random() < p2:
                                new.add(n)
                activated.update(new)
                active = new
            blncd2[j] = i == 0 and current or blncd2[j] & current

    r1, r2 = set(), set()
    r1 = reduce(lambda acc, i: acc | blncd1[i], i1, r1)
    r2 = reduce(lambda acc, i: acc | blncd2[i], i2, r2)
    init = n - len(r1 ^ r2)
    max1 = max2 = init

    s1, s2 = set(), set()
    while len(s1) + len(s2) < k:
        u1 = i1 | s1
        u2 = i2 | s2
        v1_id = v2_id = -1
        for i in graph1.keys():
            if i not in u1:
                res = n - len((r1 | blncd1[i]) ^ r2)
                if res > max1:
                    max1 = res
                    v1_id = i

        for i in graph2.keys():
            if i not in u2:
                sim = n - len(r1 ^ (r2 | blncd2[i]))
                if sim > max2:
                    max2 = sim
                    v2_id = i

        if v1_id == -1 and v2_id == -1:
            break
        elif v1_id != -1 and v2_id != -1:
            if max1 < max2:
                s2.add(v2_id)
                r2 = r2 | blncd2[v2_id]
            else:
                s1.add(v1_id)
                r1 = r1 | blncd1[v1_id]
        else:
            if v1_id == -1:
                s2.add(v2_id)
                r2 = r2 | blncd2[v2_id]
            else:
                s1.add(v1_id)
                r1 = r1 | blncd1[v1_id]

        max1 = max2 = max(max1, max2)

    # print(max(max1, max2))
    return s1, s2


def read_dataset(network_path, initial_seed_path) -> (dict, dict, dict, set, set):
    LINE = 1e-7
    with open(network_path, 'r') as f:
        n, m = map(int, f.readline().split())
        graph1, graph2, neighbor = defaultdict(list), defaultdict(list), defaultdict(list)
        for _ in range(m):
            u, v, p1, p2 = map(float, f.readline().split())
            u = int(u)
            v = int(v)
            if p1 > LINE:
                graph1[u].append((v, p1))
            if p2 > LINE:
                graph2[u].append((v, p2))
            neighbor[u].append(v)

    with open(initial_seed_path, "r") as f:
        k1, k2 = map(int, f.readline().split())
        i1 = [int(f.readline().strip()) for _ in range(k1)]
        i2 = [int(f.readline().strip()) for _ in range(k2)]

    return graph1, graph2, neighbor, set(i1), set(i2)


if __name__ == "__main__":
    start = time.time()
    graph1, graph2, neighbor, i1, i2 = read_dataset(args.network, args.initial_seed)
    bset1, bset2 = heur_search(graph1, graph2, neighbor, i1, i2, len(neighbor.keys()), args.budget, 4)

    with open(args.balanced_seed, "w") as f:
        f.write(f"{len(bset1)} {len(bset2)}\n")
        for node in bset1:
            f.write(f"{node}\n")
        for node in bset2:
            f.write(f"{node}\n")

    # print(f"Exegraphion time: {time.time() - start:.2f}s")
