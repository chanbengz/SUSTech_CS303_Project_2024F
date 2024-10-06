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
    blncd1, blncd2 = defaultdict(set), defaultdict(set)
    for i in range(simtimes):
        for j in graph1.keys():
            current = set()
            current.add(j)
            activated = current.copy()
            active = current.copy()
            while active:
                new = set()
                for node in active:
                    current |=  set(neighbor[node])
                    if node not in graph1.keys():
                        continue
                    for nodei, p1 in graph1[node]:
                        if nodei not in activated and np.random.random() < p1:
                            new.add(nodei)
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
                    current |= set(neighbor[node])
                    if node not in graph2.keys():
                        continue
                    for nodei, p2 in graph2[node]:
                        if nodei not in activated and np.random.random() < p2:
                            new.add(nodei)
                activated.update(new)
                active = new
            blncd2[j] = i == 0 and current or blncd2[j] & current

    r1, r2 = set(), set()
    s1, s2 = set(), set()
    r1 = reduce(lambda acc, i: acc | blncd1[i], i1, r1)
    r2 = reduce(lambda acc, i: acc | blncd2[i], i2, r2)
    init = n - len(r1 ^ r2)
    max1 = max2 = init

    while len(s1) + len(s2) < k:
        u1, u2 = i1 | s1, i2 | s2
        id1 = id2 = -1
        for i in [_ for _ in graph1.keys() if _ not in u1]:
            res = n - len((r1 | blncd1[i]) ^ r2)
            if res > max1:
                max1 = res
                id1 = i

        for i in [_ for _ in graph2.keys() if _ not in u2]:
            res = n - len(r1 ^ (r2 | blncd2[i]))
            if res > max2:
                max2 = res
                id2 = i

        if id1 == -1 and id2 == -1:
            break

        if id1 != -1 and id2 != -1:
            if max1 < max2:
                s2.add(id2)
                r2 |= blncd2[id2]
            else:
                s1.add(id1)
                r1 |= blncd1[id1]
        elif id1 == -1:
            s2.add(id2)
            r2 |= blncd2[id2]
        else:
            s1.add(id1)
            r1 |= blncd1[id1]

        max1 = max2 = max(max1, max2)

    print(max(max1, max2))
    return s1, s2


def read_dataset(network_path, initial_seed_path) -> (dict, dict, dict, set, set):
    with open(network_path, 'r') as f:
        _, m = map(int, f.readline().split())
        graph1, graph2, neighbor = defaultdict(list), defaultdict(list), defaultdict(list)
        for _ in range(m):
            u, v, p1, p2 = map(float, f.readline().split())
            u, v = int(u), int(v)
            if p1 > 1e-7:
                graph1[u].append((v, p1))
            if p2 > 1e-7:
                graph2[u].append((v, p2))
            neighbor[u].append(v)

    with open(initial_seed_path, "r") as f:
        k1, k2 = map(int, f.readline().split())
        init1 = [int(f.readline().strip()) for _ in range(k1)]
        init2 = [int(f.readline().strip()) for _ in range(k2)]

    return graph1, graph2, neighbor, set(init1), set(init2)


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

    print(f"Execution time: {time.time() - start:.2f}s")
