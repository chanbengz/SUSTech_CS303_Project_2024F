import time
import copy
from collections import defaultdict
import numpy as np
from functools import reduce

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--network", "-n", type=str, required=True)
parser.add_argument("--initial-seed", "-i", type=str, required=True)
parser.add_argument("--balanced-seed", "-b", type=str, required=True)
parser.add_argument("--budget", "-k", type=int, required=True)
args = parser.parse_args()

S = 0
N = 100
global bln1, bln2, res1, res2, nodes, sl1, sl2, s_bln1, s_bln2, Ns, prob, dicnum, probw


def read_dataset(network_path, initial_seed_path):
    with open(network_path, 'r') as f:
        _, m = map(int, f.readline().split())
        graph1, graph2, neighbor = defaultdict(list), defaultdict(list), defaultdict(list)
        for _ in range(m):
            u, v, p1, p2 = map(float, f.readline().split())
            u = int(u)
            v = int(v)
            if p1 > 1e-5:
                graph1[u].append((v, p1))
            if p2 > 1e-5:
                graph2[u].append((v, p2))
            neighbor[u].append(v)

    with open(initial_seed_path, 'r') as f:
        k1, k2 = map(int, f.readline().split())
        init1 = [int(f.readline()) for _ in range(k1)]
        init2 = [int(f.readline()) for _ in range(k2)]

    return graph1, graph2, neighbor, set(init1), set(init2)

def simulate(graph1, graph2, neighbor, simtimes):
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
            blncd1[j] = i == 0 and current or (blncd1[j] & current)

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
            blncd2[j] = i == 0 and current or (blncd2[j] & current)
    return blncd1, blncd2


def generate_structure(nums):
    pop = []
    for _ in range(nums):
        tmp = []
        k_num = np.random.randint(args.budget - 3, args.budget + 1)
        indices = np.random.choice(dicnum, size=k_num, p=probw, replace=False).tolist()
        for index in indices:
            i = index // 2
            if index & 1 == 0 and i < len(s_bln1):
                tmp.append(s_bln2[i] + nodes if i > sl1 else s_bln1[i])
            elif i < len(s_bln2):
                tmp.append(s_bln1[i] if i > sl2 else s_bln2[i] + nodes)
        pop.append((-0.1, tmp))
    return pop


def fitness(sample):
    r1, r2 = copy.deepcopy(res1), copy.deepcopy(res2)
    s1 = [_ for _ in sample if _ < nodes]
    s2 = [_ - nodes for _ in sample if _ >= nodes]
    r1 = reduce(lambda acc, i: acc | bln1[i], s1, r1)
    r2 = reduce(lambda acc, i: acc | bln2[i], s2, r2)

    blnmax = nodes - len(r1 ^ r2)
    if len(sample) > args.budget:
        return -blnmax, sample
    return blnmax, sample


def mutation(sample):
    mut_type = np.random.randint(2)
    sample_size = len(sample)
    if mut_type == 0 or sample_size < 3:
        return sample

    if mut_type == 1:
        is_add = np.random.rand()
        if is_add < 0.5 and sample_size < args.budget:
            val = np.random.randint(S)
            sample.append(val)
        elif is_add >= 0.5:
            sample.pop(np.random.randint(sample_size))
    elif mut_type == 2:
        sample.pop(np.random.randint(sample_size))
        val = np.random.randint(S)
        sample.append(val)
    return sample


def new_population(p):
    for i, gen in enumerate(p):
        if gen[0] == -0.1:
            p[i] = fitness(gen[1])
    sorted_list = sorted(p, key=lambda x: x[0], reverse=True)
    return sorted_list[:N]


def new_son(fa):
    draw = np.random.choice(Ns, size=2, p=prob, replace=False)
    child1, child2 = [], []
    c_point = np.random.randint(nodes * 2)
    for i in fa[draw[0] - 1][1]:
        (child1 if i < c_point else child2).append(i)
    for i in fa[draw[1] - 1][1]:
        (child2 if i < c_point else child1).append(i)
    son1 = mutation(sorted(child1))
    son2 = mutation(sorted(child2))
    return (-0.1, son1), (-0.1, son2)


def generate_probability(nums):
    pr = [1 / (i + 1) for i in range(nums)]
    pr = [p / sum(pr) for p in pr]
    return pr


if __name__ == "__main__":
    start = time.time()
    graph1, graph2, neighbor, i1, i2 = read_dataset(args.network, args.initial_seed)
    bln1, bln2 = simulate(graph1, graph2, neighbor, 3)

    nodes = len(neighbor.keys())
    Ns = list(range(1, N + 1))
    sl1, sl2 = len(bln1), len(bln2)
    S = sl1 + sl2
    prob, probw = generate_probability(N), generate_probability(S)

    res1, res2 = set(), set()
    res1 = reduce(lambda acc, i: acc | bln1[i], i1, res1)
    res2 = reduce(lambda acc, i: acc | bln2[i], i2, res2)
    dicnum = list(range(1, S + 1))
    s_bln1 = list(dict(sorted(bln1.items(), key=lambda x: len(x[1]), reverse=True)).keys())
    s_bln2 = list(dict(sorted(bln2.items(), key=lambda x: len(x[1]), reverse=True)).keys())

    # Evolve
    pop = generate_structure(N * 2)
    pop = new_population(pop)
    for _ in range(20):
        new_pop = []
        for _ in range(N):
            sons = new_son(pop)
            new_pop.append(sons[0])
            new_pop.append(sons[1])
        pop = new_population(new_pop)

    set1 = [_ for _ in pop[0][1] if _ < nodes]
    set2 = [_ - nodes for _ in pop[0][1] if _ >= nodes]

    with open(args.balanced_seed, 'w') as f:
        f.write(f"{len(set1)} {len(set2)}\n")
        for num in set1:
            f.write(f"{num}\n")
        for num in set2:
            f.write(f"{num}\n")

    print(f"Execution time: {time.time() - start:.2f}s")
