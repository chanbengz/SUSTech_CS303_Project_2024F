# Project1 Information Exposure Maximization Problem

## Description

This project is to solve the Information Exposure Maximization Problem (IEMP) with Evolutionary Algorithm and Heuristic Algorithm. The IEMP is a NP problem to find a set of balanced seeds in a social network to maximize the information exposure. We have three stages in this project:

- [Evaluator](./src/Evaluator.py): to evaluate the result of given dataset including initial and balanced seeds and networks
- [IEMP_Heur](./src/IEMP_Heur.py): to solve this NP problem with Heuristic Algorithm
- [IEMP_Evol](./src/IEMP_Evol.py): to solve this NP problem with Evolutionary Algorithm

## Usage

Three program in this projects:
1. `Evaluator.py` to evaluate the result of given dataset including initial and balanced seeds and networks
2. `IEMP_{Evol, Heur}.py` are two programs to solve this NP problem with Evolutionary Algorithm and Heuristic Algorithm respectively.

Simply, you can run these programs with given makefile commands:
```bash
# run Evaluator.py
make evaluate

# run IEMP_{Evol, Heur}.py
make evol # or make heur

# or to specific the dataset
make DATASET=2 evaluate # or evol, heur
```

## Options

`Evaluator.py` is different from the other two.

## `Evaluator.py`

```
usage: Evaluator.py -n NETWORK -i INITIAL_SEED -b BALANCED_SEED -k BUDGET -o OUTPUT

options:
  --network, -n Path of Social Networks Dataset
  --initial-seed, -i Path of Initial Seed Dataset
  --balanced-seed, -b Path of Balanced Seed Dataset
  --budget, -k Budget, Length of Balanced Seeds
  --output, -o Output Path
```

## `IEMP_{Evol, Heur}.py`

```
usage: IEMP_Evol.py -n NETWORK -i INITIAL_SEED -b BALANCED_SEED -k BUDGET

options:
  --network, -n Input Path of Social Networks Dataset
  --initial-seed, -i Input Path of Initial Seed Dataset
  --balanced-seed, -b Output Path of Balanced Seed Dataset
  --budget, -k Budget, Length of Balanced Seeds
```

