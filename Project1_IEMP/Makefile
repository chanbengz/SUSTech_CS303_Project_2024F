DATASET=1

all: install run

run: evaluate evol heur

install:
	pip install -r requirements.txt

evaluate:
	python3 src/Evaluator.py -n test/Evaluator/map${DATASET}/dataset${DATASET} -i test/Evaluator/map${DATASET}/seed -b test/Evaluator/map${DATASET}/seed_balanced -k 15 -o test/Evaluator/map${DATASET}/output

heur:
	python3 src/IEMP_Heur.py -n test/Heuristic/map${DATASET}/dataset${DATASET} -i test/Heuristic/map${DATASET}/seed -b test/Heuristic/map${DATASET}/seed_balanced -k 15

evol:
	python3 src/IEMP_Evol.py -n test/Evolutionary/map${DATASET}/dataset${DATASET} -i test/Evolutionary/map${DATASET}/seed -b test/Evolutionary/map${DATASET}/seed_balanced -k 15

.phony: all install run
