DATASET=1

all: install run

run: evaluate

install:
	pip install -r requirements.txt

evaluate:
	python3 src/Evaluator.py -n test/Evaluator/map${DATASET}/dataset${DATASET} -i test/Evaluator/map${DATASET}/seed -b test/Evaluator/map${DATASET}/seed_balanced -k 15 -o test/Evaluator/map${DATASET}/output

.phony: all install run
