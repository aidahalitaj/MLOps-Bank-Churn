#!/bin/sh
dvc exp run -S 'model.random_forest.n_estimators=range(50, 200, 50)' \
            -S 'model.random_forest.max_depth=range(5, 20, 5)' --queue
