#!/bin/sh
dvc exp run -S 'models.random_forest.n_estimators=range(50, 200, 50)' \
            -S 'models.random_forest.max_depth=range(5, 20, 5)' --queue
