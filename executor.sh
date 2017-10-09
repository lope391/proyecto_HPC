#!/usr/bin/env bash
EXAMPLE=paralellMain
CORES=4
mpiexec -np ${CORES} python ./${EXAMPLE}*.py
