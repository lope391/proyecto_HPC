#!/usr/bin/env bash
PROCESS=paralellMain
CORES=4
mpiexec -np ${CORES} python3 ./${PROCESS}*.py
