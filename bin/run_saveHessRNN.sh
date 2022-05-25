#!/bin/sh
n_runs=5

for i in $(seq $n_runs)
do
     python3 saveHess_seqMNIST.py --n_iter=10000 --print_every=250 --custom_mode=0 --comment="saveH_BPTT"
     python3 saveHess_seqMNIST.py --n_iter=16000 --print_every=400 --custom_mode=1 --comment="saveH_3Fac"         
done
