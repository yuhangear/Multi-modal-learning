#!/usr/bin/env bash

./slurm.pl --quiet --gpu 1  --num-threads 10 --exclude=node0[1-8]  exp/log1  python dict.py
