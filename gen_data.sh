#!/usr/bin/env bash



for i in {1..5}; do
    srun -N 1 -C gpu ./bin/murmur3 11 > data/murmur3/tellico/fixed_run/data"$i".txt
done

