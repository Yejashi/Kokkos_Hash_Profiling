#!/usr/bin/env bash



for i in {1..5}; do
    srun -N 1 -C gpu ./bin/murmur3 15 > data/murmur3/tellico/data"$i".txt
done

