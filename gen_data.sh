#!/usr/bin/env bash


srun -N 1 -C gpu ./bin/murmur3 15 > data/murmur3/data.txt
