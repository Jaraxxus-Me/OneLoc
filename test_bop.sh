#!/bin/bash

# lmo
for obj_id in 1 5 6 8 9 10 11 12
do
    python3 test.py id=$obj_id
done

# ycb
for scene_id in 48 49 50 51 52 53 54 55 56 57 58 59
do
    for obj_id in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21
    do
        python3 train.py outdir='outputs/2023-02-14/ycbv' id=$obj_id scene_id=$scene_id
    done
done