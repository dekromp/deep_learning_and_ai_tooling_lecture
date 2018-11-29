#!/bin/bash

# Simple way to automate experiment execution.
for batch_size in 8 16 32
do
    for l2_factor in 1e-4 1e-3 1e-2
    do
        for learning_rate in 1e-2 1e-1 1
        do
            echo Executing: keras_example.py -ep 200 -bs $batch_size -l2 $l2_factor -lr $learning_rate -d "./experiments/modelV1_bs[$batch_size]_l2[$l2_factor]_lr[$learning_rate]"
            python keras_example.py -ep 200 -bs $batch_size -l2 $l2_factor -lr $learning_rate -d "./experiments/modelV1_bs[$batch_size]_l2[$l2_factor]_lr[$learning_rate]"
        done
    done
done
