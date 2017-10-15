#!/bin/bash


for i in 100 250 1000 2500 10000; do
    for j in `seq 1 15`; do
	for k in 0.9 0.99; do
	    python discrepancy.py $i $j 0.01 $k &
	done
    done
done