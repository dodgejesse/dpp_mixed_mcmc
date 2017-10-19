#!/bin/bash

for i in `seq 1 20`; do
    python discrepancy.py ${i}_$1
done