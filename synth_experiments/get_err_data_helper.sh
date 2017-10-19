#!/bin/bash

for i in `seq 1 5`; do
    python discrepancy ${i}_$1
done