#!/bin/bash

for i in `seq 1 5`;do
    bash get_err_data_helper.sh $i &
done
