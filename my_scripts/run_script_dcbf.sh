#!/bin/bash -l
for test_name in dcbf_2
do
  for gamma in 0.99
  do
    for time_steps in 1 2 3 4 5 6 7 8 9 10
    do
      ./gpu_batch_dcbf.sh $test_name $gamma $time_steps
    done
  done
done
