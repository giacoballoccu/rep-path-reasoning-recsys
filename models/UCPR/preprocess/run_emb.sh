#!/usr/bin/env 
export CUDA_VISIBLE_DEVICES=$1
# export python3PATH="./"
#export PYTHON_PATH="../"

dataset=lfm1m

cmd="python3 ./train_transe_rw.py --dataset ${dataset}"
echo "Executing $cmd"
$cmd 

dataset=ml1m


cmd="python3 ./train_transe_rw.py --dataset ${dataset}"
echo "Executing $cmd"
$cmd 


dataset=cell


cmd="python3 ./train_transe_rw.py --dataset ${dataset}"
echo "Executing $cmd"
$cmd 


