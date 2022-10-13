#!/usr/bin/env 
export CUDA_VISIBLE_DEVICES=$1
# export python3PATH="./"
#export PYTHON_PATH="../"

dataset=ml1m

cmd="python3 ./preprocess.py --dataset ${dataset}"
echo "Executing $cmd"
$cmd &

# cmd="python3 ./train_transe_rw.py --dataset ${dataset}"
# echo "Executing $cmd"
# $cmd 

dataset=lfm1m

cmd="python3 ./preprocess.py --dataset ${dataset}"
echo "Executing $cmd"
$cmd &

# cmd="python3 ./train_transe_rw.py --dataset ${dataset}"
# echo "Executing $cmd"
# $cmd 


dataset=cell

cmd="python3 ./preprocess.py --dataset ${dataset}"
echo "Executing $cmd"
$cmd &

# cmd="python3 ./train_transe_rw.py --dataset ${dataset}"
# echo "Executing $cmd"
# $cmd 

