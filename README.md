# SP-DARTS
Source code for SP-DARTS

## version: pytorch 1.7.0

## NAS-BENCH-201:  
`
cd AutoDL-Projects_sp/;
bash ./scripts-search/algos/[DARTS-V1_100.sh|DARTS-V1_10.sh] [cifar10|cifar100] 1 -1

## DARTS search space:  
`
cd SP-DARTS/;
python train_search_mf.py  
`

## S1-S4 search space:  
`
cd SmoothDARTS/sota/cnnmul;
python train_search.py --search_space=[s1|s2|s3|s4]  --temp_start [0.0007|0.0006|0.0005|0.0004] --dataset [cifar10|cifar100|svhn]
`