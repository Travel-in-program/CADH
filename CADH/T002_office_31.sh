#!/bin/bash
set -e

#for i in 16 32 64 128
#for j in 1 2 3 4 5 6 7 8 9 10
#do
#for mask in 0 0.05
#do
for i in 64

do
#    for domain in  'AmazonToDslr' 'AmazonToWebcam' 'DslrToAmazon' 'DslrToWebcam' 'WebcamToAmazon' 'WebcamToDslr'
    for domain in   'amazon_mat_vit_b16Towebcam_mat_vit_b16' #'amazon_mat_vit_b16Todslr_mat_vit_b16' 'dslr_mat_vit_b16Toamazon_mat_vit_b16'  #'dslr_mat_vit_b16Towebcam_mat_vit_b16' 'webcam_mat_vit_b16Toamazon_mat_vit_b16'  'webcam_mat_vit_b16Todslr_mat_vit_b16'
    do
        python -u main1.py --nbit $i --dataset Office-31 --domain $domain  #CUDA_VISIBLE_DEVICES=1   test_file     --lamda1 0.1 --lamda2 0.01 --lamda3 0.1
        cd matlab &&
        matlab -nojvm -nodesktop -r "demo_eval_PWCF($i, '$domain', 'Office-31', 'T002_test'); quit;" &&
        cd ..
    done
done
#done
#done