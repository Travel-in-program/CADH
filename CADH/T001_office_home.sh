#!/bin/bash
set -e

#for i in 16 32 64 128
for j in 1 2 3 4 5
do
for i in 64
do
    for domain in 'Real_World_mat_vit_b16ToArt_mat_vit_b16' 'Product_mat_vit_b16ToReal_World_mat_vit_b16' 'Real_World_mat_vit_b16ToProduct_mat_vit_b16' #'Clipart_mat_vit_b16ToReal_World_mat_vit_b16' # 'Product_mat_vit_b16ToReal_World_mat_vit_b16' 'Real_World_mat_vit_b16ToProduct_mat_vit_b16' 'Art_mat_vit_b16ToReal_World_mat_vit_b16'#'amazon_mat_vit_b16Todslr_mat_vit_b16' 'amazon_mat_vit_b16Towebcam_mat_vit_b16' 'dslr_mat_vit_b16Toamazon_mat_vit_b16' 'dslr_mat_vit_b16Towebcam_mat_vit_b16' 'webcam_mat_vit_b16Toamazon_mat_vit_b16' 'webcam_mat_vit_b16Todslr_mat_vit_b16'  #
    #for domain in 'ProductToReal_World'     amazon_mat_vit_b16   dslr_mat_vit_b16   webcam_mat_vit_b16'
    #for domain in 'Product_feature_matToReal_World_feature_mat'    'Real_World_feature_matToProduct_feature_mat' 'Art_feature_matToReal_World_feature_mat' 
    do
        python -u main1.py --nbit $i --dataset Office-Home --domain $domain #--lamda1 1 --lamda2 1 --lamda3 100 #CUDA_VISIBLE_DEVICES=1       test_file  main1
        cd matlab &&
        matlab -nojvm -nodesktop -r "demo_eval_PWCF($i, '$domain', 'Office-Home', 'T001'); quit;" &&
        cd ..
    done
done
done
#export PATH=$PATH:/media/abc/ware/njh/matlab/bin