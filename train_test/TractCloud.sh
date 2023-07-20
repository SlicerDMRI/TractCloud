#!/bin/bash

# test params
test_realdata_batch_size=1024   # the batch size for testing on real data. larger batch size may further accelerate the parcellation.
k_ds_rate_for_testing=0.1       # downsample the tractography when calculating neighbor streamlines.
test_dataset_lst='dHCP ABCD HCP PPMI'  # Testing datasets in our paper (except the private tumor data). We put one example data for each testing dataset.
for test_dataset in ${test_dataset_lst}; do 
    # model weight path
    weight_path_base=../TrainedModel
    # paths for example subjects of dHCP, ABCD, HCP, PPMI 
    if [ $test_dataset = HCP ]; then
        subject_idx='101006'
        ukf_name=${subject_idx}_ukf_pp_with_region.vtp
    elif [ $test_dataset = ABCD ]; then
        subject_idx='000_WD2'
        ukf_name=sub-${subject_idx}_ses-baselineYear1Arm1_run-01_dwi_b3000_pp.vtp
    elif [ $test_dataset = PPMI ]; then
        subject_idx='3104'
        ukf_name=${subject_idx}_pp.vtp
    elif [ $test_dataset = dHCP ]; then
        subject_idx='CC00069XX12_ses-26300'
        ukf_name=sub-${subject_idx}_pp.vtp
    fi
    tractography_path=../TestData/${test_dataset}/${ukf_name}

    # run test
    out_SS_cluster_path=../parcellation_results/${test_dataset}/${subject_idx}/SS   # SS: subject space
    python test_realdata.py --weight_path_base ${weight_path_base} --tractography_path ${tractography_path} --out_path ${out_SS_cluster_path} \
    --test_realdata_batch_size ${test_realdata_batch_size} --k_ds_rate ${k_ds_rate_for_testing}

done