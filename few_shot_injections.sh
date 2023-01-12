#!/bin/bash

model_names=("gpt2-xl" "EleutherAI/gpt-j-6B")
injection_ds_names=("none" "5c_v_5i_goodbad" "5c_v_0_goodbad" "5i_v_0_goodbad")
test_ds_names=("0_v_0_goodbad" "5c_v_5i_goodbad")
adversarials=("False" "True")

for model_name in ${model_names[@]}; do
for adversarial in ${adversarials[@]}; do
for injection_ds_name in ${injection_ds_names[@]}; do
for test_ds_name in ${test_ds_names[@]}; do
python few_shot_injection_measurements.py --model_name $model_name --injection_ds_name $injection_ds_name --test_ds_name $test_ds_name --adversarial $adversarial
done
done
done
done

# model_name="distilgpt2"
# injection_ds_names=(""5c_v_5i_goodbad 5c_v_0_goodbad 5i_v_0_goodbad)
# test_ds_names=(0_v_0_goodbad 5c_v_5i_goodbad)
# adversarials=(False True)

# for adversarial in ${adversarials[@]}; do
# for injection_ds_name in ${injection_ds_names[@]}; do
# for test_ds_name in ${test_ds_names[@]}; do
# python few_shot_injection_measurements.py --model_name $model_name --injection_ds_name $injection_ds_name --test_ds_name $test_ds_name --adversarial $adversarial
# done
# done
# done