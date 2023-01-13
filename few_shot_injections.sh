#!/bin/bash

model_names=("EleutherAI/pythia-19m-deduped" "EleutherAI/pythia-2.7b-deduped" "EleutherAI/pythia-6.7b-deduped" "EleutherAI/pythia-13b-deduped")
injection_ds_names=("none" "imdb_10c_v_0_goodbad" "imdb_10c_v_0_positivenegative")
test_ds_names=("imdb_0_v_0_goodbad" "imdb_10c_v_0_goodbad" "imdb_10i_v_0_goodbad" "imdb_0_v_0_positivenegative" "imdb_10c_v_0_positivenegative" "imdb_10i_v_0_positivenegative")

for model_name in ${model_names[@]}; do
for injection_ds_name in ${injection_ds_names[@]}; do
for test_ds_name in ${test_ds_names[@]}; do
python few_shot_injection_measurements.py --model_name $model_name --injection_ds_name $injection_ds_name --test_ds_name $test_ds_name --batch_size 2
done
done
done

# model_names=("gpt2-xl" "EleutherAI/gpt-j-6B")
# injection_ds_names=("none" "5c_v_5i_positivenegative" "5c_v_0_positivenegative" "5i_v_0_positivenegative" "5c_v_0_goodbad")
# test_ds_names=("0_v_0_positivenegative" "5c_v_5i_positivenegative" "0_v_0_goodbad")
# adversarials=("False" "True")

# for model_name in ${model_names[@]}; do
# for adversarial in ${adversarials[@]}; do
# for injection_ds_name in ${injection_ds_names[@]}; do
# for test_ds_name in ${test_ds_names[@]}; do
# python few_shot_injection_measurements.py --model_name $model_name --injection_ds_name $injection_ds_name --test_ds_name $test_ds_name --adversarial $adversarial
# done
# done
# done
# done

# model_names=("gpt2-xl" "EleutherAI/gpt-j-6B")
# injection_ds_names=("none" "5c_v_5i_goodbad" "5c_v_0_goodbad" "5i_v_0_goodbad")
# test_ds_names=("0_v_0_goodbad" "5c_v_5i_goodbad")
# adversarials=("False" "True")

# for model_name in ${model_names[@]}; do
# for adversarial in ${adversarials[@]}; do
# for injection_ds_name in ${injection_ds_names[@]}; do
# for test_ds_name in ${test_ds_names[@]}; do
# python few_shot_injection_measurements.py --model_name $model_name --injection_ds_name $injection_ds_name --test_ds_name $test_ds_name --adversarial $adversarial
# done
# done
# done
# done

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