#!/usr/bin/env bash

dataset="dev_clean     test_clean  train_960_text      train_960_text_oold  train_clean_100_sp dev             test_other  train_960_text_old  train_960_text_test  train_much_text"
dis_dir=/home3/yuhang001/espnet/egs2/librispeech_100/letter_bid_fine/dump/raw/
for i in $dataset ;do
python graphome.py $dis_dir/$i/text  $dis_dir/$i/phone_text_with_id   
done







