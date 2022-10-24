#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_clean_100"
valid_set="dev"
test_sets=" dev_clean "

asr_tag=conformer_lr2e-3_warmup15k_amp_nondeterministic_start_33_add_bert
# asr_tag=conformer_lr2e-3_warmup15k_amp_nondeterministic
asr_config=conf/train_asr.yaml
inference_config=conf/decode_asr.yaml

# ls -all  ../lm_pre/ > log_temp
# sleep 7200 &
# wai
# ls -all  ../lm_pre/ > log_temp2

./asr.sh \
    --skip_data_prep false \
    --skip_train false \
    --skip_eval false \
    --lang en \
    --ngpu 1 \
    --nj 20 \
    --inference_nj 10 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --audio_format "flac.ark" \
    --feats_type raw \
    --use_lm false \
    --asr_tag "${asr_tag}" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --lm_train_text "data/${train_set}/text" \
    --bpe_train_text "data/${train_set}/text" "$@" \
    --stage 11 \
    --stop_stage 13
