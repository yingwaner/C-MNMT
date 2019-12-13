export CUDA_VISIBLE_DEVICES=${2}
python3 train.py data-bin/iwslt17/${1}-En \
    --arch transformer_wmt_en_de \
    --max-epoch 70 --fp16 --fp16-init-scale 16 \
    --task translation \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 0.0007 --min-lr 1e-09 --ddp-backend=no_c10d \
    --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.0 \
    --max-tokens 2048  --update-freq 16 \
    --no-progress-bar --log-format json --log-interval 10 \
    --save-dir checkpoints/iwslt17/single_${1}baseline |tee -a  logs/iwslt17/single_${1}baseline.log
