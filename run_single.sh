export CUDA_VISIBLE_DEVICES=${2}
python3 train.py data-bin/zhihuan/${1}-En \
    --arch transformer_wmt_en_de \
    --max-epoch 70 --fp16 \
    --task translation --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 0.0007 --min-lr 1e-09 --ddp-backend=no_c10d \
    --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.0 \
    --max-tokens 4096  --update-freq 8 \
    --no-progress-bar --log-format json --log-interval 20 \
    --save-dir checkpoints/zhihuan/single_${1}baseline |tee -a  logs/zhihuan/single_${1}baseline.log
