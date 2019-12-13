export CUDA_VISIBLE_DEVICES=6,7
python3 train.py data-bin/iwslt17/FrZh-En \
    --arch multilingual_transformer_iwslt_de_en \
    --max-epoch 80 --fp16 --fp16-init-scale 16 \
    --task multilingual_translation --lang-pairs zh-en,fr-en \
    --share-decoders --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 0.0005 --min-lr 1e-09 --ddp-backend=no_c10d \
    --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.0001 \
    --max-tokens 4096  --update-freq 4 \
    --no-progress-bar --log-format json --log-interval 20 \
    --save-dir checkpoints/iwslt17/frzh |tee -a  logs/iwslt17/frzh.log
