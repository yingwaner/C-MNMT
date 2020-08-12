export CUDA_VISIBLE_DEVICES=4,5,6,7
python3 train.py data-bin/iwslt17/DeFrItNlRo-En \
    --arch multilingual_transformer_iwslt_de_en \
    --fp16 --fp16-init-scale 16 \
    --task multilingual_translation --lang-pairs fr-en,it-en \
    --share-decoders --share-decoder-input-output-embed \
    --share-encoders --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 0.0005 --min-lr 1e-09 --ddp-backend=no_c10d \
    --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.0001 \
    --max-tokens 4096  --update-freq 2 --focus-lang fr-en --process-threshold 0.7 \
    --no-progress-bar --log-format json --log-interval 5 \
    --save-dir checkpoints/iwslt17/tst1 |tee -a  logs/iwslt17/tst1.log
