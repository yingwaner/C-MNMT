export CUDA_VISIBLE_DEVICES=4,5,6,7
python3 train.py data-bin/iwslt17/DeFrItNlRo-En \
    --arch multilingual_transformer \
    --max-epoch 60 --fp16 \
    --task multilingual_translation --lang-pairs de-en,fr-en,it-en,nl-en,ro-en \
    --share-encoders --share-decoders \
    --share-all-embeddings --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 0.0007 --min-lr 1e-09 --ddp-backend=no_c10d \
    --dropout 0.3 \
    --weight-decay 0.0 --clip-norm 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 2048  --update-freq 4 \
    --no-progress-bar --log-format json --log-interval 10 \
    --save-dir checkpoints/iwslt17/baseline_base |tee -a  logs/iwslt17/baseline_base.log
