export CUDA_VISIBLE_DEVICES=4,5
path=curr1
output=codes
#cp data-bin/iwslt17/zeroshards/train.* data-bin/iwslt17/$path
declare -a lang=('fr' 'it' 'ro' 'nl' 'de')
#declare -a update=(
#    '500' '1000' '1500' '2000' '2500' '3000' '3500' '4000' '4500' '5000' '5500' '6000' '6500' '7000' '15000'
#    )
#    '150' '300' '450' '600' '750' '900' '1050' '1200' '1350' '1500' \
#    '1650' '1800' '1950' '2100' '2250' '2400' '2550' '2700' '2850' '3000' \
#    '3150' '3300' '3450' '3600' '3750' '3900' '4050' '4200' '4350' '4500' \
#    '4650' '4800' '4950' '5100' '5250' '5400' '5550' '5700' '5850' '6000' \
#    '6150' '6300' '6450' '6600' '6750' '6900' '7050' '7200' '7350' '15000')
#for j in $(seq 0 4)
#do
#for i in $(seq 0 4)
#do
#for j in $(seq 0 4)
#do
l=${lang[$i]}
#index=`echo "scale=2;$i*5+$j"|bc`
#up=`echo "scale=2;$index*300+300"|bc`
#up=${update[$index]}
#cp data-bin/iwslt17/DeFrItNlRo-En/train.${l}-en.* data-bin/iwslt17/$path
#cp data-bin/iwslt17/shards5.shuffle/$l-$j/train.${l}-en.* data-bin/iwslt17/$path
python3 train.py data-bin/iwslt17/$path \
    --arch multilingual_transformer \
    --max-update 1500 --fp16 --fp16-init-scale 16 \
    --task multilingual_translation --lang-pairs fr-en,it-en,ro-en,nl-en,de-en \
    --share-decoders --share-decoder-input-output-embed \
    --share-encoders --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 0.0007 --min-lr 1e-09 --ddp-backend=no_c10d \
    --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.0 \
    --max-tokens 4096  --update-freq 4 \
    --no-progress-bar --log-format json --log-interval 10 \
    --earlystop-max-update 10 \
    --save-dir checkpoints/iwslt17/$output |tee -a  logs/iwslt17/$output.log
#done
#done
"""
python3 train.py data-bin/iwslt17/$path \
    --arch multilingual_transformer \
    --max-update 7500 --fp16 --fp16-init-scale 16 \
    --task multilingual_translation --lang-pairs fr-en,it-en,ro-en,nl-en,de-en \
    --share-decoders --share-decoder-input-output-embed \
    --share-encoders --share-all-embeddings \
    --reset-lr-scheduler --reset-optimizer \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 0.000175 --min-lr 1e-09 --ddp-backend=no_c10d \
    --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.0 \
    --max-tokens 4096  --update-freq 4 \
    --no-progress-bar --log-format json --log-interval 10 \
    --save-dir checkpoints/iwslt17/$output |tee -a  logs/iwslt17/$output.log
"""
