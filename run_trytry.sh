export CUDA_VISIBLE_DEVICES=4,5,6,7
path=curr
output=curr.shards10.earlystop50.reset-05
#output=lowresource.curr.numfocus.fortest
assign=iwslt17

cp data-bin/$assign/zero/train.* data-bin/$assign/$path
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
for i in $(seq 0 4)
do
for j in $(seq 0 9)
do
l=${lang[$i]}
#index=`echo "scale=2;$i*5+$j"|bc`
#up=`echo "scale=2;$index*300+300"|bc`
#up=${update[$index]}
#cp data-bin/$assign/baseline/train.${l}-en.* data-bin/$assign/$path
cp data-bin/iwslt17/shards10.shuffle_noEN/$l-$j/train.${l}-en.* data-bin/iwslt17/$path
python3 train.py data-bin/$assign/$path \
    --arch multilingual_transformer \
    --fp16 --focus-lang ${l}-en \
    --task multilingual_translation --lang-pairs fr-en,it-en,ro-en,nl-en,de-en \
    --share-decoders --share-decoder-input-output-embed \
    --share-encoders --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 0.0007 --min-lr 1e-09 --ddp-backend=no_c10d \
    --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --dropout 0.3 --clip-norm 0.0 \
    --max-tokens 4096  --update-freq 2 \
    --no-progress-bar --log-format json --log-interval 20 \
    --earlystop-max-update 50 \
    --save-dir checkpoints/$assign/$output |tee -a  logs/$assign/$output.log
done
done

python3 train.py data-bin/$assign/$path \
    --arch multilingual_transformer \
    --max-update 3000 --fp16 \
    --task multilingual_translation --lang-pairs fr-en,ro-en,de-en,ja-en,zh-en \
    --share-decoders --share-decoder-input-output-embed \
    --reset-lr-scheduler --reset-optimizer \
    --share-encoders --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 0.0007 --min-lr 1e-09 --ddp-backend=no_c10d \
    --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --dropout 0.3 --clip-norm 0.0 \
    --max-tokens 4096  --update-freq 2 \
    --no-progress-bar --log-format json --log-interval 20 \
    --save-dir checkpoints/$assign/$output |tee -a  logs/$assign/$output.log

