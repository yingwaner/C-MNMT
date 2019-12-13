# !/bin/bash
model=iwslt17/curr.update1.5k.finetune
CKPT_DIR=/data/wanying/1.research/multilingual/checkpoints/$model
OUT_DIR=/data/wanying/1.research/multilingual/checkpoints/$model/test
GPU=6

mkdir -p $OUT_DIR

declare -a testset=('fr' 'it' 'ro' 'nl' 'de')
declare -a refset=(
       '/data/wanying/2.data/multilingual/DeEnItNlRo-DeEnItNlRo/preprocessed/tst.fr-en.en.tok'
       '/data/wanying/2.data/multilingual/DeEnItNlRo-DeEnItNlRo/preprocessed/tst.it-en.en.tok'
       '/data/wanying/2.data/multilingual/DeEnItNlRo-DeEnItNlRo/preprocessed/tst.ro-en.en.tok'
       '/data/wanying/2.data/multilingual/DeEnItNlRo-DeEnItNlRo/preprocessed/tst.nl-en.en.tok'
       '/data/wanying/2.data/multilingual/DeEnItNlRo-DeEnItNlRo/preprocessed/tst.de-en.en.tok'
		)
valid_ref='/data/wanying/pretrain/pure/mt02_u8.en.low'
v='valid'
len=${#testset[@]}
max=`echo $len-1|bc`
script='/data/wanying/pretrain/eval/multi-bleu.perl'
BEAM=5
ALPHA=0.6
BATCH=250
DATABIN=/data/wanying/1.research/multilingual/data-bin/iwslt17/DeFrItNlRo-En

#for m in $CKPT_DIR/checkpoint9.pt; do
#for m in $CKPT_DIR/*.pt; do
m=$CKPT_DIR/checkpoint$1.pt
#    if [ ! -f $m ]; then
#        continue
#    fi
    m_name=`basename ${m}`
    echo $m_name
    sum=0
#    t=valid
#    CUDA_VISIBLE_DEVICES=$GPU python generate.py $DATABIN --path $m --gen-subset valid --beam $BEAM --batch-size $BATCH --remove-bpe --lenpen $ALPHA --log-format=none > $OUT_DIR/${m_name}.$t
#	python3 choose-translation.py  $OUT_DIR/${m_name}.$t  $OUT_DIR/${m_name}.$t.out
#    bleu=`perl $script  $valid_ref < $OUT_DIR/${m_name}.$t.out`
#    echo -e "\t$v\t$bleu"

    for i in $(seq 0 $max); do
        t=${testset[$i]}
        r=${refset[$i]}
        cat /data/wanying/2.data/multilingual/DeEnItNlRo-DeEnItNlRo/preprocessed/fastBPE/iwslt17/tst.32k.$t-en.$t \
        | CUDA_VISIBLE_DEVICES=$GPU fairseq-interactive $DATABIN --path $m \
        --task multilingual_translation --source-lang $t --target-lang en \
        --buffer-size 2000 --lang-pairs de-en,fr-en,it-en,nl-en,ro-en \
        --beam $BEAM --batch-size $BATCH --remove-bpe \
        --log-format=none > $OUT_DIR/${m_name}.$t
        #grep ^H $OUT_DIR/${m_name}.$t | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > $OUT_DIR/${m_name}.$t.out
	python3 choose-translation.py  $OUT_DIR/${m_name}.$t  $OUT_DIR/${m_name}.$t.out
       bleu=`perl $script  $r < $OUT_DIR/${m_name}.$t.out`
       num=`echo $bleu|sed "s/.*BLEU\ =\ \([0-9.]\{1,\}\).*/\1/"`
        sum=`echo "scale=2;$sum+$num"|bc`
        echo -e "\t$t\t$bleu"
    done    
    avg=`echo "scale=2;$sum/$len"|bc`
    echo -e "\tVTAVG\t$avg" 
#done
