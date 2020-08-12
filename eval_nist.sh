# !/bin/bash
#model=iwslt17/curr.threshold0.7.lossthreshold120-60.focuslang
#model=iwslt17/curr.cosdis.shardsreverse.shards321.validloss300.focustwice
#model=iwslt17/competence.cdf.shards10
model=iwslt17/again_baseline
#model=zhihuan/competence.zhihuan.shards10.shuffle.up1.2k.reset
#model=zhihuan/lowresource.cosdis.curr.fdzrj.shards33.focustwice.up300
CKPT_DIR=/data/wanying/1.research/multilingual/checkpoints/$model
OUT_DIR=/data/wanying/1.research/multilingual/checkpoints/$model/test
GPU=4

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
ALPHA=1.4
BATCH=100
DATABIN=/data/wanying/1.research/multilingual/data-bin/iwslt17/curr

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

    for i in $(seq 2 $max); do
        t=${testset[$i]}
        r=${refset[$i]}
        cat /data/wanying/2.data/multilingual/DeEnItNlRo-DeEnItNlRo/preprocessed/fastBPE/iwslt17_ori/tst.32k.$t-en.$t \
        | CUDA_VISIBLE_DEVICES=$GPU python3 interactive.py $DATABIN --path $m \
        --task multilingual_translation --source-lang $t --target-lang en \
        --buffer-size 2000 --lang-pairs fr-en,it-en,ro-en,nl-en,de-en \
        --beam $BEAM --lenpen $ALPHA --batch-size $BATCH --remove-bpe \
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
