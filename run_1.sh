#TEXT=/data/wanying/2.data/multilingual/DeEnItNlRo-DeEnItNlRo/preprocessed/fastBPE/iwslt17
work=shards5.shuffle
TEXT=/data/wanying/3.tools/srilm/bin/i686-m64/output/$work
DICT=/data/wanying/1.research/multilingual/data-bin/iwslt17
#rm data-bin/iwslt17/curr/dict.${1}.txt
declare -a lang=('fr' 'it' 'ro' 'nl' 'de')
for j in $(seq 0 4)
do
l=${lang[$j]}
for i in $(seq 0 4)
do
python3 preprocess.py --joined-dictionary \
        --source-lang ${l} --target-lang en \
        --trainpref $TEXT/${l}-en.$i \
        --destdir data-bin/iwslt17/$work/$l-$i \
        --tgtdict data-bin/iwslt17/curr/dict.en.txt \
        --workers 8 --fp16
done
done
