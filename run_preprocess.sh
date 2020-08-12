TEXT=/data/wanying/2.data/multilingual/DeEnItNlRo-DeEnItNlRo/preprocessed/fastBPE/euro
DICT=/data/wanying/1.research/multilingual/data-bin/
declare -a lang=('cs' 'el' 'hu' 'lt' 'lv' 'pl' 'pt' 'sk' 'sl' 'sv' 'es' 'fi')
for i in $(seq 0 11); do
    l=${lang[$i]}
    python3 preprocess.py --joined-dictionary \
        --source-lang $l --target-lang en \
        --trainpref $TEXT/fil0.train.90k.${l}-en \
        --destdir data-bin/iwslt17/euro_0 \
        --tgtdict data-bin/iwslt17/euro_al/dict.en.txt \
        --workers 32 --fp16
    done
