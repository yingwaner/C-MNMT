TEXT=/data/wanying/2.data/multilingual/DeEnItNlRo-DeEnItNlRo/preprocessed/fastBPE
DICT=/data/wanying/1.research/multilingual/data-bin/iwslt17
python3 preprocess.py --joined-dictionary \
        --source-lang $1 --target-lang en \
        --trainpref $TEXT/try.32k.${1}-en \
        --validpref $TEXT/try.32k.dev.${1}-en  \
        --testpref $TEXT/try.32k.tst.${1}-en  \
        --destdir data-bin/iwslt17/FrDe-En \
        --tgtdict data-bin/iwslt17/try.all/dict.en.txt \
        --workers 8 --fp16
