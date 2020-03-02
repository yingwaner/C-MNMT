TEXT=/data/wanying/2.data/multilingual/DeEnItNlRo-DeEnItNlRo/preprocessed/fastBPE/iwslt17
DICT=/data/wanying/1.research/multilingual/data-bin/zhihuan_baseline
python3 preprocess.py --joined-dictionary \
        --source-lang $1 --target-lang en \
        --trainpref $TEXT/train.40k.${1}20k-en \
        --destdir data-bin/zhihuan/lowresource \
        --tgtdict data-bin/zhihuan/all/dict.en.txt \
        --workers 8 --fp16
