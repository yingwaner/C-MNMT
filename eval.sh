model=iwslt17/curr.update1.5k.finetune30
#model=iwslt17/baseline_base
list=`seq $1 $2`
for i in $list
do
	bash eval_nist.sh $i >>checkpoints/$model/result.txt
done
