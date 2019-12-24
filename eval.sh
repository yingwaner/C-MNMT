export LANGUAGE=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

model=iwslt17/curr.earlystop1000
#model=iwslt17/baseline_base
list=`seq $1 $2`
for i in $list
do
	bash eval_nist.sh $i >>checkpoints/$model/result.txt
done
