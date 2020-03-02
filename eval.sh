#model=iwslt17/curr.earlystop500.reset-05.random
model=zhihuan/curr+label
list=`seq $1 $2`
for i in $list
do
	bash eval_nist.sh $i >>checkpoints/$model/result.txt
done
