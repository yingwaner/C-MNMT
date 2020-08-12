#model=iwslt17/curr.threshold0.7.lossthreshold120-60.focuslang
#model=iwslt17/curr.lmscore.langrandom.itrofrdenl.shards311.validloss100.focustwice
#model=iwslt17/curr.cosdis.shardsreverse.shards321.validloss300.focustwice
#model=iwslt17/many2one.lmscore.shards311.validloss50.focustwice.4ka
#model=iwslt17/competence.cdf.shards10
#model=iwslt17/chlow.unblc.shards321.validloss300.focustwice
#model=iwslt17/chlow.unbanlance.baseline
#model=iwslt17/unblc.shards32.virozhja.validloss100.focustwice
model=iwslt17/again_baseline
#model=zhihuan/lowresource.cosdis.curr.fdzrj.shards321.focustwice.up300.reverse
#model=zhihuan/competence.zhihuan.shards10.shuffle.up1.2k.reset
list=`seq $1 $2`
for i in $list
do
	bash eval_nist.sh $i >>checkpoints/$model/result.txt
done
