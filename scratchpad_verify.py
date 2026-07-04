import pandas as pd, numpy as np
df=pd.read_csv('data/train.csv'); df.columns=[c.strip().lstrip('﻿') for c in df.columns]
for c in df.columns:
    if c!='id': df[c]=pd.to_numeric(df[c],errors='coerce')
d=df.fillna(0)
def R(x): return pd.Series(x).rank().values
k=100000
def top(a): return set(np.argsort(-np.asarray(a))[:k])
def ov(a,b): 
    return len(top(a)&top(b))/k

f=lambda n: d['f'+str(n)].values
# base dollar P&L pieces
net_inter=0.018*(f(7)+f(8)+f(10))+0.010*(f(6)+f(9))
interest=0.16*f(1)
fee=625*f(20)
loss=-0.90*f(11)*f(1)
benefit=-(f(14)+f(16)+50*f(13)+15*f(15))
distress=-200*f(3)-25*f(2)
core=net_inter+interest+fee+loss+distress
WITH=core+benefit
WITHOUT=core
B04=core+0.4*benefit
# honest 0768
sig=f(6)+f(7)+f(8)+f(9)+f(10)
honest=0.573*R(sig)+0.427*R(f(1))-0.18*R(f(11)*f(1))-0.15*R(f(3))-0.06*R(f(21))
print('WITH vs WITHOUT',ov(WITH,WITHOUT))
print('honest vs WITH',ov(honest,WITH))
print('honest vs WITHOUT',ov(honest,WITHOUT))
print('honest vs 0.4x',ov(honest,B04))
print('0.4x vs WITH',ov(B04,WITH))
print('0.4x vs WITHOUT',ov(B04,WITHOUT))

print('\n--- direction vs proven anchors ---')
revolve=f(1)
rawsum=sig
# net-interchange+interest+fee+loss+distress vs proven single axes
print('rawsum vs WITH   ',ov(rawsum,WITH))
print('rawsum vs WITHOUT',ov(rawsum,WITHOUT))
print('rawsum vs 0.4x   ',ov(rawsum,B04))
print('revolve vs WITH   ',ov(revolve,WITH))
print('revolve vs WITHOUT',ov(revolve,WITHOUT))
print('revolve vs 0.4x   ',ov(revolve,B04))
# best measured combo anchor 0.573*spend+0.427*revolve (raw, not rank) LB0.726
combo=0.573*rawsum+0.427*revolve
print('combo726 vs WITH   ',ov(combo,WITH))
print('combo726 vs WITHOUT',ov(combo,WITHOUT))
print('combo726 vs 0.4x   ',ov(combo,B04))

# who swaps: WITHOUT-promoted (in WITHOUT top not WITH top)
tw=top(WITH); two=top(WITHOUT)
promo=list(two-tw)  # promoted by dropping subtraction
demo=list(tw-two)
idx=np.array(promo)
print('\n# swapped',len(promo))
print('PROMOTED-by-dropping: spend %.0f revolve %.0f f20 %.2f benefitcost %.0f'%(
  rawsum[idx].mean(),f(1)[idx].mean(),f(20)[idx].mean(),(f(14)+f(16)+50*f(13)+15*f(15))[idx].mean()))
idxd=np.array(demo)
print('DEMOTED-by-dropping : spend %.0f revolve %.0f f20 %.2f benefitcost %.0f'%(
  rawsum[idxd].mean(),f(1)[idxd].mean(),f(20)[idxd].mean(),(f(14)+f(16)+50*f(13)+15*f(15))[idxd].mean()))
print('POP                 : spend %.0f revolve %.0f f20 %.2f benefitcost %.0f'%(
  rawsum.mean(),f(1).mean(),f(20).mean(),(f(14)+f(16)+50*f(13)+15*f(15)).mean()))
