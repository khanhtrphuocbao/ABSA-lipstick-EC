"""Render deterministic figures from the reconciled reporting dataset."""
import math,statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from src.analysis.test02d.sources import CANDIDATES,SEEDS,ASPECTS,SENTIMENTS,LABELS,SUBSETS,mean_sd

COLORS=('#0072B2','#D55E00','#009E73'); MARKERS=('o','s','^')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120,'savefig.dpi':220,'svg.hashsalt':'test02d'})
def values(d,key,c):return [float(d[key][c,str(s)]['joint_macro_f1']) for s in SEEDS]
def finish(fig):fig.tight_layout();return fig

def make_figure(number,d):
 if number==1:
  fig,ax=plt.subplots(1,2,figsize=(9,3.5),sharey=True)
  for j,(key,title) in enumerate([('joint_macro_f1','Fixed-21 Joint Macro-F1'),('joint_micro_f1','Joint Micro-F1')]):
   for i,c in enumerate(CANDIDATES):
    m,s=mean_sd([float(d['model'][c,str(seed)][key]) for seed in SEEDS]);ax[j].errorbar(i,m,yerr=s,color=COLORS[i],marker=MARKERS[i],capsize=4)
   ax[j].set_xticks(range(3),CANDIDATES,rotation=15);ax[j].set_title(title);ax[j].set_ylim(0,1);ax[j].set_ylabel('F1 (mean ± sample SD)')
 elif number==2:
  fig,ax=plt.subplots(figsize=(7,3.8))
  for i,c in enumerate(CANDIDATES):
   val=mean_sd(d['validation']['values'][c]);test=mean_sd(values(d,'model',c))
   ax.plot([i-.12,i+.12],[val[0],test[0]],color=COLORS[i],alpha=.7)
   for offset,(mean,sd),mark in [(-.12,val,'o'),(.12,test,'s')]:ax.errorbar(i+offset,mean,yerr=sd,color=COLORS[i],marker=mark,capsize=4)
  ax.set_xticks(range(3),CANDIDATES);ax.set_ylim(0,1);ax.set_ylabel('Fixed-21 Joint Macro-F1');ax.set_title('Validation ○  /  Official test □')
 elif number==3:
  fig,axes=plt.subplots(1,3,figsize=(11,3.5),sharey=True)
  for j,sent in enumerate(SENTIMENTS):
   for i,c in enumerate(CANDIDATES):
    for k,metric in enumerate(('precision','recall','f1')):
     vals=[None if d['sentiment'][c,str(seed),sent][metric]=='' else float(d['sentiment'][c,str(seed),sent][metric]) for seed in SEEDS]; m,sd=mean_sd(vals)
     if m is not None:axes[j].errorbar(k+(i-1)*.14,m,yerr=sd,color=COLORS[i],marker=MARKERS[i],capsize=2,label=c if k==0 else None)
   axes[j].set_xticks(range(3),['Precision','Recall','F1'],rotation=20);axes[j].set_title(sent.title());axes[j].set_ylim(0,1)
  axes[0].set_ylabel('Pooled Joint score');axes[-1].legend(fontsize=7)
 elif number==4:
  fig,ax=plt.subplots(figsize=(7,10)); matrix=[];labels=[]
  for label in LABELS:
   support=int(d['pairs'][CANDIDATES[0],'42',label]['support']);labels.append(f'{label}  (n={support})')
   matrix.append([statistics.mean(float(d['pairs'][c,str(s),label]['f1']) for s in SEEDS) for c in CANDIDATES])
  img=ax.imshow(matrix,vmin=0,vmax=1,cmap='viridis',aspect='auto');ax.set_xticks(range(3),CANDIDATES,rotation=15);ax.set_yticks(range(21),labels)
  for i,row in enumerate(matrix):
   for j,v in enumerate(row):ax.text(j,i,f'{v:.2f}',ha='center',va='center',color='white' if v<.55 else 'black',fontsize=7)
  fig.colorbar(img,ax=ax,label='Joint pair F1 mean');ax.set_title('Fixed 21 pairs; n = gold support')
 elif number==5:
  fig,ax=plt.subplots(figsize=(9,4));comps=[('PHOBERT-02','BIGRU-02'),('PHOBERT-02','TFIDF-02'),('BIGRU-02','TFIDF-02')]
  for i,(a,b) in enumerate(comps):
   comp=f'{a} minus {b}'
   for k,sent in enumerate(SENTIMENTS):
    vals=[100*float(d['decomposition'][comp,str(seed),sent]['fixed21_contribution']) for seed in SEEDS];m,sd=mean_sd(vals);ax.errorbar(i+(k-1)*.18,m,yerr=sd,color=COLORS[k],marker=MARKERS[k],capsize=3,label=sent if i==0 else None)
  ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(3),[f'{a} − {b}' for a,b in comps],rotation=12);ax.set_ylabel('Contribution (percentage points)');ax.legend();ax.set_title('Fixed-21 Macro-F1 difference decomposition')
 elif number==6:
  fig,axes=plt.subplots(1,3,figsize=(13,4));ev=d['events']
  for i,c in enumerate(CANDIDATES):
   for k,outcome in enumerate(('CORRECT_ASPECT_CORRECT_SENTIMENT','CORRECT_ASPECT_WRONG_SENTIMENT','MISSED_ASPECT')):
    vals=[sum(r['candidate']==c and r['seed']==str(seed) and r['outcome']==outcome for r in ev)/2969 for seed in SEEDS];m,sd=mean_sd(vals);axes[0].errorbar(k+(i-1)*.16,m,yerr=sd,color=COLORS[i],marker=MARKERS[i],capsize=2)
   vals=[sum(r['candidate']==c and r['seed']==str(seed) and r['outcome']=='SPURIOUS_ASPECT' for r in ev)/(1623*7-2969) for seed in SEEDS];m,sd=mean_sd(vals);axes[1].errorbar(i,m,yerr=sd,color=COLORS[i],marker=MARKERS[i],capsize=3)
   for k,(gold,pred) in enumerate((('neutral','positive'),('negative','positive'),('negative','neutral'))):
    vals=[]
    for seed in SEEDS:
     rr=[r for r in d['confusion'] if r['candidate']==c and r['seed']==str(seed) and r['definition']=='detection_gated' and r['gold_sentiment']==gold and r['predicted_sentiment']==pred]
     den=sum(int(r['denominator']) for r in rr);vals.append(sum(int(r['count']) for r in rr)/den if den else None)
    m,sd=mean_sd(vals)
    if m is not None:axes[2].errorbar(k+(i-1)*.16,m,yerr=sd,color=COLORS[i],marker=MARKERS[i],capsize=2,label=c if k==0 else None)
  axes[0].set_xticks(range(3),['Correct','Wrong\nsentiment','Missed'],rotation=10);axes[0].set_title('Gold-present outcomes');axes[0].set_ylabel('Fraction of 2,969 gold aspects');axes[0].set_ylim(0,1)
  axes[1].set_xticks(range(3),CANDIDATES,rotation=30);axes[1].set_title('Spurious aspect');axes[1].set_ylabel('Fraction of gold-absent opportunities');axes[1].set_ylim(0,1)
  axes[2].set_xticks(range(3),['N→P','Neg→P','Neg→N']);axes[2].set_title('Detection-gated confusion');axes[2].set_ylabel('Fraction of gold sentiment support');axes[2].set_ylim(0,1);axes[2].legend(fontsize=7)
 elif number==7:
  fig,axes=plt.subplots(1,2,figsize=(11,4),gridspec_kw={'width_ratios':[3,1]}); subsets=list(SUBSETS)[:4]
  for i,c in enumerate(CANDIDATES):
   means=[];sds=[]
   for sub in subsets:
    m,sd=mean_sd([float(d['diagnostics'][c,str(seed),sub]['joint_macro_f1']) for seed in SEEDS]);means.append(m);sds.append(sd)
   axes[0].errorbar(range(4),means,yerr=sds,marker=MARKERS[i],color=COLORS[i],capsize=3,label=c)
   p=d['diagnostics'][c,'42','test_placeholder_only'];axes[1].scatter(i,float(p['joint_macro_f1']),marker=MARKERS[i],color=COLORS[i])
  axes[0].set_xticks(range(4),['Original\n1623, 21/21','L1\n1592, 20/21','L2\n1585, 20/21','No placeholder\n1615, 21/21']);axes[0].set_ylim(0,1);axes[0].set_ylabel('Fixed-21 Joint Macro-F1');axes[0].legend(fontsize=7)
  axes[1].set_xticks(range(3),CANDIDATES,rotation=40);axes[1].set_ylim(-.02,1);axes[1].set_title('Placeholder only\nn=8, support 0/21')
 elif number==8:
  fig,ax=plt.subplots(figsize=(10,4))
  for i,c in enumerate(CANDIDATES):
   for k,a in enumerate(ASPECTS):
    m,sd=mean_sd([float(d['pairs'][c,str(seed),a+':negative']['f1']) for seed in SEEDS]);ax.errorbar(k+(i-1)*.16,m,yerr=sd,color=COLORS[i],marker=MARKERS[i],capsize=2,label=c if k==0 else None)
  labels=[f"{a}\n(n={d['pairs'][CANDIDATES[0],'42',a+':negative']['support']})" for a in ASPECTS];ax.set_xticks(range(7),labels);ax.set_ylim(0,1.05);ax.set_ylabel('Negative Joint pair F1');ax.legend();ax.set_title('Seven core aspects; n = gold support')
 else:raise ValueError(number)
 return finish(fig)
