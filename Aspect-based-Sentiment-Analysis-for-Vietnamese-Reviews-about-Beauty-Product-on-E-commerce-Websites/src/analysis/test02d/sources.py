"""Read and reconcile sealed TEST-01/02B/02C evidence without fitting models."""
import csv, hashlib, json, math, statistics
from pathlib import Path
from collections import Counter
from src.analysis.test02b.core import CANDIDATES, SEEDS, validate_seed_matrix

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'reports/experiments/test_02'
ASPECTS=('stayingpower','texture','smell','price','colour','shipping','packing')
SENTIMENTS=('positive','neutral','negative')
LABELS=tuple(f'{a}:{s}' for a in ASPECTS for s in SENTIMENTS)
SUBSETS={'original_test':(1623,21),'test_nonoverlap_l1':(1592,20),'test_nonoverlap_l2':(1585,20),'test_without_placeholder':(1615,21),'test_placeholder_only':(8,0)}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def near(a,b): return math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12)
def mean_sd(v):
 if len(v)!=3 or any(x is None for x in v): return (None,None)
 return statistics.mean(v),statistics.stdev(v)
def read_csv(path, required):
 with Path(path).open(encoding='utf8',newline='') as f:
  reader=csv.DictReader(f)
  if not reader.fieldnames or not set(required)<=set(reader.fieldnames): raise ValueError('schema mismatch: '+str(path))
  return list(reader)
def matrix(rows, dimensions):
 keys=[tuple(r[k] for k in dimensions) for r in rows]
 if len(keys)!=len(set(keys)): raise ValueError('duplicate key: '+str(dimensions))
 return {tuple(r[k] for k in dimensions):r for r in rows}
def check_grid(rows, extra):
 expected={(c,str(s),*v) for c in CANDIDATES for s in SEEDS for v in extra}
 actual={(r['candidate'],r['seed'],*(r[k] for k in extra.fields)) for r in rows}
 if len(rows)!=len(expected) or actual!=expected: raise ValueError('missing/extra/duplicate seed records')

def load(validation_path):
 files={'summary':ROOT/'absa-runs-final-test/final_test_summary.json','freeze':ROOT/'reports/experiments/test_01/final_selection_freeze.json','lock':ROOT/'protocols/freeze_lock_v1.json','protocol':ROOT/'protocols/evaluation_protocol_v1.yaml','model':OUT/'model_comparison.csv','sentiment':OUT/'sentiment_comparison.csv','pairs':OUT/'per_pair_metrics.csv','decomposition':OUT/'macro_f1_decomposition.csv','events':OUT/'error_taxonomy.csv','confusion':OUT/'confusion_summary.csv','diagnostics':OUT/'diagnostic_subset_comparison.csv','b_integrity':OUT/'test_02b_integrity.json','c_completion':OUT/'test_02c_completion.json','c_integrity':OUT/'test_02c_integrity.json'}
 hashes={str(p):sha(p) for p in files.values()}
 freeze=json.loads(files['freeze'].read_text()); summary=json.loads(files['summary'].read_text()); lock=json.loads(files['lock'].read_text())
 if freeze['status']!='SEALED_SELECTION' or summary['freeze_sha256']!=sha(files['freeze']):raise ValueError('freeze integrity mismatch')
 for path,h in lock['artifacts'].items():
  if sha(ROOT/path)!=h:raise ValueError('frozen artifact drift: '+path)
 if set(summary['families'])!=set(CANDIDATES):raise ValueError('candidate coverage')
 selected=[(r['candidate_id'],r['seed']) for r in freeze['runs']];validate_seed_matrix(selected)
 b=json.loads(files['b_integrity'].read_text()); completion=json.loads(files['c_completion'].read_text()); ci=json.loads(files['c_integrity'].read_text())
 if b['status']!='PASS' or completion['status']!='TEST-02C COMPLETE':raise ValueError('missing verification receipt')
 for name,h in completion['artifacts_sha256'].items():
  if sha(OUT/name)!=h:raise ValueError('TEST-02C artifact drift '+name)
 for name,h in ci['input_sha256'].items():
  p=ROOT/name
  if sha(p)!=h:raise ValueError('upstream drift '+name)
  hashes[str(p)]=h
 model=read_csv(files['model'],['candidate','seed','joint_macro_f1','joint_micro_f1']); validate_seed_matrix([(r['candidate'],int(r['seed'])) for r in model]); mm=matrix(model,['candidate','seed'])
 sent=read_csv(files['sentiment'],['candidate','seed','sentiment','precision','recall','f1','tp','fp','fn','support']); sm=matrix(sent,['candidate','seed','sentiment'])
 pairs=read_csv(files['pairs'],['candidate','seed','label','support','tp','fp','fn','f1']); pm=matrix(pairs,['candidate','seed','label'])
 decomp=read_csv(files['decomposition'],['comparison','seed','sentiment','fixed21_contribution']); dm=matrix(decomp,['comparison','seed','sentiment'])
 events=read_csv(files['events'],['candidate','seed','source_row_index','aspect','event_type','outcome']); diag=read_csv(files['diagnostics'],['candidate','seed','subset','reviews','joint_macro_f1','label_support','others_tp','others_fp','others_fn','core_fp']); dg=matrix(diag,['candidate','seed','subset'])
 confusion=read_csv(files['confusion'],['candidate','seed','aspect','definition','gold_sentiment','predicted_sentiment','count','denominator','rate'])
 comparisons=[('PHOBERT-02','BIGRU-02'),('PHOBERT-02','TFIDF-02'),('BIGRU-02','TFIDF-02')]
 for c in CANDIDATES:
  records=summary['families'][c]['per_seed'];validate=Counter(r['seed'] for r in records)
  if validate!=Counter(SEEDS):raise ValueError('summary seed mismatch '+c)
  aggregate=summary['families'][c]['aggregate']['original_test']['joint']
  for column,record in [('joint_macro_f1',aggregate['macro_f1']),('joint_micro_f1',aggregate['micro']['f1'])]:
   calculated=mean_sd([float(mm[c,str(seed)][column]) for seed in SEEDS])
   if not near(calculated[0],record['mean']) or not near(calculated[1],record['sample_sd']):raise ValueError('TEST-01 aggregate mismatch '+c+' '+column)
  for seed in SEEDS:
   row=mm[c,str(seed)]; report=next(r['reports'] for r in records if r['seed']==seed)['original_test']
   if not near(row['joint_macro_f1'],report['joint']['macro_f1']) or not near(row['joint_micro_f1'],report['joint']['micro']['f1']):raise ValueError('model summary mismatch')
   for s in SENTIMENTS:
    v=sm[c,str(seed),s]; ref=report['joint_pooled_per_sentiment'][s]
    for key in ('tp','fp','fn','support'): 
     if int(v[key])!=ref[key]:raise ValueError('sentiment counts mismatch')
    for key in ('precision','recall','f1'):
     if (v[key]=='' and ref[key] is not None) or (v[key]!='' and (ref[key] is None or not near(v[key],ref[key]))):raise ValueError('sentiment metric mismatch')
   for label in LABELS:
    v=pm[c,str(seed),label];ref=report['joint']['per_label'][label]
    if any(int(v[k])!=ref[k] for k in ('tp','fp','fn','support')) or not near(v['f1'],ref['f1']):raise ValueError('pair mismatch')
   if sum(1 for r in events if r['candidate']==c and r['seed']==str(seed) and r['event_type']=='GOLD_PRESENT')!=2969:raise ValueError('event count mismatch')
   for subset,(n,support_count) in SUBSETS.items():
    v=dg[c,str(seed),subset]; sr=json.loads(v['label_support']);ref=next(r['reports'][subset] for r in records if r['seed']==seed)
    if int(v['reviews'])!=n or set(sr)!=set(LABELS) or sum(x>0 for x in sr.values())!=support_count or not near(v['joint_macro_f1'],ref['joint']['macro_f1']):raise ValueError('subset mismatch')
    if any(sr[k]!=ref['joint']['per_label'][k]['support'] for k in LABELS):raise ValueError('subset support mismatch')
   for a,bb in comparisons:
    contributions=[]
    for s in SENTIMENTS:
     item=dm[f'{a} minus {bb}',str(seed),s]
     differences=[float(pm[a,str(seed),aspect+':'+s]['f1'])-float(pm[bb,str(seed),aspect+':'+s]['f1']) for aspect in ASPECTS]
     if not near(item['mean_pair_f1_difference'],statistics.mean(differences)) or not near(item['fixed21_contribution'],sum(differences)/21):raise ValueError('sentiment decomposition group mismatch')
     contributions.append(float(item['fixed21_contribution']))
    v=sum(contributions)
    if not near(v,float(mm[a,str(seed)]['joint_macro_f1'])-float(mm[bb,str(seed)]['joint_macro_f1'])):raise ValueError('decomposition mismatch')
 if len(sm)!=27 or len(pm)!=189 or len(dm)!=27 or len(dg)!=45:raise ValueError('unexpected metric matrix')
 val={'status':'PASS','path':str(validation_path),'reason':None,'values':{}}
 try:
  if sha(validation_path)!=freeze['aggregate_sha256']:raise ValueError('validation aggregate SHA-256 does not match freeze')
  va=json.loads(Path(validation_path).read_text());hashes[str(Path(validation_path))]=sha(validation_path)
  for c in CANDIDATES:
   group=next(v for v in va['families'].values() if v['selected_candidate_id']==c)
   target=group['candidate_validation_summary'][c]
   vals=[next(r['validation_macro_f1'] for r in freeze['runs'] if r['candidate_id']==c and r['seed']==s) for s in SEEDS]
   if target['seeds']!=list(SEEDS) or not near(statistics.mean(vals),target['mean_val_joint_macro_f1']):raise ValueError('validation aggregate values do not match freeze')
   val['values'][c]=vals
 except (OSError,ValueError,KeyError,StopIteration,TypeError) as e:val={'status':'BLOCKED','path':str(validation_path),'reason':str(e),'values':{}}
 return dict(files=files,hashes=hashes,summary=summary,freeze=freeze,model=mm,sentiment=sm,pairs=pm,decomposition=dm,events=events,confusion=confusion,diagnostics=dg,validation=val)
