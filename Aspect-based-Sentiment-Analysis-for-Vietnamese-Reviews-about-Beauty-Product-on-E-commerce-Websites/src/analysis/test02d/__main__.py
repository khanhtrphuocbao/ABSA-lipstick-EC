"""Generate auditable TEST-02D figures or verify existing figures are current."""
import argparse, hashlib, io, json, platform, sys
from pathlib import Path
from src.analysis.test02d.sources import ROOT,OUT,CANDIDATES,SEEDS,ASPECTS,SENTIMENTS,LABELS,sha,load
from src.analysis.test02b.__main__ import write_once

FIGS=[
('fig_01_model_comparison.png','Overall model comparison','Fixed-21 Joint Macro-F1 and Joint Micro-F1','Original test; mean ± sample SD across three seeds. Fixed-21 macro averages 21 core aspect–sentiment F1 values. TF-IDF zero seed SD does not quantify data sampling uncertainty. Source: model_comparison.csv.'),
('fig_02_validation_vs_test.png','Validation and test','Fixed-21 Joint Macro-F1','Validation and original test; mean ± sample SD across three seeds. Validation aggregate and six neural manifest links rely on assumed equivalence; split differences do not establish causation. Source: final_selection_freeze.json and validation aggregate, model_comparison.csv.'),
('fig_03_sentiment_performance.png','Pooled sentiment performance','Pooled Joint sentiment Precision, Recall and F1','Original test; mean ± sample SD across three seeds. Counts are pooled across seven core aspects for each sentiment; pooled F1 is not the fixed-21 contribution. Source: sentiment_comparison.csv.'),
('fig_04_aspect_sentiment_heatmap.png','Aspect–sentiment F1','Mean Joint pair F1 with gold support','Original test; arithmetic mean across three seeds, for all fixed 21 aspect–sentiment pairs. n is gold support; small support makes estimates unstable. Source: per_pair_metrics.csv.'),
('fig_05_macro_f1_decomposition.png','Fixed-21 contribution decomposition','Contribution to Joint Macro-F1 difference','Original test; mean ± sample SD across three paired seeds. Contribution is sum of seven pair F1 differences divided by 21, in percentage points; not pooled sentiment F1. Source: macro_f1_decomposition.csv.'),
('fig_06_error_analysis.png','Review–aspect error analysis','Gold-present outcomes, spurious aspects and detection-gated confusion','Original test; mean ± sample SD across three seeds. Gold outcomes use 2,969 gold-present aspects; spurious uses gold-absent aspect opportunities; confusion uses gold sentiment support. Shared test reviews are not independent seed datasets. Source: error_taxonomy.csv and confusion_summary.csv.'),
('fig_07_diagnostic_sensitivity.png','Diagnostic subset sensitivity','Fixed-21 Joint Macro-F1','Frozen test subsets; mean ± sample SD across three seeds. n and supported labels shown; placeholder-only has zero core gold support. Scores are not normalized by support ceiling; overlap screening is not proof of no leakage. Source: diagnostic_subset_comparison.csv.'),
('fig_08_negative_by_aspect.png','Negative sentiment by aspect','Joint Negative pair F1 with support','Original test; mean ± sample SD across three seeds; n is gold support. Rare labels have unstable estimates and three seeds share one test dataset. Source: per_pair_metrics.csv.')]
MAP={1:['model','summary'],2:['model','freeze'],3:['sentiment','summary'],4:['pairs','summary'],5:['decomposition','model','summary'],6:['events','confusion','summary'],7:['diagnostics','summary'],8:['pairs','summary']}
def generator_sha():
 h=hashlib.sha256()
 for p in sorted(Path(__file__).parent.glob('*.py')): h.update(p.name.encode()+b'\0'+p.read_bytes())
 return h.hexdigest()
def payload(obj):return (json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
def stale(manifest):
 reasons=[]; current=generator_sha()
 for f in manifest['figures']:
  if f['generator_sha256']!=current:reasons.append(f['figure_id']+': generator changed')
  for s in f['sources']:
   p=Path(s['path'])
   if not p.is_file() or sha(p)!=s['sha256']:reasons.append(f['figure_id']+': source changed '+str(p))
  for o in f['outputs']:
   p=Path(o['path'])
   if not p.is_file() or sha(p)!=o['sha256']:reasons.append(f['figure_id']+': output changed '+str(p))
 return reasons

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--validation-aggregate',type=Path,default=ROOT.parent/'absa-run/validation_aggregate_assumed_equivalence.json');parser.add_argument('--check-stale',action='store_true');args=parser.parse_args()
 manifest_path=OUT/'figure_manifest.json'
 if args.check_stale:
  m=json.loads(manifest_path.read_text());errors=stale(m)
  print(json.dumps({'status':'STALE' if errors else 'CURRENT','reasons':errors},indent=2));return 1 if errors else 0
 data=load(args.validation_aggregate)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from src.analysis.test02d.figures import make_figure
 env={'python':platform.python_version(),'matplotlib':matplotlib.__version__,'backend':matplotlib.get_backend(),'platform':platform.platform()}
 gen=generator_sha();alloutputs={};figmeta=[]
 for number,(filename,title,metric,caption) in enumerate(FIGS,1):
  sources=[{'path':str(data['files'][key]),'sha256':sha(data['files'][key])} for key in MAP[number]]
  if number==2 and data['validation']['status']=='PASS':sources.append({'path':str(args.validation_aggregate.resolve()),'sha256':sha(args.validation_aggregate)})
  status='BLOCKED' if number==2 and data['validation']['status']!='PASS' else 'READY'
  outputs=[]
  if status=='READY':
   figure=make_figure(number,data); buf=io.BytesIO();figure.savefig(buf,format='png',metadata={'Software':'matplotlib '+matplotlib.__version__});plt.close(figure)
   path=OUT/'figures'/filename;raw=buf.getvalue();alloutputs[path]=raw;outputs=[{'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}]
  figmeta.append({'figure_id':f'FIG-{number:02d}','title':title,'sources':sources,'filters':{'split':'test' if number!=2 else 'validation and test','subset':'five registered subsets' if number==7 else 'original_test'},'candidate_order':list(CANDIDATES),'seed_order':list(SEEDS),'aspect_order':list(ASPECTS),'sentiment_order':list(SENTIMENTS),'label_order':list(LABELS),'metric_definition':metric,'aggregation':'arithmetic mean and sample SD (ddof=1) across three seeds; no pooled counts for mean F1','units':'percentage points' if number==5 else 'F1 or rate','outputs':outputs,'generator_sha256':gen,'environment':env,'caption':caption,'limitations':['ASSUMED_MANIFEST_EQUIVALENCE: six neural historical links not verified','Three training seeds share one official test set']+(['Rare-label support limits reliability'] if number in (4,8) else []),'status':status,'blocked_reason':data['validation']['reason'] if status=='BLOCKED' else None})
 manifest={'version':'test02d_figures_v1','figures':figmeta,'provenance':'ASSUMED_MANIFEST_EQUIVALENCE; historical training source mismatch accepted by user, not verified'}
 inventory={'status':'PASS' if data['validation']['status']=='PASS' else 'PARTIAL','source_sha256':data['hashes'],'validation':data['validation'],'candidate_order':list(CANDIDATES),'seed_order':list(SEEDS),'fixed21_labels':list(LABELS),'environment':env,'generator_sha256':gen,'historical_provenance':'ASSUMED_MANIFEST_EQUIVALENCE; not verified'}
 alloutputs[manifest_path]=payload(manifest);alloutputs[OUT/'test_02d_evidence_inventory.json']=payload(inventory)
 report=['# TEST-02D.0 / 02D.1 figure verification','',f"Reporting runtime: Python {env['python']}; Matplotlib {env['matplotlib']}; backend {env['backend']}; {env['platform']}.",'','Nine runs, fixed 21 labels, pooled sentiment, five subsets and decomposition reconciled with TEST-01/02B/02C. Historical neural manifest and source equivalence are assumed, not verified.','']
 for f in figmeta: report+=['## '+f['figure_id']+' — '+f['title'],'',f['status']+(': '+f['blocked_reason'] if f['blocked_reason'] else ''),'',f['caption'],'']
 alloutputs[OUT/'test_02d_figures_validation.md']=('\n'.join(report)+'\n').encode()
 for path,raw in alloutputs.items():
  if path.exists() and path.read_bytes()!=raw:raise FileExistsError('existing output differs; preserve and resolve explicitly: '+str(path))
 for path,raw in alloutputs.items():write_once(path,raw)
 for name,digest in data['hashes'].items():
  if sha(name)!=digest:raise ValueError('source changed during generation: '+name)
 print(json.dumps({'status':'PASS' if all(f['status']=='READY' for f in figmeta) else 'PARTIAL','figures':{f['figure_id']:f['status'] for f in figmeta},'environment':env},indent=2))
 return 0 if all(f['status']=='READY' for f in figmeta) else 2
if __name__=='__main__':raise SystemExit(main())
