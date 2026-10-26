"""Local train-only VnCoreNLP/PhoBERT tokenizer audit; no model weights or training."""
import argparse
import base64
from collections import Counter
import csv
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import unicodedata

ROOT=Path(__file__).resolve().parents[1]

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def ceiling_metadata(supports,reviews):
    from absa_protocol import JOINT_LABELS
    if set(supports)!=set(JOINT_LABELS):raise ValueError('Need exactly 21 fixed support values')
    if any(type(v) is not int or v<0 for v in supports.values()):raise ValueError('Invalid support')
    observed=sum(v>0 for v in supports.values())
    return {'reviews':reviews,'supported_labels':observed,'fixed_labels':21,
            'zero_support_labels':[k for k,v in supports.items() if v==0],
            'support_only_upper_bound':observed/21 if reviews else None,
            'metric_unchanged':True,'not_a_model_score':True}

def prepare_text(text):
    return re.sub(r'\s+',' ',unicodedata.normalize('NFC',text)).strip()

def percentile(values,q):
    values=sorted(values)
    if not values:return None
    pos=(len(values)-1)*q;lo=int(pos);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(pos-lo)

def length_statistics(lengths):
    return {'reviews':len(lengths),'mean':sum(lengths)/len(lengths) if lengths else None,
            'min':min(lengths) if lengths else None,'max':max(lengths) if lengths else None,
            **{name:percentile(lengths,q) for name,q in [('p50',.5),('p90',.9),('p95',.95),('p99',.99)]},
            'truncation':{str(n):{'reviews':sum(x>n for x in lengths),
                                 'rate':sum(x>n for x in lengths)/len(lengths) if lengths else None} for n in [128,256]}}

def write_once(path,data):
    payload=(json.dumps(data,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and path.read_bytes()!=payload:raise ValueError(f'Refusing to replace existing artifact {path}; version a new run')
    path.write_bytes(payload)

def reporting_guidance():
    from absa_protocol import JOINT_LABELS,CORE_ASPECTS,SENTIMENTS
    with (ROOT/'protocols/data_audit_manifest.csv').open(encoding='utf-8',newline='') as f:manifest=list(csv.DictReader(f))
    summary={}
    for split in ['train','val']:
        rows=[r for r in manifest if r['source_split']==split]
        supports={label:sum(r[a]==s for r in rows) for label in JOINT_LABELS for a,s in [label.split(':')]}
        summary[split]=ceiling_metadata(supports,len(rows))
    census=json.loads((ROOT/'reports/protocol_freeze/audit_summary_v1.json').read_text())
    for subset,r in census['diagnostic_subsets'].items():summary[subset]=ceiling_metadata(r['support_21'],r['reviews'])
    write_once(ROOT/'reports/experiments/macro_f1_ceiling_v1.json',summary)
    lines=['# Hướng dẫn báo cáo: Macro-F1 ceiling theo protocol v1','',
           'Đây là bổ sung cách diễn giải; không đổi công thức, evaluator, zero_division, targets hoặc freeze-lock v1.', '',
           'Với K=21 labels cố định và m labels có gold support>0, mỗi label absent có F1=0 theo convention v1 ',
           '(kể cả khi có false positives). Vì vậy trên subset không rỗng: **Macro-F1 ≤ m/21**.',
           'Đây là support-only upper bound đạt được bởi oracle sao chép gold, không phải dự đoán về mức model thực tế đạt được.', '',
           '| Tập | Reviews | Supported/21 | Upper bound | Zero-support labels |','|---|---:|---:|---:|---|']
    for name,r in summary.items():
        value=r['support_only_upper_bound']
        lines.append(f'| {name} | {r["reviews"]} | {r["supported_labels"]}/21 | {100*value:.4f}% | '+(', '.join(r['zero_support_labels']) or '—')+' |')
    lines += ['', '## Cách diễn giải trong báo cáo thực nghiệm', '',
              '- Validation thiếu PRICE-Negative: ngay cả oracle cũng chỉ có primary joint Macro-F1 20/21 ≈95,2381%. Đó là convention support, không phải lỗi model trên lớp chưa được kiểm tra.',
              '- Non-overlap L1/L2 thiếu PRICE-Neutral: cùng ceiling 20/21. Không so raw Macro-F1 giữa các subsets như chúng có cùng độ khó hoặc composition.',
              '- Placeholder-only test có 0/21 supported core labels: ceiling=0, core recall N/A; vẫn report false positives và OTHERS detection. Điểm 0 ở đây không chứng minh model không hiểu review.',
              '- Conditional summary fixed7×fixed3 có support ceiling tương ứng m/21, vì mỗi aspect nhận trọng số bằng nhau; per-aspect conditional ceiling là số sentiment có support /3.',
              '- Luôn report primary fixed21, observed-support macro với tên riêng, TP/FP/FN/support, Negative P/R và zero-support list. Không chia primary cho ceiling, không thay 21 bằng m trong primary.',
              '- Support ceiling khác với best achievable score do annotation conflicts, model capacity hoặc input information; nó chỉ là giới hạn theo công thức và support.',
              '- Chọn configuration bằng mean validation primary trên 3 seeds đã khóa; ceiling validation cố định cho mọi candidate nên không đổi model ranking hoặc tie-break.',
              '- Nếu thay denominator, bỏ absent labels khỏi primary hoặc đổi zero_division: tạo protocol v2. Thêm bảng ceiling/cách diễn giải giữ nguyên protocol v1.', '',
              'Nguồn kiểm chứng convention: [scikit-learn F1 docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html).',
              'Các giá trị trên tính từ manifest/source đã freeze, không có model predictions.', '']
    report=ROOT/'reports/experiments/reporting_guidance_v1.md'
    payload='\n'.join(lines).encode()
    report.parent.mkdir(parents=True,exist_ok=True)
    if report.exists() and report.read_bytes()!=payload:raise ValueError('Reporting guidance already exists with different content')
    report.write_bytes(payload)
    return summary

def run_audit(assets,gate_python):
    gate_out=ROOT/'reports/modeling_preflight/reproducibility_gate_v1.json'
    subprocess.run([gate_python,str(ROOT/'src/reproducibility_gate.py'),'--output',str(gate_out)],cwd=ROOT,check=True,timeout=240)
    # Modeling Python is separate from the frozen audit runtime; both are recorded explicitly.
    receipt_path=ROOT/'reports/modeling_preflight/assets_v1.json'
    receipt=json.loads(receipt_path.read_text())
    for group in ['phobert','vncorenlp']:
        for item in receipt[group]['files']:
            if digest(assets/item['relative_path'])!=item['sha256']:raise ValueError('Asset hash mismatch')
    java=assets/receipt['java']['java_relative_path']
    if digest(assets/receipt['java']['file']['relative_path'])!=receipt['java']['file']['sha256']:raise ValueError('JDK archive checksum mismatch')
    java_version=subprocess.run([str(java),'-version'],capture_output=True,text=True,check=True).stderr
    config=json.loads((ROOT/'protocols/evaluation_protocol_v1.yaml').read_text())
    train_file=ROOT/config['source_files']['train']['path']
    if digest(train_file)!=config['source_files']['train']['sha256']:raise ValueError('Train fingerprint drift')
    with train_file.open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    prepared=[prepare_text(r['data']) for r in rows]
    smoke=['Ông Nguyễn Khắc Chúc đang làm việc tại Đại học Quốc gia Hà Nội.',
           'Màu đẹp!!! Nhưng son KHÔ quá :( 😊 ko thích.',
           'Chưa dùng nên không biết son có lâu trôi không.', 'NAN', '!!!', '👍🏻']
    inputs=prepared+[prepare_text(x) for x in smoke]
    with tempfile.TemporaryDirectory(prefix='absa-word-segmentation-') as temp:
        temp=Path(temp);inp=temp/'reviews.tsv';out=temp/'segmented.tsv'
        inp.write_text(''.join(str(i)+'\t'+base64.b64encode(s.encode()).decode()+'\n' for i,s in enumerate(inputs)),encoding='utf-8')
        cmd=[str(java),'-Xmx1g','--class-path',str(assets/'vncorenlp/VnCoreNLP-1.2.jar'),
             str(ROOT/'src/VnCoreNLPBatch.java'),str(inp),str(out)]
        subprocess.run(cmd,cwd=assets/'vncorenlp',check=True,timeout=600)
        segmented={}
        for line in out.read_text(encoding='utf-8').splitlines():
            i,text=line.split('\t');index=int(i)
            if index in segmented:raise ValueError('Duplicate segmented record')
            segmented[index]=base64.b64decode(text).decode('utf-8')
    if set(segmented)!=set(range(len(inputs))):raise ValueError('Segmentation row coverage mismatch')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(assets/'phobert',use_fast=False,local_files_only=True)
    specials=tokenizer.num_special_tokens_to_add(pair=False)
    if specials!=2:raise ValueError('Unexpected PhoBERT special token count')
    records=[];lengths=[];unknowns=0;pieces=0
    for i,row in enumerate(rows):
        seg=segmented[i]
        bare=tokenizer.encode(seg,add_special_tokens=False,truncation=False)
        ids=tokenizer.build_inputs_with_special_tokens(bare)
        if len(ids)!=len(bare)+specials or ids[0]!=tokenizer.bos_token_id or ids[-1]!=tokenizer.eos_token_id:
            raise ValueError('Special token framing mismatch')
        for limit in [128,256]:
            cut=tokenizer(seg,add_special_tokens=True,truncation=True,max_length=limit)
            if len(cut['input_ids'])>limit or cut['input_ids'][-1]!=tokenizer.eos_token_id:raise ValueError('Truncation lost EOS')
        lengths.append(len(ids));unknowns+=bare.count(tokenizer.unk_token_id);pieces+=len(bare)
        records.append({'source_split':'train','source_row_index':i,'raw_text_sha256':hashlib.sha256(row['data'].encode()).hexdigest(),
                        'prepared_text':prepared[i],'segmented_text':seg,'ids_with_special_tokens':ids,
                        'subword_length_with_special_tokens':len(ids),'segmented_unit_count':len(seg.split())})
        if (i+1)%2000==0:print('Tokenized train reviews:',i+1,flush=True)
    probe=[]
    for j,text in enumerate(smoke,start=len(rows)):
        ids=tokenizer.encode(segmented[j],add_special_tokens=True,truncation=False)
        probe.append({'raw':text,'segmented':segmented[j],'tokens':tokenizer.convert_ids_to_tokens(ids),'ids':ids})
    if not any('_' in t for t in segmented[len(rows)].split()):raise ValueError('Official segmentation example has no joined words')
    if not all(word in segmented[len(rows)+2].lower() for word in ['chưa','không']):raise ValueError('Segmentation lost negation in smoke example')
    statistics=length_statistics(lengths)
    # This choice uses TRAIN lengths only, never validation/test performance.
    selected_length=128 if statistics['p99']<=128 and statistics['truncation']['128']['rate']<=.01 else 256
    packages={d.metadata['Name']:d.version for d in distributions()}
    summary={'status':'PASS','training_started':False,'source_split_used_for_length_selection':'train',
             'train_sha256':digest(train_file),'train_reviews':len(rows),'model':'vinai/phobert-base',
             'model_revision':receipt['phobert']['revision'],'segmenter':'VnCoreNLP-1.2 / RDRSegmenter wseg-only',
             'segmenter_revision':receipt['vncorenlp']['revision'],'java_version':java_version,
             'gate_runtime':{'python':'3.15.1','unicode':'17.0.0'},
             'modeling_runtime':{'python':platform.python_version(),'unicode':unicodedata.unidata_version,'packages':packages},
             'tokenizer_class':type(tokenizer).__name__,'is_fast':tokenizer.is_fast,
             'num_added_special_tokens':specials,'special_tokens_map':tokenizer.special_tokens_map,
             'special_token_ids':{x:getattr(tokenizer,x+'_token_id') for x in ['bos','eos','pad','unk','mask']},
             'subword_lengths_including_specials':statistics,'segmented_unit_lengths':length_statistics([r['segmented_unit_count'] for r in records]),
             'unknown_subword_count':unknowns,'total_bare_subwords':pieces,'unknown_subword_rate':unknowns/pieces if pieces else None,
             'empty_segmented_train_reviews':sum(not r['segmented_text'].strip() for r in records),
             'max_length_selection_rule':'128 if train P99<=128 and train truncation@128<=1%; otherwise 256',
             'registered_max_length':selected_length,'smoke_examples':probe,
             'assets_receipt_sha256':digest(receipt_path),
             'scope':'Local tokenizer-only; all train reviews retained including NAN; no model weights, no training, no validation/test token-based selection.'}
    destination=ROOT/'reports/modeling_preflight'
    write_once(destination/'phobert_tokenizer_audit_v1.json',summary)
    cache=destination/'train_tokenization_v1.jsonl'
    data=''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in records).encode()
    if cache.exists() and cache.read_bytes()!=data:raise ValueError('Token cache changed')
    cache.write_bytes(data)
    req=destination/'tokenizer_environment_v1.txt'
    req.write_text('\n'.join(f'{k}=={v}' for k,v in sorted(packages.items(),key=lambda x:x[0].lower()))+'\n',encoding='utf-8')
    write_once(destination/'tokenizer_outputs_v1.json',{'cache_sha256':digest(cache),'summary_sha256':digest(destination/'phobert_tokenizer_audit_v1.json'),
                                                     'requirements_sha256':digest(req),'java_bridge_sha256':digest(ROOT/'src/VnCoreNLPBatch.java')})
    print('TOKENIZER AUDIT PASS',json.dumps(statistics),'selected max_length',selected_length,flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path);p.add_argument('--gate-python')
    p.add_argument('--reporting-only',action='store_true')
    args=p.parse_args();reporting_guidance()
    if not args.reporting_only:
        if not args.assets or not args.gate_python:p.error('--assets and --gate-python required')
        run_audit(args.assets.resolve(),args.gate_python)
