"""Create/verify a separate experiment registration without changing frozen protocol v1."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
REG='protocols/experiments/experiment_registration_v1.json'
LOCK='protocols/experiments/experiment_registration_lock_v1.json'
BASE_LOCK='16e9b76b6a71530d08704cadc2d1ddeddf54ca57615c35f8e650ccc57ac86ee8'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def payload(value):return (json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()

def registration():
    if sha(ROOT/'protocols/freeze_lock_v1.json')!=BASE_LOCK:raise ValueError('Original v1 lock changed')
    protocol=json.loads((ROOT/'protocols/evaluation_protocol_v1.yaml').read_text())
    audit=json.loads((ROOT/'reports/modeling_preflight/phobert_tokenizer_audit_v1.json').read_text())
    assets=json.loads((ROOT/'reports/modeling_preflight/assets_v1.json').read_text())
    if audit['status']!='PASS' or audit['train_reviews']!=12981:raise ValueError('Train tokenizer audit is incomplete')
    max_length=audit['registered_max_length']
    text={'common':'src/modeling_preflight.py::prepare_text: NFC + collapse Unicode whitespace; retain case, accents, punctuation, emoji and negation',
          'audit_keys_as_features':False,'row_ids_as_features':False,'original_NAN_rows_retained':True,
          'fit_statistics_split':'official train only','no_global_cleaning':True}
    lexical={'preprocessing':'common NFC/whitespace then vectorizer lowercase=True; no word segmentation',
             'word_vectorizer':{'ngram_range':[1,2],'token_pattern':r'(?u)\b\w+\b','min_df':2,'max_df':1.0,
                                'max_features':20000,'sublinear_tf':True,'norm':'l2','dtype':'float32','stop_words':None},
             'char_vectorizer':{'analyzer':'char','ngram_range':[3,5],'min_df':2,'max_df':1.0,'max_features':30000,
                                'sublinear_tf':True,'norm':'l2','dtype':'float32'},
             'feature_union_weights':[1.0,1.0],
             'estimator':{'class':'sklearn.linear_model.LogisticRegression','C':1.0,'solver':'liblinear',
                          'max_iter':1000,'tol':0.0001,'class_weight':None,'fit_intercept':True,
                          'multiclass':'OneVsRestClassifier; conditional probabilities normalized over 3 classes'},
             'heads':'8 independent binary aspect classifiers; 7 gold-present sentiment OVR classifiers; shared train-fitted features',
             'wallclock_limit_minutes_per_seed':30,'threads':1,
             'candidates':[{'id':'TFIDF-01','features':'word_1_2'},
                           {'id':'TFIDF-02','features':'word_1_2_plus_char_3_5'}]}
    neural_loss={'aspect':'BCEWithLogitsLoss mean over all review x 8 aspect cells',
                 'sentiment':'CrossEntropy over gold-present core aspect cells only; sum / number of such cells in batch; zero if none',
                 'total':'aspect_loss + sentiment_loss','class_weights':None,'focal_loss':False}
    segmented={'engine':'VnCoreNLP 1.2 wseg-only (RDRSegmenter)',
               'revision':assets['vncorenlp']['revision'],'java_release':assets['java']['release'],
               'bridge':'src/VnCoreNLPBatch.java; join segmented sentences by one space; each review independent'}
    recurrent={'preprocessing':{'common':True,'segmentation':segmented,'tokens':'segmented_text.split(); no lowercasing',
                                'vocabulary':'train token frequency descending, Unicode lexical ascending ties; PAD=0, UNK=1; max 30000 including specials; min_freq=1',
                                'max_length':128,'padding':'right','truncation':'right','add_special_tokens':False},
               'architecture':{'embedding_dim':128,'embedding_initialization':'random; trainable; padding_idx=0',
                               'bidirectional_GRU_hidden_per_direction':128,'GRU_layers':1,
                               'pooling':'masked max + masked mean over bidirectional GRU states',
                               'conv_variant':'parallel Conv1D on embeddings, 128 channels, kernel=3, stride=1, padding=1, ReLU, masked global max; concatenate with GRU pooling',
                               'dropout':0.3,'heads':'linear 8 aspect logits + 7 x 3 sentiment logits'},
               'loss':neural_loss,'optimizer':{'name':'Adam','lr':0.001,'weight_decay':0.0},
               'batch_size':32,'gradient_accumulation':1,'max_epochs':20,'patience':3,
               'gradient_clip_norm':1.0,'scheduler':None,'wallclock_limit_minutes_per_seed':60,
               'candidates':[{'id':'BIGRU-01','encoder':'BiGRU'}, {'id':'BIGRU-02','encoder':'BiGRU+parallelConv1D'}]}
    contextual={'preprocessing':{'common':True,'segmentation':segmented,'tokenizer':'PhobertTokenizer slow; transformers 4.57.1; use_fast=False',
                                 'max_length':max_length,'padding':'dynamic right padding to batch maximum capped at max_length',
                                 'truncation':'right; add BOS and EOS; preserve EOS when truncating','add_special_tokens':True,
                                 'length_choice_rule':audit['max_length_selection_rule']},
                'initialization':{'repo':'vinai/phobert-base','revision':assets['phobert']['revision'],
                                  'model_class':'transformers.RobertaModel','pooler':'not used',
                                  'representation':'last_hidden_state[:,0,:] (BOS/CLS)','fine_tune_all_encoder_layers':True,
                                  'weight_download':'only at training launch, never main/latest; pretrained weights have not been downloaded in preflight'},
                'heads':{'dropout':0.1,'output':'linear 8 aspect logits + 7 x 3 sentiment logits'},
                'loss':neural_loss,'optimizer':{'name':'AdamW','weight_decay':0.01,'no_decay':['bias','LayerNorm.weight'],
                                               'eps':1e-8,'betas':[0.9,0.999]},
                'scheduler':'linear warmup for first 10% optimizer steps, then linear decay',
                'micro_batch_size':16,'gradient_accumulation':2,'effective_batch_size':32,
                'max_epochs':5,'patience':2,'gradient_clip_norm':1.0,'wallclock_limit_minutes_per_seed':120,
                'candidates':[{'id':'PHOBERT-01','lr':1e-5},{'id':'PHOBERT-02','lr':2e-5}]}
    return {'registration_id':'absa_model_comparison_registration_v1','registered_before_fitting':True,
            'data_evaluation_protocol':'absa_data_evaluation_v1','original_freeze_lock_sha256':BASE_LOCK,
            'protocol_sha256':sha(ROOT/'protocols/evaluation_protocol_v1.yaml'),
            'source_train_sha256':audit['train_sha256'],'tokenizer_audit_sha256':sha(ROOT/'reports/modeling_preflight/phobert_tokenizer_audit_v1.json'),
            'code_base_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'code_state':'uncommitted additions; registration lock hashes this preflight code; future trainer code needs a run manifest before fit',
            'target_order':protocol['targets'],'text_policy':text,'seeds':protocol['selection']['seeds'],
            'families':{'tfidf_logistic':lexical,'bigru_conv1d':recurrent,'phobert':contextual},
            'budget':{'candidates_per_family':2,'seed_runs_per_candidate':3,'maximum_main_training_runs':18,
                      'no_automatic_extra_search':True,'loss_ablations_in_main_registration':False,
                      'failed_or_timed_out_runs':'report as failed/censored; do not silently raise budget, pick best seed or add candidates'},
            'selection':{'primary':'mean official val joint.macro_f1 across all 3 registered seeds',
                         'candidate_tie_break':'higher mean val joint micro-F1, then lexicographically lower preregistered candidate id',
                         'aspect_threshold_grid':[0.3,0.4,0.5,0.6,0.7],
                         'threshold_application':'one global threshold over all 8 aspect probabilities per trained checkpoint; sigmoid for neural; binary probability for LR',
                         'threshold_tie_break':'higher val joint micro-F1; then closest to 0.5; then smaller threshold',
                         'sentiment_decoding':'argmax over positive, neutral, negative; ties follow this fixed order',
                         'checkpoint_selection':'after each epoch on val using selected global threshold; best primary, then micro, then earliest epoch',
                         'early_stopping':'patience on no primary improvement; min_delta=0.0; never test-based',
                         'missing_validation_class':'PRICE-negative not individually tunable; no class-specific thresholds or test compensation',
                         'grouped_train_cv':'disabled under frozen v1',
                         'test':'one final evaluation round per preregistered selected family/checkpoints/all seeds, only after full run manifests; original + frozen diagnostic subsets'},
            'runtime':{'gate':{'python':'3.15.1','unicode':'17.0.0'},
                       'modeling':{'python':audit['modeling_runtime']['python'],'unicode':audit['modeling_runtime']['unicode'],
                                   'resolved_environment':'protocols/experiments/modeling_requirements_v1.lock',
                                   'installed_tokenizer_environment':'reports/modeling_preflight/tokenizer_environment_v1.txt',
                                   'training_packages_status':'torch/sklearn dependency graph resolved and hash-locked; full GPU runtime not installed or executed in this preflight'},
                       'neural_hardware_required':'single CUDA GPU with >=16 GiB; record GPU/driver/CUDA and actual package versions at launch; CPU is only for diagnostics, not silent replacement of the registered GPU budget',
                       'precision':'float32; AMP off; TF32 off',
                       'determinism':{'torch_deterministic_algorithms':True,'cudnn_benchmark':False,'cudnn_deterministic':True,
                                      'CUBLAS_WORKSPACE_CONFIG':':4096:8','torch_threads':1,'BLAS_threads':1,
                                      'dataloader_workers':0,'shuffle':'train only, generator seeded per registered seed',
                                      'seed_all':['PYTHONHASHSEED before process start','Python random','NumPy','torch CPU','torch CUDA']},
                       'exact_GPU_reproducibility':'must be measured on target hardware; package hashes alone do not guarantee bitwise CUDA determinism'},
            'metrics':'unchanged frozen evaluator v1; support ceilings are interpretation only, not a normalized new metric',
            'sensitivity_tracks':'evaluation subsets already frozen; training-without-NAN and weighting/focal require a separate preregistered extension before fitting',
            'training_launch_gates':['original exact-runtime gate PASS','registered asset/package/source hashes match',
                                     'new trainer implementation and exact code/dirty hash recorded',
                                     'target hardware and training dependencies verified','save run manifest before fit'],
            'model_training_started':False}

def verify():
    lock=json.loads((ROOT/LOCK).read_text())
    if sha(ROOT/'protocols/freeze_lock_v1.json')!=BASE_LOCK:raise ValueError('Original v1 lock changed')
    for relative,expected in lock['artifacts'].items():
        if sha(ROOT/relative)!=expected:raise ValueError('Registered artifact changed: '+relative)
    reg=json.loads((ROOT/REG).read_text())
    if reg['target_order']!=json.loads((ROOT/'protocols/evaluation_protocol_v1.yaml').read_text())['targets']:
        raise ValueError('Target semantics changed')
    candidates=[c['id'] for family in reg['families'].values() for c in family['candidates']]
    if len(candidates)!=6 or len(set(candidates))!=6 or len(reg['seeds'])!=3:raise ValueError('Candidate/seed registration invalid')
    print('PASS registration integrity: 3 families x 2 candidates x 3 seeds = 18 planned fits; no training executed')

def create():
    target=ROOT/REG;target.parent.mkdir(parents=True,exist_ok=True)
    content=payload(registration())
    if target.exists() and target.read_bytes()!=content:raise ValueError('Existing registration differs; create a new version before fitting')
    target.write_bytes(content)
    artifacts=[REG,'src/register_experiments.py','src/modeling_preflight.py','src/reproducibility_gate.py',
               'src/restore_model_assets.py','src/VnCoreNLPBatch.java','tests/test_modeling_preflight.py',
               'protocols/experiments/modeling_requirements_v1.in','protocols/experiments/modeling_requirements_v1.lock',
               'reports/modeling_preflight/assets_v1.json','reports/modeling_preflight/phobert_tokenizer_audit_v1.json',
               'reports/modeling_preflight/train_tokenization_v1.jsonl','reports/modeling_preflight/tokenizer_outputs_v1.json',
               'reports/modeling_preflight/tokenizer_environment_v1.txt','reports/experiments/reporting_guidance_v1.md',
               'reports/experiments/macro_f1_ceiling_v1.json']
    lock=payload({'registration_id':'absa_model_comparison_registration_v1','original_freeze_lock_sha256':BASE_LOCK,
                  'artifacts':{f:sha(ROOT/f) for f in artifacts},'training_started':False})
    path=ROOT/LOCK
    if path.exists() and path.read_bytes()!=lock:raise ValueError('Registration lock already exists with different content')
    path.write_bytes(lock);verify()

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    verify() if args.verify else create()
