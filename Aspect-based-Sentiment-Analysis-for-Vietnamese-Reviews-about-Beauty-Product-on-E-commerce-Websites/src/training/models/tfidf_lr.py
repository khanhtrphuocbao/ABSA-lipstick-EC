"""Train-only TF-IDF and registered LogisticRegression heads."""
import json
from pathlib import Path

try:
    from ...absa_protocol import ASPECTS, CORE_ASPECTS, SENTIMENTS
    from ..preprocessing import prepare_texts
except ImportError:
    from absa_protocol import ASPECTS, CORE_ASPECTS, SENTIMENTS
    from training.preprocessing import prepare_texts


class TfidfLogisticABSA:
    def __init__(self, candidate_id, root=None):
        if candidate_id not in ('TFIDF-01', 'TFIDF-02'):
            raise ValueError('Unknown registered TF-IDF candidate')
        self.candidate_id = candidate_id
        self.root = Path(root) if root else Path(__file__).resolve().parents[3]
        registration = json.loads((self.root / 'protocols/experiments/experiment_registration_v1.json').read_text())
        self.config = registration['families']['tfidf_logistic']
        if candidate_id not in {c['id'] for c in self.config['candidates']}:
            raise ValueError('Candidate is not present in experiment registration')
        self.fitted = False

    def _make_vectorizer(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        word = self.config['word_vectorizer']
        common = {'min_df': word['min_df'], 'max_df': word['max_df'],
                  'max_features': word['max_features'], 'ngram_range': tuple(word['ngram_range']),
                  'sublinear_tf': word['sublinear_tf'], 'norm': word['norm'],
                  'dtype': __import__('numpy').float32, 'lowercase': True}
        word_vectorizer = TfidfVectorizer(analyzer='word', token_pattern=word['token_pattern'],
                                         stop_words=word['stop_words'], **common)
        if self.candidate_id == 'TFIDF-01':
            return word_vectorizer
        from sklearn.pipeline import FeatureUnion
        char = self.config['char_vectorizer']
        char_vectorizer = TfidfVectorizer(analyzer=char['analyzer'], min_df=char['min_df'],
                                          max_df=char['max_df'], max_features=char['max_features'],
                                          ngram_range=tuple(char['ngram_range']),
                                          sublinear_tf=char['sublinear_tf'], norm=char['norm'],
                                          dtype=__import__('numpy').float32, lowercase=True)
        return FeatureUnion([('word', word_vectorizer), ('char', char_vectorizer)],
                            transformer_weights={'word': self.config['feature_union_weights'][0],
                                                  'char': self.config['feature_union_weights'][1]})

    def _make_logistic(self):
        from sklearn.linear_model import LogisticRegression
        config = self.config['estimator']
        return LogisticRegression(C=config['C'], solver=config['solver'], max_iter=config['max_iter'],
                                  tol=config['tol'], class_weight=config['class_weight'],
                                  fit_intercept=config['fit_intercept'], n_jobs=1, random_state=None)

    def fit(self, rows):
        import numpy as np
        from sklearn.multiclass import OneVsRestClassifier
        if not rows:
            raise ValueError('Cannot fit TF-IDF on an empty train split')
        if any(row.get('source_split') != 'train' for row in rows):
            raise ValueError('TF-IDF vocabulary and heads may be fit on train rows only')
        texts = prepare_texts([row['data'] for row in rows])
        self.vectorizer = self._make_vectorizer()
        matrix = self.vectorizer.fit_transform(texts)
        if matrix.shape[0] != len(rows) or matrix.shape[1] == 0:
            raise ValueError('Fitted TF-IDF matrix has incompatible dimensions')
        self.feature_count_ = matrix.shape[1]
        self.aspect_heads = []
        for aspect in ASPECTS:
            labels = np.asarray([int(bool(row[aspect])) for row in rows], dtype=np.int64)
            if set(np.unique(labels)) != {0, 1}:
                raise ValueError(f'Aspect {aspect} training head is missing a binary class')
            head = self._make_logistic().fit(matrix, labels)
            self._require_class_order(head.classes_, (0, 1), f'aspect {aspect}')
            if head.n_features_in_ != self.feature_count_:
                raise ValueError('Aspect head feature dimension mismatch')
            self.aspect_heads.append(head)
        self.sentiment_heads = []
        for aspect in CORE_ASPECTS:
            examples = [row for row in rows if row[aspect]]
            labels = np.asarray([row[aspect] for row in examples], dtype=object)
            missing = set(SENTIMENTS) - set(labels)
            if missing:
                raise ValueError(f'{aspect} missing registered sentiments: {", ".join(s for s in SENTIMENTS if s in missing)}')
            head = OneVsRestClassifier(self._make_logistic(), n_jobs=1).fit(
                matrix[[i for i, row in enumerate(rows) if row[aspect]]], labels)
            self._require_class_set(head.classes_, SENTIMENTS, f'sentiment {aspect}')
            if head.estimators_ and any(getattr(estimator, 'n_features_in_', self.feature_count_) != self.feature_count_
                                        for estimator in head.estimators_):
                raise ValueError('Conditional head feature dimension mismatch')
            self.sentiment_heads.append(head)
        self.sentiment_class_order = SENTIMENTS
        self.fitted = True
        return self

    @staticmethod
    def _require_class_order(actual, expected, name):
        actual = tuple(actual.tolist() if hasattr(actual, 'tolist') else actual)
        if actual != tuple(expected):
            raise ValueError(f'Unexpected estimator class order for {name}: {actual}; expected {tuple(expected)}')

    @staticmethod
    def _require_class_set(actual, expected, name):
        actual = tuple(actual.tolist() if hasattr(actual, 'tolist') else actual)
        if len(actual) != len(expected) or set(actual) != set(expected):
            raise ValueError(f'Unexpected estimator classes for {name}: {actual}; expected exactly {tuple(expected)}')

    def _aligned_probabilities(self, estimator, matrix, expected, name):
        import numpy as np
        classes = tuple(estimator.classes_.tolist())
        values = np.asarray(estimator.predict_proba(matrix), dtype=float)
        if values.shape != (matrix.shape[0], len(classes)) or not np.isfinite(values).all():
            raise ValueError(f'Invalid probabilities from {name}')
        if set(classes) != set(expected):
            raise ValueError(f'Missing registered classes from {name}')
        aligned = values[:, [classes.index(label) for label in expected]]
        if len(expected) == 3:
            sums = aligned.sum(axis=1, keepdims=True)
            if np.any(sums <= 0) or not np.isfinite(sums).all():
                raise ValueError(f'Cannot normalize probabilities from {name}')
            aligned = aligned / sums
        return aligned

    def predict_probabilities(self, texts):
        import numpy as np
        if not self.fitted:
            raise ValueError('TF-IDF adapter is not fitted')
        matrix = self.vectorizer.transform(prepare_texts(list(texts)))
        if matrix.shape[1] != self.feature_count_:
            raise ValueError('Prediction sparse feature dimension mismatch')
        aspect = np.column_stack([self._aligned_probabilities(head, matrix, (0, 1), f'aspect {i}')[:, 1]
                                  for i, head in enumerate(self.aspect_heads)])
        conditional = np.stack([self._aligned_probabilities(head, matrix, SENTIMENTS,
                                                              f'sentiment {aspect_name}')
                                for aspect_name, head in zip(CORE_ASPECTS, self.sentiment_heads)], axis=1)
        if aspect.shape != (len(texts), 8) or conditional.shape != (len(texts), 7, 3):
            raise ValueError('TF-IDF output interface mismatch')
        return aspect, conditional
