import math
import collections
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.util import ngrams

try:
    nltk.data.find("corpora/wordnet.zip")
    nltk.data.find("corpora/omw-1.4.zip")
except LookupError:
    print("Downloading necessary NLTK data...")
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    nltk.download("punkt")
    nltk.download("punkt_tab")


class Metrics:
    def __init__(self, references_corpus=None):
        self.smoothing = SmoothingFunction().method1
        self.cider_idf = None
        if references_corpus:
            self.cider_idf = self._compute_cider_idf(references_corpus)

    def calculate_bleu(self, references, candidate):
        b1 = sentence_bleu(
            references,
            candidate,
            weights=(1.0, 0, 0, 0),
            smoothing_function=self.smoothing,
        )
        b4 = sentence_bleu(
            references,
            candidate,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=self.smoothing,
        )
        return {"BLEU-1": b1, "BLEU-4": b4}

    def calculate_meteor(self, references, candidate):
        ref_sentences = [" ".join(ref) for ref in references]
        cand_sentence = " ".join(candidate)
        return meteor_score(ref_sentences, cand_sentence)

    def calculate_rouge_l(self, references, candidate):
        def lcs_len(x, y):
            m, n = len(x), len(y)
            table = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i - 1] == y[j - 1]:
                        table[i][j] = table[i - 1][j - 1] + 1
                    else:
                        table[i][j] = max(table[i - 1][j], table[i][j - 1])
            return table[m][n]

        best_score = 0.0

        for ref in references:
            lcs = lcs_len(ref, candidate)
            if lcs == 0:
                score = 0.0
            else:
                recall = lcs / len(ref)
                precision = lcs / len(candidate)
                beta = 1.2
                if precision + recall == 0:
                    score = 0.0
                else:
                    score = ((1 + beta**2) * (recall * precision)) / (
                        recall + beta**2 * precision
                    )

            if score > best_score:
                best_score = score

        return best_score

    def _compute_cider_idf(self, corpus_refs, n=4):
        idf_counts = collections.defaultdict(int)
        total_docs = len(corpus_refs)

        for refs in corpus_refs:
            doc_ngrams = set()
            for ref in refs:
                for i in range(1, n + 1):
                    doc_ngrams.update(ngrams(ref, i))

            for ng in doc_ngrams:
                idf_counts[ng] += 1

        idf = {k: math.log(total_docs / (v + 1)) for k, v in idf_counts.items()}
        return idf

    def calculate_cider(self, references, candidate, n=4):
        if not self.cider_idf:
            raise ValueError(
                "CIDEr requires an initialized IDF dictionary. Pass 'references_corpus' to __init__."
            )

        def get_tf_vec(tokens):
            vec = collections.defaultdict(float)
            length = len(tokens)
            for i in range(1, n + 1):
                for ng in ngrams(tokens, i):
                    vec[ng] += 1
            for k in vec:
                vec[k] /= length
            return vec

        cand_vec = get_tf_vec(candidate)

        scores = []
        for ref in references:
            ref_vec = get_tf_vec(ref)

            dot_prod = 0.0
            norm_c = 0.0
            norm_r = 0.0

            all_ngrams = set(cand_vec.keys()) | set(ref_vec.keys())

            for ng in all_ngrams:
                weight = self.cider_idf.get(ng, 0.0)
                tfidf_c = cand_vec.get(ng, 0.0) * weight
                tfidf_r = ref_vec.get(ng, 0.0) * weight

                dot_prod += tfidf_c * tfidf_r
                norm_c += tfidf_c**2
                norm_r += tfidf_r**2

            if norm_c > 0 and norm_r > 0:
                similarity = dot_prod / (math.sqrt(norm_c) * math.sqrt(norm_r))
                scores.append(similarity)
            else:
                scores.append(0.0)

        return np.mean(scores) * 10.0 if scores else 0.0
    
    def calculate_precision(references, candidate):
        best_p = 0.0
        cand_counts = collections.Counter(candidate)

        for ref in references:
            ref_counts = collections.Counter(ref)
            overlap = sum((cand_counts & ref_counts).values())

            if len(candidate) > 0:
                p = overlap / len(candidate)
            else:
                p = 0.0

            if p > best_p:
                best_p = p

        return best_p


    def calculate_recall(references, candidate):
        best_r = 0.0
        cand_counts = collections.Counter(candidate)

        for ref in references:
            ref_counts = collections.Counter(ref)
            overlap = sum((cand_counts & ref_counts).values())

            if len(ref) > 0:
                r = overlap / len(ref)
            else:
                r = 0.0

            if r > best_r:
                best_r = r

        return best_r


    def calculate_f1(references, candidate):
        best_f1 = 0.0
        cand_counts = collections.Counter(candidate)

        for ref in references:
            ref_counts = collections.Counter(ref)
            overlap = sum((cand_counts & ref_counts).values())

            if len(candidate) > 0:
                p = overlap / len(candidate)
            else:
                p = 0.0

            if len(ref) > 0:
                r = overlap / len(ref)
            else:
                r = 0.0

            if p + r > 0:
                f1 = 2 * p * r / (p + r)
            else:
                f1 = 0.0

            if f1 > best_f1:
                best_f1 = f1

        return best_f1
