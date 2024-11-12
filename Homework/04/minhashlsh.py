import pandas as pd
import numpy as np
from math import ceil

from minhash import MinHash

class MinHashLSH(MinHash):
    def __init__(self, num_permutations: int, num_buckets: int, threshold: float):
        self.num_permutations = num_permutations
        self.num_buckets = num_buckets
        self.threshold = threshold
        
    def get_buckets(self, minhash: np.array) -> np.array:
        '''
        Возвращает массив из бакетов, где каждый бакет представляет собой N строк матрицы сигнатур.
        '''
        
        minhash_buckets = []
        
        bucket_size = ceil(minhash.shape[0]/min(self.num_buckets, self.num_permutations))

        start_index = 0

        while start_index + bucket_size < minhash.shape[0]:
            minhash_buckets.append(minhash[start_index: start_index + bucket_size, :])
            start_index+= bucket_size

        if start_index < minhash.shape[0]:
            minhash_buckets.append(minhash[start_index: , :])
    
        return np.array(minhash_buckets, dtype=object)
    
    def get_similar_candidates(self, buckets) -> list[tuple]:
        '''
        Находит потенциально похожих кандижатов.
        Кандидаты похожи, если полностью совпадают мин хеши хотя бы в одном из бакетов.
        Возвращает список из таплов индексов похожих документов.
        '''

        similar_candidates = []

        for bucket in buckets:
            rows, cols = bucket.shape

            for c in range(cols):
                for c_next in range(c+1, cols):
                    if (c, c_next) in similar_candidates:
                        continue

                    flag = True
                    for r in range(rows):
                        if bucket[r][c] != bucket[r][c_next]:
                            flag = False
                            break

                    if flag == True:
                        similar_candidates.append((c, c_next))

        return similar_candidates
        
    def run_minhash_lsh(self, corpus_of_texts: list[str]) -> list[tuple]:
        occurrence_matrix = self.get_occurrence_matrix(corpus_of_texts)
        minhash = self.get_minhash(occurrence_matrix)
        buckets = self.get_buckets(minhash)
        similar_candidates = self.get_similar_candidates(buckets)
        
        return set(similar_candidates)
    
