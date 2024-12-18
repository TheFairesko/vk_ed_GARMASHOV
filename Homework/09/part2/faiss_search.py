from typing import List, Optional
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from part1.search_engine import Document, SearchResult

class FAISSSearcher:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Инициализация индекса
        """
        self.model = SentenceTransformer(model_name)
        self.documents: List[Document] = []
        self.index: Optional[faiss.Index] = None
        self.dimension: int = 384  # Размерность для 'all-MiniLM-L6-v2'

    def build_index(self, documents: List[Document]) -> None:
        """
        TODO: Реализовать создание FAISS индекса
        
        1. Сохранить документы
        2. Получить эмбеддинги через model.encode()
        3. Нормализовать векторы (faiss.normalize_L2)
        4. Создать индекс:
            - Создать quantizer = faiss.IndexFlatIP(dimension)
            - Создать индекс = faiss.IndexIVFFlat(quantizer, dimension, n_clusters)
            - Обучить индекс (train)
            - Добавить векторы (add)
        """
        self.documents = documents
        documents_emb = np.array([self.model.encode((doc.title + " " + doc.text), convert_to_numpy=True) for doc in documents])

        faiss.normalize_L2(documents_emb)

        quantizer = faiss.IndexFlatIP(self.dimension)

        n_clusters = int(np.ceil(np.sqrt(len(documents))))
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, n_clusters, faiss.METRIC_INNER_PRODUCT)

        self.index.train(documents_emb)
        self.index.add(documents_emb)

    def save(self, path: str) -> None:
        """
        TODO: Реализовать сохранение индекса
        
        1. Сохранить в pickle:
            - documents
            - индекс (faiss.serialize_index)
        """
        with open(path, 'wb') as f:
            pickle.dump({'documents': self.documents, 'index': faiss.serialize_index(self.index)}, f)

    def load(self, path: str) -> None:
        """
        TODO: Реализовать загрузку индекса
        
        1. Загрузить из pickle:
            - documents
            - индекс (faiss.deserialize_index)
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.index = faiss.deserialize_index(data['index'])

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        TODO: Реализовать поиск
        
        1. Получить эмбеддинг запроса
        2. Нормализовать вектор
        3. Искать через index.search()
        4. Вернуть найденные документы
        """
        query_emb = self.model.encode(query, convert_to_numpy=True)
        query_emb = query_emb.reshape(1, -1)

        faiss.normalize_L2(query_emb)

        scores, ids = self.index.search(query_emb, top_k)

        results = []
        for score_k, id_k in zip(scores[0], ids[0]):
            doc = self.documents[id_k]
            
            results.append(SearchResult(
                doc_id=doc.id,
                score=score_k,
                title=doc.title,
                text=doc.text
            ))

        return results

    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[SearchResult]]:
        """
        TODO: Реализовать batch-поиск
        
        1. Получить эмбеддинги всех запросов
        2. Нормализовать векторы
        3. Искать через index.search()
        4. Вернуть результаты для каждого запроса
        """
        query_emb = self.model.encode(queries, convert_to_numpy=True)

        faiss.normalize_L2(query_emb)

        scores, ids = self.index.search(query_emb, top_k)

        results = []
        for q_scores, q_ids in zip(scores, ids):
            query_results = []

            for score_k, id_k in zip(q_scores, q_ids):
                doc = self.documents[id_k]
                query_results.append(SearchResult(
                    doc_id=doc.id,
                    score=score_k,
                    title=doc.title,
                    text=doc.text
                ))

            results.append(query_results)

        return results
