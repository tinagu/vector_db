import chromadb
from chromadb.api import CreateCollectionConfiguration
from chromadb.api.collection_configuration import json_to_create_hnsw_configuration
from chromadb.errors import NotFoundError
import os
from fastembed import TextEmbedding, LateInteractionTextEmbedding
import pymongo
import uuid
import pickle
import numpy as np
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridCollection:

    def __init__(self, collection_name_or_path):
        self.chroma_client = chromadb.PersistentClient(path=os.path.abspath(os.path.join(".",
                                                                                 collection_name_or_path)))

        try:
            self.chroma_collection_dense = self.chroma_client.get_collection(name=f"{collection_name_or_path}_dense")
        except NotFoundError:
            logger.exception(f"Collection {collection_name_or_path} not found.")
            self.chroma_collection_dense = self.chroma_client.get_or_create_collection(name=f"{collection_name_or_path}_dense",
                                                                                       configuration=CreateCollectionConfiguration(
                                                                  hnsw=json_to_create_hnsw_configuration({
                                                                      "space": "cosine"
                                                                  })
                                                              ))

        self.mongo_client = pymongo.MongoClient()
        self.my_mongo_db = self.mongo_client.get_database(collection_name_or_path)

        self.mongo_collection_multi = self.my_mongo_db.get_collection(f"{collection_name_or_path}_multi")
        self.mongo_collection_doc = self.my_mongo_db.get_collection(f"{collection_name_or_path}_doc")

        self.dense_model = TextEmbedding('BAAI/bge-base-en')
        self.cobert_model = LateInteractionTextEmbedding('colbert-ir/colbertv2.0')

    def ingest(self,
               answer,
               question=None,
               doc_id=None):
        if not doc_id:
            doc_id = str(uuid.uuid4())

        self.mongo_collection_doc.insert_one({
            "doc_id": doc_id,
            "question": question,
            "answer": answer,
        })

        dense_embedding = list(self.dense_model.passage_embed(answer))
        self.chroma_collection_dense.add(
            ids=[doc_id],
            embeddings=dense_embedding,
            metadatas={
                "doc_id": doc_id,
            }
        )

        cobert_embedding = list(self.cobert_model.passage_embed(answer))
        self.mongo_collection_multi.insert_one({
            "doc_id": doc_id,
            "cobert_embedding": pickle.dumps(cobert_embedding)
        })

    def search(self, question):
        query_dense_vector = next(self.dense_model.query_embed(question))
        query_cobert_vector = next(self.cobert_model.query_embed(question))

        dense_vectors_result = self.chroma_collection_dense.query(
            query_embeddings=query_dense_vector,
            n_results=20
        )
        rerank_results = {}
        for meta_data, distance in zip(dense_vectors_result["metadatas"][0], dense_vectors_result["distances"][0]):
            doc_id: str = meta_data["doc_id"]

            doc = self.mongo_collection_doc.find_one({"doc_id": doc_id})
            if not doc:
                continue

            rerank_results[doc_id] = {
                "cos_sim": distance,
                "max_sim": 0,
                "answer": doc["answer"],
                "answer_question": doc["question"],
            }

            str_cobert_embedding = self.mongo_collection_multi.find_one({"doc_id": doc_id})
            if not str_cobert_embedding:
                continue

            cobert_embedding = np.array(pickle.loads(str_cobert_embedding["cobert_embedding"]), np.float32)
            if cobert_embedding.ndim == 3:
                cobert_embedding = cobert_embedding.squeeze(0)

            cos_sim = np.dot(query_cobert_vector, cobert_embedding.T)
            max_score = cos_sim.max(axis=1)
            rerank_results[doc_id]["max_sim"] = max_score.sum()

        return rerank_results
