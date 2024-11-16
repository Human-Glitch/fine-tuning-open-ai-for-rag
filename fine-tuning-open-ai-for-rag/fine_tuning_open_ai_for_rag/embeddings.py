from typing import List
import numpy as np
import pandas as pd
from tenacity import retry, wait_exponential
from tqdm import tqdm
from dotenv import load_dotenv
from preprocessor import Preprocessor
from fastembed import TextEmbedding

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import os

class QdrantService:
    def __init__(self, qdrant_url, qdrant_api_key) -> None:
        self.qdrant_client = QdrantClient(
            url=qdrant_url, 
            api_key=qdrant_api_key, 
            timeout=6000, 
            prefer_grpc=True)
        
        
        self.collection_name = "squadv2-cookbook"
        
        # Create the collection, run this only once
        # self.qdrant_client.create_collection(
        #     collection_name=self.collection_name,
        #     vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        # )
        
    def generate_points_from_dataframe(self, df: pd.DataFrame) -> List[PointStruct]:
        batch_size = 512
        questions = df["question"].tolist()
        total_batches = len(questions) // batch_size + 1
        
        pbar = tqdm(total=len(questions), desc="Generating embeddings")
        
        embedding_model = TextEmbedding()
        
        # Generate embeddings in batches to improve performance
        embeddings = []
        for i in range(total_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(questions))
            batch = questions[start_idx:end_idx]
            batch_embeddings = list(embedding_model.embed(batch, batch_size=batch_size))
            embeddings.extend(batch_embeddings)
            pbar.update(len(batch))
            
        pbar.close()
        
        # Convert embeddings to list of lists
        embeddings_list = [embedding.tolist() for embedding in embeddings]
        
        # Create a temporary DataFrame to hold the embeddings and existing DataFrame columns
        temp_df = df.copy()
        temp_df["embeddings"] = embeddings_list
        temp_df["id"] = temp_df.index
        
        # Generate PointStruct objects using DataFrame apply method
        points = temp_df.progress_apply(
            lambda row: PointStruct(
                id=row["id"],
                vector=row["embeddings"],
                payload={
                    "question": row["question"],
                    "title": row["title"],
                    "context": row["context"],
                    "is_impossible": row["is_impossible"],
                    "answers": row["answers"],
                },
            ),
            axis=1,
        ).tolist()

        return points
        
    @retry(wait=wait_exponential(multiplier=1, min=2, max=6))
    def upsert(self, points):
        return self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        
    def get_few_shot_prompt(self, row):
        query, row_context = row["question"], row["context"]
        
        embedding_model = TextEmbedding()

        embeddings = list(embedding_model.embed([query]))
        query_embedding = embeddings[0].tolist()

        num_of_qa_to_retrieve = 5

        # Query Qdrant for similar questions that have an answer
        q1 = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            with_payload=True,
            limit=num_of_qa_to_retrieve,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="is_impossible",
                        match=models.MatchValue(
                            value=False,
                        ),
                    ),
                ],
            )
        )

        # Query Qdrant for similar questions that are IMPOSSIBLE to answer
        q2 = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="is_impossible",
                        match=models.MatchValue(
                            value=True,
                        ),
                    ),
                ]
            ),
            with_payload=True,
            limit=num_of_qa_to_retrieve,
        )


        instruction = """Answer the following Question based on the Context only. Only answer from the Context. If you don't know the answer, say 'I don't know'.\n\n"""
        # If there is a next best question, add it to the prompt
    
        def q_to_prompt(q):
            question, context = q.payload["question"], q.payload["context"]
            answer = q.payload["answers"][0] if len(q.payload["answers"]) > 0 else "I don't know"
            return [
                {
                    "role": "user", 
                    "content": f"""Question: {question}\n\nContext: {context}\n\nAnswer:"""
                },
                {"role": "assistant", "content": answer},
            ]

        rag_prompt = []
        
        if len(q1) >= 1:
            rag_prompt += q_to_prompt(q1[1])
        if len(q2) >= 1:
            rag_prompt += q_to_prompt(q2[1])
        if len(q1) >= 1:
            rag_prompt += q_to_prompt(q1[2])
        
        rag_prompt += [
            {
                "role": "user",
                "content": f"""Question: {query}\n\nContext: {row_context}\n\nAnswer:"""
            },
        ]

        rag_prompt = [{"role": "system", "content": instruction}] + rag_prompt
        return rag_prompt

# load_dotenv() 
# tqdm.pandas()

# train_df = Preprocessor.json_to_dataframe_with_titles(json.load(open('local_cache/train.json')))

# qdrant_service = QdrantService(os.getenv("QDRANT_URL"), os.getenv("QDRANT_API_KEY"))
# points = qdrant_service.generate_points_from_dataframe(train_df)

# operation_info = qdrant_service.upsert(points)

# print(operation_info)

