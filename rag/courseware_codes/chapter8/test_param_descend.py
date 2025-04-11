import lazyllm
import os
import os
import os, sys
curt_file_path = os.path.realpath(__file__) if "__file__" in globals() else os.getcwd()
sys.path = ["/home/mnt/jisiyuan/projects/LazyLLM"] + sys.path
from lazyllm import OnlineEmbeddingModule, SentenceSplitter, Retriever
from lazyllm.tools.rag import Document

import lazyllm
from typing import List
import numpy as np

@lazyllm.tools.rag.register_similarity(mode="embedding", descend=False)
def euclidean_distance(query: List[float], node: List[float], **kwargs) -> float:
    point1 = np.array(query)
    point2 = np.array(node)
    return np.linalg.norm(point1 - point2)
























































# documents = Document(os.path.join(os.getcwd(), "rag_data"), 
# embed=OnlineEmbeddingModule(), manager=False)
# documents.create_node_group(name="sentences", 
# transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
# ppl = Retriever(documents, group_name="CoarseChunk", 
# similarity="euclidean_distance", similarity_cut_off=0.003, topk=3)

# query = "人工智能"
# nodes = ppl(query)
# print(f"query: {query}")
# print("results:")
# for node in nodes:
#     print(f"node: {node.text}\n")
    
