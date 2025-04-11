from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from scipy.linalg import norm
import sys
import heapq
from typing import List, Tuple
from lazyllm.tools.rag import DocNode
import lazyllm
import os
from lazyllm import pipeline, bind, OnlineEmbeddingModule, SentenceSplitter, Retriever, Reranker
from lazyllm.tools.rag import Document


@lazyllm.tools.rag.register_similarity(mode='text', batch=True)
def tfidf_similarity(query: str, nodes: List[DocNode], **kwargs) -> List[Tuple[DocNode, float]]:
    def add_space(s):
        return ' '.join(list(s))
    corpus_tokens = [add_space(node.get_text()) for node in nodes]
    query = add_space(query)
    topk = min(len(nodes), kwargs.get("topk", sys.maxsize))
    datasets = [[query, corpus] for corpus in corpus_tokens]
    cv = TfidfVectorizer(tokenizer=lambda s: s.split())
    vectors = [cv.fit_transform(corpus).toarray() for corpus in datasets]
    scores = [np.dot(vector[0], vector[1]) / (norm(vector[0]) * norm(vector[1])) for vector in vectors]
    indexes = heapq.nlargest(topk, range(len(scores)), scores.__getitem__)
    results = [(nodes[idx], score) for idx, score in zip(indexes, scores)]
    return results


prompt = ('You will play the role of an AI Q&A assistant and complete a dialogue task. '
          'In this task, you need to provide your answer based on the given context and question.')

documents = Document(dataset_path=os.path.join(os.getcwd(), "rag_data"),
                     embed=OnlineEmbeddingModule(source="glm", embed_model_name="embedding-2"), manager=False)
documents.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)

with pipeline() as ppl:
    ppl.prl = Retriever(documents, group_name="CoarseChunk", similarity="tfidf_similarity",
                        similarity_cut_off=0.003, topk=3)
    ppl.reranker = Reranker("ModuleReranker",
                            model=OnlineEmbeddingModule(type="rerank", source="glm", embed_model_name="rerank"),
                            topk=1, output_format='content', join=True) | bind(query=ppl.input)
    ppl.formatter = (lambda nodes, query: dict(context_str=nodes, query=query)) | bind(query=ppl.input)
    ppl.llm = lazyllm.OnlineChatModule(source="glm",
                                       model="glm-4",
                                       stream=False).prompt(lazyllm.ChatPrompter(prompt, extra_keys=["context_str"]))
print(ppl("全国住房城乡建设工作会议的主要内容"))
