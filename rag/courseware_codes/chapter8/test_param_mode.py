import os
import lazyllm
from typing import List
import numpy as np
from lazyllm.tools.rag import DocNode

def euclidean_distance(query: List[float], node: List[float], **kwargs) -> float:
    point1 = np.array(query)
    point2 = np.array(node)
    return np.linalg.norm(point1 - point2)


query_t = "hello world."
node_t = [DocNode(text="hello lazyllm.")]
query_e = {"key": [1.0, 0.4, 2.1]}
node_e = [DocNode(embedding={"key": [4.2, 2.1, 3.9]})]


func1 = lazyllm.tools.rag.register_similarity(euclidean_distance, mode="text")
func2 = lazyllm.tools.rag.register_similarity(euclidean_distance, mode="embedding")

ret = func1(query_e, node_e)
print(ret)






































# exit()
# func1 = lazyllm.tools.rag.register_similarity(euclidean_distance, mode="text")
# func2 = lazyllm.tools.rag.register_similarity(euclidean_distance, mode="embedding")



# # ret = func1(query_t, node_t)
# # print(f"ret: {ret}")  # TypeError: unsupported operand type(s) for -: 'str' and 'DocNode'


# ret = func1(query_e, node_e)
# print(f"ret: {ret}")  # TypeError: unsupported operand type(s) for -: 'dict' and 'DocNode'


# ret = func2(query_t, node_t)
# print(f"ret: {ret}")  # AssertionError: query must be of dict type, used for similarity calculation.


# ret_2e = func2(query_e, node_e)
# print(f"ret_2e: {ret_2e}")   # ret: {'key': [(<Node id=2865d5c9-730b-4fda-8077-57b706944ad9>, 4.045985664828782)]}