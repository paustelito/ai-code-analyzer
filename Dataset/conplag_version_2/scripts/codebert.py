import os
import json
import torch

from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

from algorithm import Algorithm

class CodeBERTAlgorithm(Algorithm):

    def __init__(self, version: int, quiet: bool) -> None:
        Algorithm.__init__(self, 'codebert', version, quiet)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/codebert-base"
        )

        self.model = AutoModel.from_pretrained(
            "microsoft/codebert-base"
        )

        os.makedirs(self.GEN_DIR, exist_ok=True)