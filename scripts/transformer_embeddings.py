from sentence_transformers import SentenceTransformer
import torch
from tqdm.notebook import tqdm
from sklearn.decomposition import PCA
import numpy as np


def reduce_dim(embeddings):
    embeddings = PCA(n_components=100).fit_transform(embeddings)
    return embeddings


class TransformersEmbedder:
    def __init__(self, hf_model_name: str, batch_size: int = 64, reduce_dim: bool = True):
        self.model = SentenceTransformer(hf_model_name)
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reduce_dim = reduce_dim

    def embed(self, texts: list[str]) -> np.ndarray:
        embeddings = []
        for idx in tqdm(range(0, len(texts), self.batch_size)):
            batch_embeddings = self.model.encode(texts[idx: idx + self.batch_size], batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True).astype(np.float16)
            embeddings.append(batch_embeddings)

        embeddings = np.vstack(embeddings)

        return reduce_dim(embeddings) if self.reduce_dim else embeddings