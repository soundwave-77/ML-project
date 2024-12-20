import numpy as np


def compute_title_description_sim(title_embeddings, description_embeddings):
    norm1 = np.linalg.norm(title_embeddings, axis=1, keepdims=True)
    norm2 = np.linalg.norm(description_embeddings, axis=1, keepdims=True)

    # Предотвращаем деление на ноль
    norm1[norm1 == 0] = 1e-10
    norm2[norm2 == 0] = 1e-10

    # Нормализованные векторы
    normalized1 = title_embeddings / norm1
    normalized2 = description_embeddings / norm2

    cosine_similarities = np.sum(normalized1 * normalized2, axis=1)

    return cosine_similarities