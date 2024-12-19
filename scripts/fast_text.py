import fasttext
import numpy as np
import razdel

def load_fasttext_model(model_path='cc.ru.300.bin'):
    """Load FastText model from the specified path."""
    return fasttext.load_model(model_path)

def get_embedding(text, model):
    """Compute the average embedding for the input text using the given FastText model."""
    words = [token.text.lower() for token in razdel.tokenize(text) if token.text.isalpha()]
    embeddings_list = [model.get_word_vector(word) for word in words if word in model.words]

    if not embeddings_list:
        return np.zeros(model.get_dimension())

    return np.mean(embeddings_list, axis=0)

# Example usage
# model = load_fasttext_model('cc.ru.300.bin')
# embedding = get_embedding("Пример текста", model)