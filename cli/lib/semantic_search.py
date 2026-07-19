from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self, device: str | None = None):
        kwargs = {"device": device} if device else {}
        self.model = SentenceTransformer("all-MiniLM-L6-v2", **kwargs)

    def generate_embedding(self, text: str):
        if not text or text.strip() == "":
            raise ValueError("Input text cannot be empty")
        return self.model.encode([text])[0]


def verify_model() -> None:
    ss = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def embed_text(text: str) -> None:
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
