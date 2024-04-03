from langchain_openai.embeddings import OpenAIEmbeddings
import numpy as np


def cosine_similarity(vec1: np.array, vec2: np.array):
    dot_prod = np.dot(vec1, vec2)
    norm_1 = np.linalg.norm(vec1)
    norm_2 = np.linalg.norm(vec2)
    cos_sim = dot_prod / (norm_1 * norm_2)

    return cos_sim


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Computes the cosine difference between two words/setences")
    parser.add_argument("input1", type=str, help="The word/sentence to perform cosine similarity against")
    parser.add_argument("input2", type=str, help="The word/sentence to perform cosine similarity against")
    parser.add_argument("--model", type=str, default="text-embedding-ada-002")

    args = parser.parse_args()

    embedder = OpenAIEmbeddings(model=args.model)

    vec1 = np.array(embedder.embed_query(args.input1)) 
    vec2 = np.array(embedder.embed_query(args.input2))

    print(
        f"Cosine similarity between '{args.input1}' and '{args.input2}':", 
        cosine_similarity(vec1, vec2)
    )
