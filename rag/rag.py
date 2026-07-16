"""Minimal retrieval-augmented generation (RAG) example using local Ollama models.

This script demonstrates the complete RAG pipeline without a vector database or
framework. It reads one fact per line from ``cat-facts.txt``, embeds each fact,
stores the resulting vectors in process memory, embeds a user's question, and
ranks facts by cosine similarity. The highest-scoring facts are then inserted
into a grounding prompt for a streaming chat completion.

The module is intentionally independent from the ReAct agent and evaluation
platform elsewhere in this repository. Run it from the repository root so the
relative dataset path resolves correctly. Both models named below must be
available to the local Ollama service, and all indexed data is discarded when
the process exits.
"""

import ollama

EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'
VECTOR_DB = []
dataset = []

def load_dataset():
    """Load newline-delimited cat facts into the module-level ``dataset`` list.

    The function expects ``cat-facts.txt`` in the current working directory.
    Lines are retained as individual retrieval chunks, including their trailing
    newline characters.
    """

    with open('cat-facts.txt', 'r') as file:
        global dataset
        dataset = file.readlines()

def add_chunk_to_database(chunk):
    """Embed one text chunk and append it to the in-memory vector index.

    Args:
        chunk: Source text to index with ``EMBEDDING_MODEL``.

    The index stores ``(text, vector)`` tuples and performs no deduplication or
    persistence. Ollama connection and model errors are allowed to propagate.
    """

    embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
    VECTOR_DB.append((chunk, embedding))


def cosine_similarity(a, b):
    """Return the cosine similarity between two equal-length numeric vectors.

    Values near ``1`` point in similar directions, values near ``0`` are
    orthogonal, and negative values point in opposing directions. This compact
    teaching implementation assumes non-zero vectors of compatible dimensions.
    """

    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([y ** 2 for y in b]) ** 0.5

    return dot_product / (norm_a * norm_b)


def retrieve(query, top_n=3):
    """Retrieve the ``top_n`` indexed chunks most similar to a query.

    The query is embedded with the same model used during indexing. Results are
    returned as ``(chunk, similarity_score)`` tuples sorted from most to least
    similar. Calling this function before indexing returns an empty list.

    Args:
        query: Natural-language search query.
        top_n: Maximum number of ranked chunks to return.
    """

    query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    similarities = []
    
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_n]


if __name__ == '__main__':
    # Load data
    load_dataset()
    print(f'Loaded {len(dataset)} entries')

    # Add chunks to DB
    for i, chunk in enumerate(dataset):
        add_chunk_to_database(chunk)
        print(f'Added chunk {i+1}/{len(dataset)} to the database')


    # Generation phase
    input_query = input('Ask me a question: ')
    retrieved_knowledge = retrieve(input_query)

    print('Retrieved knowledge:')
    for chunk, similarity in retrieved_knowledge:
        print(f' - (similarity: {similarity:.2f}) {chunk}')

    instruction_prompt = f'''You are a helpful chatbot.
    Use only the following pieces of context to answer the question. Don't make up any new information:
    {'\n'.join([f' - {chunk}' for chunk, similarity in retrieved_knowledge])}
    '''

    stream = ollama.chat(
        model=LANGUAGE_MODEL,
        messages=[
            {'role': 'system', 'content': instruction_prompt},
            {'role': 'user', 'content': input_query},
        ],
        stream=True,
    )

    # print the response from the chatbot in real-time
    print('Chatbot response:')
    for chunk in stream:
        print(chunk['message']['content'], end='', flush=True)