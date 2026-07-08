from pathlib import Path
from flask import Flask, jsonify, render_template, request
import ollama

BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDING_MODEL = 'hf.co/CompendiumLabs/bge-base-en-v1.5-gguf'
LANGUAGE_MODEL = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'

app = Flask(__name__, static_folder='static', template_folder='templates')
VECTOR_DB = []


def load_dataset():
    dataset_path = BASE_DIR / 'cat-facts.txt'
    if not dataset_path.exists():
        return []

    with open(dataset_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0.0


def build_vector_db(dataset):
    for chunk in dataset:
        try:
            embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
            VECTOR_DB.append((chunk, embedding))
        except ConnectionError as exc:
            raise RuntimeError(
                'Could not connect to Ollama while building the knowledge database.'
            ) from exc


def retrieve(query, top_n=3):
    if not VECTOR_DB:
        return []

    try:
        query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
    except ConnectionError as exc:
        raise RuntimeError('Could not connect to Ollama while retrieving query embeddings.') from exc

    scored = [
        (chunk, cosine_similarity(query_embedding, embedding))
        for chunk, embedding in VECTOR_DB
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return [chunk for chunk, _ in scored[:top_n]]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.get_json(force=True)
    prompt = data.get('prompt', '').strip()
    if not prompt:
        return jsonify({'error': 'Prompt cannot be empty.'}), 400

    context = retrieve(prompt)
    instruction_prompt = (
        'You are a helpful chatbot. Use only the following context to answer the question. ' 
        'Do not invent information.\n\n' + '\n'.join(f'- {line}' for line in context)
    )

    try:
        response_stream = ollama.chat(
            model=LANGUAGE_MODEL,
            messages=[
                {'role': 'system', 'content': instruction_prompt},
                {'role': 'user', 'content': prompt},
            ],
            stream=True,
        )
    except ConnectionError as exc:
        return jsonify({'error': 'Could not connect to Ollama for chat generation.'}), 503

    full_response = ''
    for chunk in response_stream:
        full_response += chunk['message']['content']

    return jsonify({'response': full_response})


if __name__ == '__main__':
    dataset = load_dataset()
    if not dataset:
        print('No dataset found. Please place cat-facts.txt in the repository root.')
    else:
        build_vector_db(dataset)
        print(f'Loaded {len(dataset)} knowledge chunks.')
        app.run(host='0.0.0.0', port=5000, debug=True)
