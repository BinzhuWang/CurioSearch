# app.py
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
import torch
import os
import threading

app = Flask(__name__)

# 设置JSON响应不转义非ASCII字符（如中文）
app.config['JSON_AS_ASCII'] = False
app.json.ensure_ascii = False

# --- Configuration ---
MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_FILE = "book_embeddings.npy"
METADATA_FILE = "book_metadata.json"
# --- End Configuration ---

# --- Global Variables ---
model = None
book_embeddings = None
book_metadata = None
device = None
# Add a lock for thread safety when modifying shared resources (files and in-memory data)
data_lock = threading.Lock()
# --- End Global Variables ---

def load_model_and_data():
    """Loads the Sentence Transformer model and precomputed data."""
    # Use lock to ensure thread-safe loading if called concurrently (though usually called once at start)
    with data_lock:
        global model, book_embeddings, book_metadata, device

        print("Loading resources...")
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        # Load Sentence Transformer model
        print(f"Loading model: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME, device=device)

        # Load precomputed embeddings
        if not os.path.exists(EMBEDDING_FILE):
            # If file doesn't exist, initialize empty structures
            print(f"Warning: Embedding file {EMBEDDING_FILE} not found. Initializing empty embeddings.")
            book_embeddings = torch.empty((0, model.get_sentence_embedding_dimension()), device=device) # Empty tensor with correct dimension
        else:
            print(f"Loading embeddings from: {EMBEDDING_FILE}")
            try:
                embeddings_np = np.load(EMBEDDING_FILE)
                book_embeddings = torch.from_numpy(embeddings_np).to(device)
            except Exception as e:
                print(f"Error loading embeddings: {e}. Initializing empty embeddings.")
                book_embeddings = torch.empty((0, model.get_sentence_embedding_dimension()), device=device)


        # Load metadata
        if not os.path.exists(METADATA_FILE):
             print(f"Warning: Metadata file {METADATA_FILE} not found. Initializing empty metadata.")
             book_metadata = []
        else:
            print(f"Loading metadata from: {METADATA_FILE}")
            try:
                with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                    book_metadata = json.load(f)
                # Ensure metadata is a list
                if not isinstance(book_metadata, list):
                    print(f"Warning: Metadata in {METADATA_FILE} is not a list. Initializing empty metadata.")
                    book_metadata = []
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {METADATA_FILE}: {e}. Initializing empty metadata.")
                book_metadata = []
            except Exception as e:
                 print(f"Error loading metadata: {e}. Initializing empty metadata.")
                 book_metadata = []


        if book_embeddings.shape[0] != len(book_metadata):
             # Attempt to reconcile or raise error if critical mismatch
             print(f"Warning: Mismatch between number of embeddings ({book_embeddings.shape[0]}) and metadata entries ({len(book_metadata)}). Check data integrity.")
             # Decide on recovery strategy: maybe truncate the longer one? Or fail?
             # For now, we print a warning and proceed, which might lead to errors later.
             # A safer approach might be to truncate to the minimum length or refuse to start.
             min_len = min(book_embeddings.shape[0], len(book_metadata))
             print(f"Attempting to proceed with {min_len} entries.")
             book_embeddings = book_embeddings[:min_len]
             book_metadata = book_metadata[:min_len]
             # Or uncomment below to raise an error:
             # raise ValueError("Mismatch between number of embeddings and metadata entries. Cannot proceed.")


        print(f"Resources loaded. Embeddings shape: {book_embeddings.shape}, Metadata count: {len(book_metadata)}")

@app.route('/search', methods=['POST'])
def search():
    """
    Handles search requests.
    Expects JSON payload with 'query', optionally 'count', and 'threshold'.
    'count' defaults to 10 if not provided.
    """
    # Added lock for reading shared data, although less critical than writing
    with data_lock:
        if not model or book_embeddings is None or book_metadata is None:
            return jsonify({"error": "Service not initialized correctly."}), 500

        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' in request JSON."}), 400

        query = data['query']
        # Default values for count and threshold if not provided
        count = data.get('count', 10) # Use 'count', default to 10
        threshold = data.get('threshold', 0.1) # Example threshold

        if not isinstance(count, int) or count <= 0:
            return jsonify({"error": "'count' must be a positive integer."}), 400
        if not isinstance(threshold, (float, int)) or not (0 <= threshold <= 1):
             return jsonify({"error": "'threshold' must be a number between 0 and 1."}), 400

        print(f"Received search query: '{query}', count={count}, threshold={threshold}")

        # 1. Encode the query
        query_embedding = model.encode(query, convert_to_tensor=True, device=device)
        print(f"Query embedding: {query_embedding}")

        # 2. Compute cosine similarities
        # util.cos_sim returns a tensor of shape [1, num_book_embeddings]
        cosine_scores = util.cos_sim(query_embedding, book_embeddings)[0] # Get the first (and only) row

        # 3. Find top K matches above the threshold
        # Combine scores with their indices
        all_matches = []
        for idx, score in enumerate(cosine_scores):
            score_item = score.item() # Convert tensor score to float
            if score_item >= threshold:
                all_matches.append({'index': idx, 'score': score_item})

        # Sort matches by score in descending order
        all_matches = sorted(all_matches, key=lambda x: x['score'], reverse=True)

        # Get the top 'count' results
        top_results = all_matches[:count] # Use count here

        # 4. Prepare response
        results_payload = []
        for match in top_results:
            # Double check index validity before accessing metadata
            if match['index'] < len(book_metadata):
                book_info = book_metadata[match['index']]
                results_payload.append({
                    'id': book_info.get('id', 'N/A'), # Add id field, safely get id
                    'title': book_info.get('title', 'N/A'), # Safely get title
                    # Add other metadata fields here if needed
                    'similarity_score': match['score']
                })
            else:
                print(f"Warning: Index {match['index']} out of bounds for metadata (length {len(book_metadata)}). Skipping result.")

        print(f"Found {len(results_payload)} results matching criteria.")
        return jsonify(results_payload)

@app.route('/add_book', methods=['POST'])
def add_book():
    """
    Adds a new book to the collection.
    Expects JSON payload with 'id', 'title', and 'topic_question'.
    Updates both in-memory data and persists changes to files.
    """
    global book_embeddings, book_metadata # Allow modification of globals

    if not model:
        return jsonify({"error": "Model not loaded."}), 503 # Service Unavailable

    data = request.get_json()
    if not data or 'id' not in data or 'title' not in data or 'topic_question' not in data:
        return jsonify({"error": "Missing 'id', 'title', or 'topic_question' in request JSON."}), 400

    book_id = data['id']
    title = data['title']
    topic_question = data['topic_question']
    print(f"Received request to add book: ID='{book_id}', Title='{title}'")

    # 1. Prepare text and generate embedding
    text_to_embed = f"{title} {topic_question}" # Combine title and topic question
    new_embedding = model.encode(text_to_embed, convert_to_tensor=True, device=device)
    # Ensure the new embedding is 2D ( [1, embedding_dim] ) for concatenation
    new_embedding = new_embedding.unsqueeze(0)

    # 2. Prepare metadata
    new_metadata_entry = {"id": book_id, "title": title} # Add id and title

    # 3. Acquire lock and update shared resources (in-memory and files)
    with data_lock:
        print("Acquired lock to update data...")
        try:
            # Update in-memory data
            book_embeddings = torch.cat((book_embeddings, new_embedding), dim=0)
            book_metadata.append(new_metadata_entry)

            # Persist changes to files
            print(f"Saving updated embeddings to: {EMBEDDING_FILE}")
            # Move to CPU before saving with numpy
            np.save(EMBEDDING_FILE, book_embeddings.cpu().numpy())

            print(f"Saving updated metadata to: {METADATA_FILE}")
            with open(METADATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(book_metadata, f, ensure_ascii=False, indent=4)

            print(f"Successfully added book. New count: {len(book_metadata)}")
            # Return success response while still holding the lock
            return jsonify({"message": "Book added successfully.", "new_count": len(book_metadata)}), 201

        except Exception as e:
            # Important: If file saving fails, should we rollback the in-memory changes?
            # For simplicity here, we don't rollback, which could lead to inconsistency
            # if the service crashes immediately after. A more robust solution would
            # involve transactional writes or saving to temp files first.
            print(f"Error adding book: {e}")
            # Return error response while still holding the lock
            return jsonify({"error": f"Failed to add book: {e}"}), 500
        finally:
            # Lock is automatically released when exiting the 'with' block
             print("Released lock.")

if __name__ == '__main__':
    try:
        load_model_and_data()
        # Run Flask app
        # Use host='0.0.0.0' to make it accessible from outside the container/machine if needed
        app.run(host='127.0.0.1', port=5000, debug=False) # Turn debug=False for production
    except FileNotFoundError as e:
        print(f"Error initializing service: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
