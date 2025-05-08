# preprocess_books.py
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import torch

def preprocess_and_save_embeddings(csv_path="/Users/binzhu/Desktop/CurioSearch/book_data.csv", model_name='all-MiniLM-L6-v2', output_emb_path="book_embeddings.npy", output_meta_path="book_metadata.json"):
    """
    Reads book data, generates embeddings for title and topic question,
    and saves embeddings and metadata.
    """
    print(f"Loading SentenceTransformer model: {model_name}")
    # Initialize the Sentence Transformer model
    # Use CUDA if available, otherwise CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = SentenceTransformer(model_name, device=device)

    print(f"Reading book data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        # Handle potential missing values in title or topic question
        df['title'] = df['title'].fillna('')
        df['topic_question'] = df['topic_question'].fillna('')
    except FileNotFoundError:
        print(f"Error: {csv_path} not found. Please make sure the file exists.")
        return
    except KeyError as e:
        print(f"Error: Column {e} not found in {csv_path}. Required columns: 'id', 'title', 'topic_question'.")
        return

    # Combine title and topic question for embedding generation
    # Assuming we want to represent each book by the combination of its title and topic question
    df['text_to_embed'] = df['title'] + " " + df['topic_question'] # Add a space separator
    texts_to_embed = df['text_to_embed'].tolist()

    print(f"Generating embeddings for {len(texts_to_embed)} books...")
    # Generate embeddings
    embeddings = model.encode(texts_to_embed, convert_to_tensor=True, show_progress_bar=True)

    # Save embeddings to a .npy file
    # Move embeddings to CPU before saving with numpy
    embeddings_np = embeddings.cpu().numpy()
    print(f"Saving embeddings to: {output_emb_path}")
    np.save(output_emb_path, embeddings_np)

    # Prepare metadata (e.g., save titles or other identifiers)
    # Here we save the id and title
    metadata = df[['id', 'title']].to_dict(orient='records') # Saving id and title as a list of dictionaries
    print(f"Saving metadata to: {output_meta_path}")
    with open(output_meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    print("Preprocessing finished successfully.")

if __name__ == "__main__":
    # Example usage: Using the updated CSV path
    preprocess_and_save_embeddings(csv_path="/Users/binzhu/Desktop/CurioSearch/book_data.csv")