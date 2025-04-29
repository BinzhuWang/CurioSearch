# Book Semantic Search API

This Flask application provides a semantic search API for books based on their titles and topic questions. It calculates semantic similarity between a query and the book data, returning the most relevant results. It also allows adding new books to the index dynamically.

## Features

*   **Semantic Search:** Find books based on the meaning of the query, not just keywords.
*   **Dynamic Updates:** Add new books to the search index via an API endpoint without restarting the service.
*   **Configurable Search:** Control the number of results (`count`) and the relevance threshold (`threshold`).
*   **Metadata Return:** Search results include book `id`, `title`, and `similarity_score`.
*   **GPU Acceleration:** Automatically utilizes CUDA-enabled GPU if available for faster embedding calculations.

## Requirements

*   Python 3.11 (Developed and tested with Python 3.11.11)
*   pip (Python package installer)
*   (Optional but recommended for speed) NVIDIA GPU with CUDA installed.

## Installation

1.  **Clone the Repository (if applicable):**
    ```bash
    # git clone <your-repo-url>
    # cd <your-repo-directory>
    ```

2.  **Create Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation

1.  **Prepare CSV:** Ensure your book data is in a CSV file (e.g., `绘本20250428.csv`) containing at least `id`, `title`, and `topic_question` columns.
2.  **Generate Embeddings:** Run the preprocessing script. Update the `csv_path` in `preprocess.py` if needed.
    ```bash
    python preprocess.py
    ```
    This command generates `book_embeddings.npy` (vector representations) and `book_metadata.json` (book IDs and titles).

## Running the Service

Start the Flask API server:

```bash
python app.py
```

By default, the service runs on `http://127.0.0.1:5000`.

## API Usage Examples

**Note:** All POST requests should have the `Content-Type: application/json` header.

### 1. Search for Books (`/search`)

Performs a semantic search for books matching the query.

*   **Method:** `POST`
*   **Request Body:**
    ```json
    {
      "query": "Your search query about book titles or topics",
      "count": 10, 
    }
    ```
    *   `query` (string, required): Search query.
    *   `count` (integer, optional, default: 10): Max number of results.
    *   `threshold` (float, optional, default: 0.5): Min similarity score [0.0, 1.0].

*   **Success Response (200 OK):** Array of matching books.
    ```json
    [
      {
        "id": "book_id_1",
        "title": "Example Book Title 1",
        "similarity_score": 0.85
      },
      {
        "id": "book_id_2",
        "title": "Another Book Title",
        "similarity_score": 0.72
      }
    ]
    ```
*   **Example `curl`:**
    ```bash
    curl -X POST -H "Content-Type: application/json" \
      -d '{"query": "dinosaurs", "count": 5, "threshold": 0.6}' \
      http://127.0.0.1:5000/search
    ```

### 2. Add a New Book (`/add_book`)

Adds a new book entry to the in-memory index and updates the data files.

*   **Method:** `POST`
*   **Request Body:**
    ```json
    {
      "id": "new_book_id_123",
      "title": "The Adventures of a New Book",
      "topic_question": "What makes this new book adventurous?"
    }
    ```
    *   `id` (string, required): Unique book identifier.
    *   `title` (string, required): Book title.
    *   `topic_question` (string, required): Associated topic question.

*   **Success Response (201 Created):** Confirmation message.
    ```json
    {
      "message": "Book added successfully.",
      "new_count": 151 
    }
    ```
*   **Example `curl`:**
    ```bash
    curl -X POST -H "Content-Type: application/json" \
      -d '{"id": "bk101", "title": "Learning About Planets", "topic_question": "Which planet is red?"}' \
      http://127.0.0.1:5000/add_book
    ```

## Configuration

Service configuration is primarily managed within the Python scripts:

*   **`preprocess.py`**: 
    *   `csv_path`: Path to the input CSV data file.
    *   `model_name`: Sentence Transformer model to use.
    *   `output_emb_path`: Path to save the generated embeddings NPY file.
    *   `output_meta_path`: Path to save the generated metadata JSON file.
*   **`app.py`**: 
    *   `MODEL_NAME`: Sentence Transformer model (should match the one used in preprocessing).
    *   `EMBEDDING_FILE`: Path to the embeddings NPY file.
    *   `METADATA_FILE`: Path to the metadata JSON file.
    *   `app.run()` parameters: Host and port for the Flask server (currently `127.0.0.1:5000`). 