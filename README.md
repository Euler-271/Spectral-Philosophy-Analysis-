# Spectral and Information-Geometric Modeling of Philosophical Assumption Structures

A deterministic, local-model research system for analyzing philosophical assumption structures in discourse. This system integrates Wittgenstein-inspired language-game detection, information geometry on simplex-valued sentence distributions, polarity vector-field analysis, and spectral graph diagnostics.

![Graph Visualization](https://via.placeholder.com/800x400?text=Spectral+Graph+Visualization+Placeholder)

## Features

- **Deterministic Analysis**: No generative LLM inference; relies on rule-based and embedding-based local models.
- **Language-Game Detection**: Maps sentences to six distinct language games (Measurement, Causal, Normative, etc.).
- **Polarity Field Analysis**: Computes vector fields over conceptual polarities (e.g., Realism vs. Constructivism).
- **Spectral Graph Diagnostics**: Constructs weighted assumption graphs and analyzes topology via spectral entropy.
- **Interactive Visualization**: React + D3 force-directed graph with real-time instability metrics.

## Architecture

The system consists of a Python FastAPI backend and a React/TypeScript frontend.

- **Backend**: Python 3.10+, FastAPI, spaCy, sentence-transformers, NetworkX, NumPy.
- **Frontend**: React 18, Vite, D3.js, TypeScript.

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- Local CPU environment (GPU optional for embeddings)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/spectral-philosophy-analysis.git
    cd spectral-philosophy-analysis
    ```

2.  **Backend Setup:**
    ```bash
    cd backend
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    ```

3.  **Frontend Setup:**
    ```bash
    cd ../frontend
    npm install
    ```

### Running the Application

1.  **Start the Backend:**
    ```bash
    cd backend
    source .venv/bin/activate
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```

2.  **Start the Frontend:**
    ```bash
    cd frontend
    npm run dev
    ```

3.  **Open the App:**
    Navigate to `http://localhost:5173` in your browser.

## API Reference

### `POST /api/analyze`

Analyzes the input text and returns a comprehensive report including language game distributions, polarity vectors, and graph data.

-   **Input**: `{ "text": "...", "weights": {...}, "config": {...} }`
-   **Output**: JSON object containing analysis results.

## License

MIT License. See [LICENSE](LICENSE) for details.
