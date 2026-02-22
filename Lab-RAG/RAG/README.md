# RAG (Retrieval-Augmented Generator) with Pinecone & Gemini

This laboratory demonstrates the implementation of a **RAG** architecture. The system enhances the capabilities of the Gemini 2.5 Flash model by providing it with a "semantic memory" hosted in Pinecone, allowing it to answer questions about specific unstructured data (Lilian Weng's technical blog) with high precision and reduced hallucinations.

## Architecture

The notebook implements a complete data pipeline and a reasoning loop where the LLM uses a vector database as its external memory:

1. **Engine:** `gemini-2.5-flash` for reasoning and `gemini-embedding-001` for vectorization.

2. **Vector store:** Pinecone (Serverless) configured with 768 dimensions and Cosine Similarity.

3. **Indexing pipeline:**
    - Loading: `WebBaseLoader` with BeautifulSoup to scrape content from Lilian Weng’s blog.
    - Splitting: `RecursiveCharacterTextSplitter` to create chunks of 1000 characters with 200-character overlap.

 4. **Tools:**
    - `retrieve_context`: A custom tool that performs semantic searches in Pinecone and returns relevant document snippets.
    - Logic: An Agent created with create_agent that autonomously decides when to query the knowledge base to avoid hallucinations.

## Installation & execution

### 1. Prerequisites
- Python 3.9 or higher.
- A Google AI Studio API Key.
- A Pinecone API Key (Free tier works).

### 2. Install dependencies

Run the following command to install the required libraries:
```bash
pip install -U langchain-google-genai langchain-pinecone pinecone-client langchain-community beautifulsoup4 langgraph langchain python-dotenv
```

### 3. Environment variables

Create a file named .env in the root directory and add your credentials:
```bash
GOOGLE_API_KEY=your_actual_api_key_here
PINECONE_API_KEY=your_actual_pinecone_api_key_here
```

### 4. Running the Notebook
- Open your editor (VS Code, Jupyter Lab, etc.).
- Select your environment as the kernel.
- Run all cells sequentially to index the blog data and test the agent.

## References and documentation

* **LangChain RAG Tutorial**: [Build a Retrieval Augmented Generation (RAG) App](https://python.langchain.com/docs/tutorials/rag/) — Used for architecting the indexing and retrieval pipelines.
* **Pinecone Integration**: [LangChain Pinecone Vector Store Guide](https://python.langchain.com/docs/integrations/vectorstores/pinecone) — Used for configuring the serverless index and managing vector operations.
* **Source Article**: [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) by Lilian Weng.
