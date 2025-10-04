# Step 1 - Document Conversion
import time
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
    TableFormerMode,
    AcceleratorDevice,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
import torch
from docling.chunking import HybridChunker
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer
# Step 2 - Chunking and Embedding
embeddings_model_path = "ibm-granite/granite-embedding-30m-english"
embeddings_model = HuggingFaceEmbeddings(
    model_name=embeddings_model_path,
)

embeddings_tokenizer = AutoTokenizer.from_pretrained(embeddings_model_path)

chunker = HybridChunker(
    tokenizer=embeddings_tokenizer,
    merge_peers=True,
)


# ✅ FIXED: Pass result (Document object) instead of converter
chunk_iter = chunker.chunk(dl_doc=result)

# ✅ FIXED: Convert to list once to avoid iterator exhaustion
chunks = list(chunk_iter)

# Now iterate over the list instead of the exhausted iterator
for i, chunk in enumerate(chunks):
    print(f"=== {i} ===")
    print(f"chunk.text:\n{chunk.text[:300]}…")

    enriched_text = chunker.contextualize(chunk=chunk)
    print(f"chunker.contextualize(chunk):\n{enriched_text[:300]}…")
    print()


for i, chunk in enumerate(chunks):
# Get contextualized text (includes heading hierarchy)
contextualized_text = self.chunker.contextualize(chunk=chunk)

# Count actual tokens
token_count = len(self.tokenizer.encode(contextualized_text))

# Create chunk metadata
chunk_metadata = {
** base_metadata,
"total_chunks": len(chunks)
"token_count": token_count,
"has_context": True # Flag indicating contextualized chunk
Edit Ctrl| Chat Ctri+L

# Estimate character positions
start_char = current_pos
end_char = start_char + len(contextualized_text)

document_chunks.append(DocumentChunk(
content=contextualized_text.strip(),
index=i,
start_char=start_char,
end_char=end_char,
metadata=chunk_metadata,
token_count=token_count
# save to vector store
# function for FAISS
# function for milvus
# note each Store all document vectors in one or a few collections.

Attach metadata per chunk (e.g., file_id, page_number, section, user_id).

Easily filter or group results by any metadata field at retrieval time.
# we need to filter by file when retrieving
# tool should support crewai[too]
# the query should be embedded and searched in the vector store

# FAISS Vector Store Setup
def load_faiss_vectorstore(faiss_index_path: str = "./faiss_index") -> FAISS:
    """
    Load the FAISS vector store from the specified path.
    
    Args:
        faiss_index_path (str): Path to the FAISS index directory
        
    Returns:
        FAISS: Loaded FAISS vector store
    """
    if not Path(faiss_index_path).exists():
        raise FileNotFoundError(f"FAISS index not found at: {faiss_index_path}")
    
    # Use the same embeddings as in the PDF processing script
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    logger.info(f"Loading FAISS index from: {faiss_index_path}")
    vectorstore = FAISS.load_local(
        faiss_index_path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    logger.info("FAISS vector store loaded successfully")
    return vectorstore

# Initialize vector store
try:
    vectorstore = load_faiss_vectorstore("./faiss_index")  # Adjust path as needed
except FileNotFoundError as e:
    logger.error(f"Vector store initialization failed: {e}")
    raise

# Define Retriever Tool as a subclass of BaseTool for CrewAI compatibility
class RetrieverTool(BaseTool):
    name: str = "retrieve_docs"
    description: str = "Retrieve and evaluate documents from the knowledge base. Input: query string."
    
    def _run(self, query: str) -> str:
        """
        Retrieve relevant documents from the FAISS vector store.
        
        Args:
            query (str): Search query
            
        Returns:
            str: Formatted results from the vector store
        """
        logger.info(f"Retrieving documents for query: {query}")
        try:
            results = vectorstore.similarity_search(query, k=5)
            if not results:
                logger.warning("No relevant documents found.")
                return "No relevant documents found."
            
            formatted_results = "\n\n".join(
                f"Chunk {i+1} (from {doc.metadata.get('source', 'Unknown')}):\n{doc.page_content}"
                for i, doc in enumerate(results)
            )
            logger.info(f"Retrieved {len(results)} document chunks.")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error during document retrieval: {str(e)}")
            return f"Error retrieving documents: {str(e)}"
create a logging utility using loggure and change prints to info logs add time calcuations
consider using docling core over docling when applicable
consider checking for gpu and create a part for gpu and cpu usage 