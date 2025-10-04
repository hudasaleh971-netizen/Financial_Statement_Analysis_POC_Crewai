from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
from pathlib import Path
from loguru import logger
import os
from pathlib import Path
# Add crewai_tools import for BaseTool
from crewai.tools import BaseTool

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LOG_DIR = Path("backend/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def setup_logger(log_file: str = "FS_RAG.log") -> None:
    logger.remove()  # Clear default handler
    logger.add(
        LOG_DIR / log_file,
        rotation="10 MB",
        level="INFO",
        enqueue=True  # Better for multi-threaded or async apps
    )
    logger.add(lambda msg: print(msg, end=""), level="INFO")


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
