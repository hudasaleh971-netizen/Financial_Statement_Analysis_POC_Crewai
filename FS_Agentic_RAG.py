from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from pathlib import Path
from loguru import logger
import os
from pathlib import Path

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

# Retriever Tool
def retrieve_docs(query: str) -> str:
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

retriever_tool = Tool(
    name="retrieve_docs",
    func=retrieve_docs,
    description="Retrieve and evaluate documents from the knowledge base. Input: query string."
)

# LLM and Agent Setup (without memory)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    api_key=GOOGLE_API_KEY
)

# Load prompt template (adjusted to remove chat_history)
prompt = PromptTemplate.from_file(
    "src/prompts/agent_prompt.txt",
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
)

tools = [retriever_tool]
agent = create_react_agent(llm, tools, prompt)

# Agent executor without memory
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True, 
    handle_parsing_errors=True,
    max_iterations=5,  # Add iteration limit for safety
    early_stopping_method="generate"
)

# Chat Function (simplified without memory)
def chat_with_agent(query: str) -> str:
    """
    Process a query through the agent without maintaining conversation memory.
    
    Args:
        query (str): User query
        
    Returns:
        str: Agent response
    """
    logger.info(f"Received query: {query}")
    try:
        response = agent_executor.invoke({"input": query})
        logger.info(f"Generated response: {response['output']}")
        return response["output"]
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return f"Sorry, I encountered an error processing your query: {str(e)}"



# Example usage
if __name__ == "__main__":
    # Test the system
    test_query = "What is this document about?"
    try:
        response = chat_with_agent(test_query)
        print(f"Query: {test_query}")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")