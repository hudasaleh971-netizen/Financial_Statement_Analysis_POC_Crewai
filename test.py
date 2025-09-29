from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
from pathlib import Path
from loguru import logger
import os
from pathlib import Path

# Add CrewAI imports
from crewai import Agent, Task, Crew, Process, LLM

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

# Instantiate the tool
retriever_tool = RetrieverTool()

# LLM Setup (unchanged)
llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
    api_key=GOOGLE_API_KEY
)

# Define CrewAI Agent (using the instantiated tool)
research_agent = Agent(
    role="Document Researcher",
    goal="Retrieve and analyze relevant documents from the knowledge base to answer queries accurately and concisely.",
    backstory="You are an expert AI researcher specialized in extracting and synthesizing information from document databases using retrieval tools.",
    tools=[retriever_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False,  # Best practice: Disable delegation for single-agent setups
    max_iter=5,  # Best practice: Limit iterations to prevent infinite loops
    max_rpm=None  # Optional: Can set rate limits if needed
)

# Chat Function (refactored for CrewAI)
def chat_with_agent(query: str) -> str:
    """
    Process a query through the CrewAI agent without maintaining conversation memory.
    
    Args:
        query (str): User query
        
    Returns:
        str: Agent response
    """
    logger.info(f"Received query: {query}")
    try:
        # Define task dynamically with the query
        research_task = Task(
            description=f"""Use the retrieve_docs tool to fetch relevant documents from the knowledge base.
            Then, analyze the retrieved information to provide a clear and accurate answer to the query.
            
            Query: {query}
            
            If no relevant documents are found, state that clearly.
            Keep the response concise and directly based on the retrieved content.""",
            expected_output="A clear, concise answer to the query based on retrieved documents, or a note if no information is found.",
            agent=research_agent,
            tools=[retriever_tool]  # Best practice: Explicitly assign tools to task
        )
        
        # Create crew with sequential process (best for single-task flows)
        crew = Crew(
            agents=[research_agent],
            tasks=[research_task],
            verbose=True,  # Matches original verbose=True
            process=Process.sequential,  # Best practice: Explicit process for clarity
            manager_llm=llm  # Optional: Use same LLM for any management if needed
        )
        
        # Run the crew
        response = crew.kickoff()
        logger.info(f"Generated response: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return f"Sorry, I encountered an error processing your query: {str(e)}"


# Example usage
if __name__ == "__main__":
    # Test the system
    test_query = "what is the total equity in 2024 & 2023?"
    try:
        response = chat_with_agent(test_query)
        print(f"Query: {test_query}")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")