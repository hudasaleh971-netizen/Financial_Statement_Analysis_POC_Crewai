# vectorstore_load.py
# This file contains the RetrieverTool for crewai, which retrieves relevant chunks from the local Qdrant vector store.
# It supports filtering by filename in the metadata.

import os
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels
from src.financial_statement_analysis.utils.logging_config import setup_logger
from crewai.tools import BaseTool
from typing import Optional, List, Dict

logger = setup_logger()

class RetrieverTool(BaseTool):
    name: str = "Document Retriever"
    description: str = """Retrieves relevant document chunks from the vector store based on a query.
    Supports filtering by filename to retrieve only chunks from a specific document.
    Useful for querying financial statements or specific PDFs.
    
    Input should include:
    - query: The search query string.
    - filename (optional): The exact filename to filter results (e.g., 'HSBC-11.pdf'). If not provided, searches all documents.
    
    Returns a list of relevant text chunks with their metadata.
    """

    def _run(self, query: str, filename: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Run the retriever tool.
        
        Args:
            query: The search query.
            filename: Optional filename to filter by.
        
        Returns:
            List of dicts, each with 'text' and 'metadata'.
        """
        try:
            # Step 1: Initialize the embedding model (same as used for ingestion)
            embeddings = HuggingFaceEmbeddings(
                model_name="ibm-granite/granite-embedding-english-r2",
                model_kwargs={"device": "cpu"}
            )
            logger.info("Initialized Granite embedding model for retrieval.")
            
            # Step 2: Initialize local Qdrant client
            qdrant_path = "./qdrant_db"
            client = QdrantClient(path=qdrant_path)
            
            collection_name = "financial_docs"
            
            # Step 3: Load the vector store
            vector_store = Qdrant(
                client=client,
                collection_name=collection_name,
                embeddings=embeddings
            )
            logger.info(f"Loaded Qdrant vector store: {collection_name}")
            
            # Step 4: Build metadata filter if filename is provided
            metadata_filter = None
            if filename:
                metadata_filter = qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="metadata.filename",
                            match=qmodels.MatchValue(value=filename)
                        )
                    ]
                )
                logger.info(f"Applying filter for filename: {filename}")
            
            # Step 5: Perform similarity search
            results = vector_store.similarity_search_with_score(
                query=query,
                k=5,  # Retrieve top 5 results; adjust as needed
                filter=metadata_filter
            )
            
            # Step 6: Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": score
                })
            
            logger.info(f"Retrieved {len(formatted_results)} relevant chunks.")
            return formatted_results
        
        except Exception as e:
            logger.error(f"Retrieval failed: {e}", exc_info=True)
            return [{"error": str(e)}]

# Example usage (for testing the tool standalone)
if __name__ == "__main__":
    tool = RetrieverTool()
    # Test with a query and optional filename filter
    results = tool._run(
        query="What is the Assets value in 2024?",
        filename="HSBC-11.pdf"  # Comment out or set to None to search all
    )
    for result in results:
        print(result)
