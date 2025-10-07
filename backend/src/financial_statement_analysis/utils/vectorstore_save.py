# Separate file: vector_store.py
# This file contains the function to save chunks to local Qdrant using Langchain for simplicity

import os
import uuid
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient, models
from src.financial_statement_analysis.utils.logging_config import setup_logger
from typing import List, Optional, Dict, Any
from src.financial_statement_analysis.utils.document_processor import DocumentProcessor
from src.financial_statement_analysis.utils.document_chunker import EnhancedDocumentChunker
logger = setup_logger()

def save_chunks_to_qdrant(chunks: List, source_file: str):
    """
    Save the document chunks to a local Qdrant vector database using Langchain integration.
    
    Args:
        chunks: List of chunk objects with .text and .meta attributes.
        source_file: The original source file path (used for deriving IDs or collection name if needed).
    """
    try:
        # Step 1: Initialize the embedding model using HuggingFaceEmbeddings
        # This wraps SentenceTransformers for the Granite model
        embeddings = HuggingFaceEmbeddings(
            model_name="ibm-granite/granite-embedding-english-r2",
            model_kwargs={"device": "cpu"}  # Use CPU for beginner simplicity; change to 'cuda' if GPU available
        )
        logger.info("Initialized Granite embedding model.")
        
        # Step 2: Prepare texts and metadatas
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.meta for chunk in chunks]
        
        # Generate unique IDs based on filename and chunk index
        filename = os.path.basename(source_file)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{filename}_{i}")) for i in range(len(chunks))]
        
        # Step 3: Initialize local Qdrant client
        qdrant_path = "./qdrant_db"  # Directory for local Qdrant storage
        client = QdrantClient(path=qdrant_path)
        
        collection_name = "financial_docs2"
        
        # Step 3.5: Check if collection exists, create if not (with correct dimension)
        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=768,  # Embedding dimension for ibm-granite/granite-embedding-english-r2
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created new Qdrant collection: {collection_name}")
        else:
            logger.info(f"Using existing Qdrant collection: {collection_name}")
        
        # Step 4: Create or use Qdrant vector store with Langchain
        # Langchain will now use the existing or newly created collection
        vector_store = Qdrant(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings
        )
        logger.info(f"Connected to Qdrant collection: {collection_name}")
        
        # Step 5: Add texts with metadata and IDs
        vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"✓ Successfully saved {len(chunks)} chunks to Qdrant collection '{collection_name}'.")
        
    except Exception as e:
        logger.error(f"Failed to save chunks to Qdrant: {e}", exc_info=True)
        raise

# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer


    # Load environment variables from .env file in the 'backend' directory
    # This should be done before any other code that needs the environment variables
    
    try:
        # Step 1: Convert document
        logger.info("Step 1: Converting document...")
        processor = DocumentProcessor()
        source_file = "C:/Users/h.goian/Documents/Maseera/Finance/Financial_Statemets_Analysis/Financial_Statement_Analysis_POC_Crewai/backend/knowledge/HSBC-11.pdf"
        result = processor.convert_document(source_path=source_file)
        docling_doc = result.document
        
        # Step 2: Initialize tokenizer
        logger.info("\nStep 2: Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-embedding-english-r2")
        #sentence-transformers/all-MiniLM-L6-v2	
        # Step 3: Create enhanced chunker with Gemini
        logger.info("\nStep 3: Creating enhanced chunks with Gemini...")
        enhanced_chunker = EnhancedDocumentChunker(
            tokenizer=tokenizer,
            model_name="gemini-2.0-flash",
            max_tokens=1024,
        )
        chunks = enhanced_chunker.chunk_document(docling_doc, source_file)  # Pass source_file here
          
        logger.info(f"\n✓ Ready for RAG pipeline with {len(chunks)} enhanced chunks.")
        
        # Save all chunks and metadata to a .md file for traceability
        output_dir = "C:/Users/h.goian/Documents/Maseera/Finance/Financial_Statemets_Analysis/Financial_Statement_Analysis_POC_Crewai/backend/src/financial_statement_analysis/output/processed_docs/"
        
        # Step 5: Save chunks to local Qdrant vector database
        logger.info("\nStep 5: Saving chunks to local Qdrant...")
        save_chunks_to_qdrant(chunks, source_file)       

    except Exception as e:
        logger.error(f"Example execution failed: {e}", exc_info=True)

# i need to save some metadata only
# like filename, headers,  type, prov
# |', 'metadata': {'filename': 'HSBC-11.pdf', 'headers': ['at 31 December'], 'doc_items': [{'type': 'table', 'prov': [{'page': 1, 'location': {'l': 22.61, 't': 114.51200000000001, 'r': 534.3100000000001, 'b': 506.88399999999996, 'coord_origin': 'TOPLEFT'}}]}]}}
