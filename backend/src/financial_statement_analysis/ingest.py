import argparse
import os
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer

from src.financial_statement_analysis.utils.document_processor import DocumentProcessor
from src.financial_statement_analysis.utils.document_chunker import EnhancedDocumentChunker
from src.financial_statement_analysis.utils.logging_config import setup_logger

logger = setup_logger()

def ingest_document(file_path: str, collection_name: str):
    """Processes, chunks, and ingests a document into a Qdrant vector store."""
    
    try:
        # Step 1: Convert document using DocumentProcessor
        logger.info(f"Step 1: Converting document: {file_path}")
        processor = DocumentProcessor()
        result = processor.convert_document(source_path=file_path)
        docling_doc = result.document
        logger.info("✓ Document converted successfully.")

        # Step 2: Initialize tokenizer and enhanced chunker
        logger.info("\nStep 2: Initializing tokenizer and chunker...")
        embedding_model_name = "ibm-granite/granite-embedding-english-r2"
        tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        
        enhanced_chunker = EnhancedDocumentChunker(
            tokenizer=tokenizer,
            max_tokens=1024
        )
        chunks = enhanced_chunker.chunk_document(docling_doc)
        logger.info("✓ Document chunked successfully.")

        # Step 3: Convert docling chunks to LangChain Documents
        logger.info("\nStep 3: Converting chunks to LangChain Documents...")
        langchain_docs = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk.text,
                metadata={
                    'source': os.path.basename(file_path),
                    'chunk_index': i,
                    **chunk.meta
                }
            )
            langchain_docs.append(doc)
        logger.info(f"✓ Converted {len(langchain_docs)} chunks.")

        # Step 4: Initialize embeddings
        logger.info("\nStep 4: Initializing embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": "cpu"} # Use 'cuda' if GPU is available
        )
        logger.info(f"✓ Initialized embedding model: {embedding_model_name}")

        # Step 5: Ingest documents into Qdrant
        logger.info("\nStep 5: Ingesting documents into Qdrant...")
        qdrant_path = "./qdrant_db"
        
        Qdrant.from_documents(
            langchain_docs,
            embeddings,
            path=qdrant_path,
            collection_name=collection_name,
            force_recreate=True, # Set to False to append to an existing collection
        )
        logger.info(f"✓ Successfully ingested documents into Qdrant collection: '{collection_name}'")

    except Exception as e:
        logger.error(f"An error occurred during the ingestion process: {e}", exc_info=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ingest a financial document into the Qdrant vector store.")
    parser.add_argument("file_path", type=str, help="The absolute path to the PDF file to ingest.")
    parser.add_argument("--collection", type=str, default="financial_docs", help="The name of the Qdrant collection.")
    args = parser.parse_args()

    ingest_document(args.file_path, args.collection)
