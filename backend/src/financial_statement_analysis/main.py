import pprint
import json
import argparse
import os
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer

from src.financial_statement_analysis.utils.document_processor import DocumentProcessor
from src.financial_statement_analysis.utils.document_chunker import EnhancedDocumentChunker
from src.financial_statement_analysis.utils.logging_config import setup_logger
from src.financial_statement_analysis.utils.vectorstore_save import save_chunks_to_qdrant
from src.financial_statement_analysis.crew import crew

# this should be exposed as a fast api endpoint for the frontend to call
logger = setup_logger()

# Step 1: Convert document
logger.info("Step 1: Converting document...")
processor = DocumentProcessor()
source_file = "C:/Users/h.goian/Documents/Maseera/Finance/Financial_Statemets_Analysis/Financial_Statement_Analysis_POC_Crewai/backend/knowledge/HSBC-11.pdf"
result = processor.convert_document(source_path=source_file)
docling_doc = result.document

# Step 2: Initialize tokenizer
logger.info("\nStep 2: Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-embedding-english-r2")

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
filename = os.path.basename(source_file)

# The inputs dictionary will be used to fill the {filename} placeholder
inputs = {'filename': filename}

# Kick off the crew with the provided inputs
result = crew.kickoff(inputs=inputs)