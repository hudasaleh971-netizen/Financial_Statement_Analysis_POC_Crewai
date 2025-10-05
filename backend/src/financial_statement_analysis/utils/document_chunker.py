"""
Enhanced document chunker with Gemini-powered table descriptions.
File: enhanced_chunker.py
"""
from typing import List, Optional, Dict, Any
import hashlib
import os
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker

from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.types.doc.labels import DocItemLabel

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from src.financial_statement_analysis.utils.logging_config import setup_logger, log_execution_time
from src.financial_statement_analysis.utils.document_processor import DocumentProcessor

logger = setup_logger()


class TableDescriptionGenerator:
    """Generates AI descriptions for tables using Google Gemini."""
    
    def __init__(
        self, 
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0,
        api_key: Optional[str] = None
    ):
        """
        Initialize the table description generator with Gemini.
        Args:
            model_name: Gemini model to use (gemini-1.5-flash, gemini-1.5-pro)
            temperature: Temperature for generation (lower = more deterministic)
            api_key: Google API key (will default to GOOGLE_API_KEY env var)
        """
        logger.info(f"Initializing TableDescriptionGenerator with {model_name}")
        self.llm = None
        self.cache: Dict[str, str] = {}

        try:
            effective_api_key = api_key or os.getenv("GOOGLE_API_KEY")
            if not effective_api_key:
                logger.warning("GOOGLE_API_KEY not found. Table description generation is disabled.")
                return

            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                google_api_key=effective_api_key
            )
            
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert at analyzing financial tables. Create a concise description (under 150 words) of the table's structure and purpose, focusing on columns, rows, and key financial data types."""),
                ("user", "Analyze this financial table and provide a clear description:\n\n{table_markdown}")
            ])
            
            logger.info("TableDescriptionGenerator initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize TableDescriptionGenerator: {e}", exc_info=True)
            self.llm = None # Ensure LLM is disabled on error
        
    def _get_table_hash(self, table_markdown: str) -> str:
        """Generate a hash for the table to use as a cache key."""
        return hashlib.md5(table_markdown.encode()).hexdigest()
    
    def generate_description(self, table_markdown: str, use_cache: bool = True) -> str:
        """
        Generate an AI description for a table using Gemini.
        
        Args:
            table_markdown: The table in markdown format
            use_cache: Whether to use cached descriptions
            
        Returns:
            A description of the table or a fallback message.
        """
        if not self.llm:
            logger.warning("LLM is not initialized. Cannot generate table description.")
            return "Table description unavailable: Gemini API key not configured."

        if use_cache:
            table_hash = self._get_table_hash(table_markdown)
            if table_hash in self.cache:
                logger.debug("Using cached table description.")
                return self.cache[table_hash]
        
        try:
            logger.info("Generating new table description with Gemini...")
            logger.debug(f"Table markdown (first 200 chars): {table_markdown[:200]}")
            
            chain = self.prompt | self.llm
            
            logger.debug("Invoking Gemini API...")
            response = chain.invoke({"table_markdown": table_markdown})
            
            logger.debug(f"Response type: {type(response)}")
            logger.debug(f"Response content type: {type(response.content)}")
            
            description = response.content.strip()
            logger.info(f"✓ Successfully generated description ({len(description)} chars)")
            
            if use_cache:
                self.cache[table_hash] = description
                
            return description
            
        except Exception as e:
            error_msg = f"Failed to generate table description: {type(e).__name__}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            # Return error details in the message for debugging
            return f"Table description unavailable due to error: {type(e).__name__}: {str(e)}"


class EnhancedMDTableSerializer(MarkdownTableSerializer):
    """Enhanced markdown table serializer with AI-generated descriptions."""
    
    def __init__(self, description_generator: TableDescriptionGenerator):
        super().__init__()
        self.description_generator = description_generator
        logger.debug("EnhancedMDTableSerializer initialized.")
    
    def serialize(self, *, item, doc_serializer, doc, **kwargs):
        """Serialize table with an AI-generated description prepended."""
        try:
            # Call parent's serialize with keyword arguments
            base_result = super().serialize(
                item=item,
                doc_serializer=doc_serializer,
                doc=doc,
                **kwargs
            )
            
            # Get the markdown text from the result
            base_markdown = base_result.text if hasattr(base_result, 'text') else str(base_result)
            
            # Generate AI description
            description = self.description_generator.generate_description(base_markdown)
            
            # Create enhanced markdown
            enhanced_markdown = f"**Table Description:**\n{description}\n\n**Table Data:**\n{base_markdown}"
            
            # Return result with same structure as parent
            return create_ser_result(text=enhanced_markdown, span_source=item)
            
        except Exception as e:
            logger.error(f"Failed to serialize table with description: {e}", exc_info=True)
            # Fallback to parent's serialize
            return super().serialize(
                item=item,
                doc_serializer=doc_serializer,
                doc=doc,
                **kwargs
            )


class EnhancedDocumentChunker:
    """A document chunker that enriches tables with AI-generated descriptions."""
    
    def __init__(
        self,
        tokenizer,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0,
        max_tokens: int = 512,
        api_key: Optional[str] = None
    ):
        """
        Args:
            tokenizer: Tokenizer for chunking.
            model_name: Gemini model for table descriptions.
            temperature: Temperature for generation.
            max_tokens: Maximum tokens per chunk.
            api_key: Google API key.
        """
        logger.info("Initializing EnhancedDocumentChunker...")
        
        description_generator = TableDescriptionGenerator(
            model_name=model_name,
            temperature=temperature,
            api_key=api_key
        )
        table_serializer = EnhancedMDTableSerializer(description_generator)

        class CustomSerializerProvider(ChunkingSerializerProvider):
            def get_serializer(self, doc):
                return ChunkingDocSerializer(doc=doc, table_serializer=table_serializer)

        self.chunker = HybridChunker(
            tokenizer=tokenizer,
            serializer_provider=CustomSerializerProvider(),
            max_tokens=max_tokens,
        )
        
        logger.info(f"EnhancedDocumentChunker initialized (Max tokens: {max_tokens}).")

    @log_execution_time
    def chunk_document(self, docling_document) -> List:
        """
        Create chunks from a DoclingDocument with AI table descriptions.
        
        Args:
            docling_document: The DoclingDocument from DocumentProcessor.
            
        Returns:
            A list of chunks with enhanced table descriptions.
        """
        logger.info("=" * 70)
        logger.info("Starting document chunking...")
        
        try:
            chunks = list(self.chunker.chunk(dl_doc=docling_document))
            logger.info(f"✓ Created {len(chunks)} chunks from document.")
            
            table_chunks = [c for c in chunks if any(item.label == DocItemLabel.TABLE for item in c.meta.doc_items)]
            if table_chunks:
                logger.info(f"  └─ {len(table_chunks)} chunks contain tables with AI-generated descriptions.")
            
            return chunks
            
        except Exception as e:
            logger.error(f"✗ Chunking failed: {e}", exc_info=True)
            raise


# Example usage
if __name__ == "__main__":
    from transformers import AutoTokenizer
    from dotenv import load_dotenv

    # Load environment variables from .env file in the 'backend' directory
    # This should be done before any other code that needs the environment variables
    load_dotenv()
    
    try:
        # Step 1: Convert document
        logger.info("Step 1: Converting document...")
        processor = DocumentProcessor()
        source_file = "C:/Users/h.goian/Documents/Maseera/Finance/Financial_Statemets_Analysis/Financial_Statement_Analysis_POC_Crewai/backend/knowledge/HSBC-11.pdf"
        result = processor.convert_document(source_path=source_file)
        docling_doc = result.document
        
        # Step 2: Initialize tokenizer
        logger.info("\nStep 2: Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
        # Step 3: Create enhanced chunker with Gemini
        logger.info("\nStep 3: Creating enhanced chunks with Gemini...")
        enhanced_chunker = EnhancedDocumentChunker(
            tokenizer=tokenizer,
            model_name="gemini-2.0-flash",
            max_tokens=512,
        )
        chunks = enhanced_chunker.chunk_document(docling_doc)
        # Step 4: Save all chunks to a file for debugging
        import json
        from pathlib import Path

        # Use the absolute path requested by the user to avoid ambiguity on Windows
        output_dir = Path(r"C:/Users/h.goian/Documents/Maseera/Finance/Financial_Statemets_Analysis/Financial_Statement_Analysis_POC_Crewai/backend/src/financial_statement_analysis/output/processed_docs")
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("\n" + "="*70)
        logger.info("SAVING CHUNKS TO FILES:")
        logger.info("="*70)

        for i, chunk in enumerate(chunks):
            has_table = any(item.label == DocItemLabel.TABLE for item in chunk.meta.doc_items)
            chunk_type = "TABLE" if has_table else "TEXT"
            
            # Save chunk text
            chunk_file = output_dir / f"chunk_{i:03d}_{chunk_type}.txt"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                f.write(f"CHUNK {i} ({chunk_type})\n")
                f.write("="*70 + "\n\n")
                f.write(chunk.text)
            
            # Save metadata separately
            meta_file = output_dir / f"chunk_{i:03d}_{chunk_type}_metadata.txt"
            with open(meta_file, 'w', encoding='utf-8') as f:
                f.write(str(chunk.meta))
            
            logger.info(f"✓ Saved chunk {i} to {chunk_file}")

        logger.info(f"\n✓ All {len(chunks)} chunks saved to {output_dir}/")

    except Exception as e:
        logger.error(f"Example execution failed: {e}", exc_info=True)