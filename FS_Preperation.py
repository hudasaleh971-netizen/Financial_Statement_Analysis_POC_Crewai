from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pathlib import Path
import logging
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, EasyOcrOptions

# Set up logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

def load_and_store_pdf(file_path: str, faiss_index_path: str = "./faiss_index") -> FAISS:
    """
    Load a PDF with OCR support, chunk it, and store in FAISS vector store.
    
    Args:
        file_path (str): Path to the PDF file
        faiss_index_path (str): Directory to save the FAISS index
    
    Returns:
        FAISS: The FAISS vector store with document chunks
    """
    # Verify PDF exists
    pdf_path = Path(file_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    
    pipeline_options = PdfPipelineOptions(
        do_table_structure=True,
        do_ocr=True,
        ocr_options=EasyOcrOptions(
            confidence_threshold=0.3,
            force_full_page_ocr=True,
            bitmap_area_threshold=0.01,
            lang=['en'],
            download_enabled=True,
            use_gpu=False
        )
    )
    pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_options.table_structure_options.do_cell_matching = True

    # ✅ Correct: use PdfFormatOption, not PdfPipelineOptions directly
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    
    try:
        # Convert PDF to document
        _log.info("Converting PDF...")
        result = converter.convert(pdf_path)
        
        # Initialize chunker
        chunker = HybridChunker(
            tokenizer="sentence-transformers/all-MiniLM-L6-v2",
            max_chunk_size=512,
            overlap=50
        )
        
        # Chunk the document
        _log.info("Chunking document...")
        chunks = chunker.chunk(result.document)
        
        if not chunks:
            raise ValueError("No content extracted from PDF")
        
        # Convert to LangChain documents
        langchain_docs = []
        for i, chunk in enumerate(chunks):
            # Create clean metadata
            metadata = {
                "source": str(pdf_path),
                "chunk_id": i,
                "page": getattr(chunk, 'page_no', 0) if hasattr(chunk, 'page_no') else 0
            }
            
            # Get text content
            content = chunk.text if hasattr(chunk, 'text') else str(chunk)
            
            langchain_docs.append(Document(
                page_content=content,
                metadata=metadata
            ))
        
        _log.info(f"Created {len(langchain_docs)} document chunks")
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        
        # Create and save FAISS vector store
        _log.info("Creating FAISS vector store...")
        vector_store = FAISS.from_documents(langchain_docs, embeddings)
        vector_store.save_local(faiss_index_path)
        
        _log.info(f"FAISS index saved to {faiss_index_path}")
        return vector_store
        
    except Exception as e:
        _log.error(f"Error processing PDF: {str(e)}")
        raise

def load_existing_faiss(faiss_index_path: str = "./faiss_index") -> FAISS:
    """Load an existing FAISS index."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    return FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)

# Example usage
if __name__ == "__main__":
    pdf_file = "Example/ADIB.pdf"
    
    try:
        # Process and store PDF
        vector_store = load_and_store_pdf(pdf_file)
        print("✓ Successfully processed and stored PDF in FAISS")
        
        # Test retrieval
        query = "What is this document about?"
        results = vector_store.similarity_search(query, k=3)
        
        print(f"\nFound {len(results)} relevant chunks:")
        for i, doc in enumerate(results):
            print(f"\nChunk {i+1}:")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}")
            
    except Exception as e:
        print(f"❌ Error: {e}")