import os
import shutil
import tempfile
import asyncio
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer

from src.financial_statement_analysis.utils.document_processor import DocumentProcessor
from src.financial_statement_analysis.utils.document_chunker import EnhancedDocumentChunker
from src.financial_statement_analysis.utils.logging_config import setup_logger
from src.financial_statement_analysis.utils.vectorstore_save import save_chunks_to_qdrant
from src.financial_statement_analysis.crew import crew

logger = setup_logger()

app = FastAPI(
    title="Financial Statement Analysis API",
    description="Async API to process financial documents (PDF, Excel, CSV) and run Crew pipeline",
    version="1.0.1"
)

SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".xlsx", ".xls"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/process-document")
async def process_document(file: UploadFile):
    """
    Upload a document (PDF, Excel, CSV), process it, chunk, save to Qdrant, and run the Crew pipeline.
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: PDF, Excel (.xlsx/.xls), CSV."
        )

    # Step 1: Save uploaded file temporarily
    try:
        temp_dir = tempfile.mkdtemp()
        source_path = os.path.join(temp_dir, file.filename)
        with open(source_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        logger.info(f"Uploaded file saved to {source_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {e}")

    try:
        # Step 2: Convert document
        logger.info("Step 1: Converting document...")
        processor = DocumentProcessor()
        result = processor.convert_document(source_path=source_path)
        docling_doc = result.document
        logger.info("✓ Document conversion completed.")

        # Step 3: Initialize tokenizer
        logger.info("Step 2: Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-embedding-english-r2")

        # Step 4: Chunk document with Gemini
        logger.info("Step 3: Creating enhanced chunks with Gemini...")
        enhanced_chunker = EnhancedDocumentChunker(
            tokenizer=tokenizer,
            model_name="gemini-2.0-flash",
            max_tokens=1024,
        )
        chunks = enhanced_chunker.chunk_document(docling_doc, source_path)
        logger.info(f"✓ Ready for RAG pipeline with {len(chunks)} enhanced chunks.")

        # Step 5: Save chunks to local Qdrant
        logger.info("Step 4: Saving chunks to local Qdrant...")
        save_chunks_to_qdrant(chunks, source_path)

        # Step 6: Kick off Crew pipeline
        logger.info("Step 5: Running Crew pipeline...")
        inputs = {"filename": os.path.basename(source_path)}

        # Crew kickoff may be blocking; run in a thread to keep FastAPI async
        result = await asyncio.to_thread(crew.kickoff, inputs=inputs)

        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "crew_result": result
        })

    except Exception as e:
        logger.exception("Processing failed.")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
