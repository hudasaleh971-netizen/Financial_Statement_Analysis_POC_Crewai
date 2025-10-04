
@dataclass
class DocumentChunk:
    """Represents a processed document chunk with metadata."""
    content: str
    index: int
    start_char: int
    end_char: int
    token_count: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def __len__(self) -> int:
        """Return content length."""
        return len(self.content)


@dataclass
class ConfidenceMetrics:
    """Quality metrics for document conversion."""
    mean_grade: float
    low_grade: float
    layout_score: float
    ocr_score: float
    parse_score: float
    table_score: float
    page_metrics: List[Dict[str, Any]]
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def is_high_quality(self, threshold: float = 0.8) -> bool:
        """Check if document quality is above threshold."""
        return self.mean_grade >= threshold


@dataclass
class ProcessingResult:
    """Result of document processing."""
    document: DoclingDocument
    chunks: List[DocumentChunk]
    processing_time: float
    confidence_metrics: ConfidenceMetrics
    source_path: str
    total_pages: int
    total_tokens: int
    
    def get_summary(self) -> dict:
        """Get processing summary."""
        return {
            "source": Path(self.source_path).name,
            "pages": self.total_pages,
            "chunks": len(self.chunks),
            "total_tokens": self.total_tokens,
            "avg_tokens_per_chunk": self.total_tokens / len(self.chunks) if self.chunks else 0,
            "processing_time_seconds": round(self.processing_time, 2),
            "quality_score": self.confidence_metrics.mean_grade,
        }

   def chunk_document(
        self,
        doc: DoclingDocument,
        base_metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Chunk document into smaller pieces with hierarchical context.
        
        Args:
            doc: DoclingDocument to chunk
            base_metadata: Base metadata to include in all chunks
            
        Returns:
            List of DocumentChunk objects
        """
        logger.info("=" * 70)
        logger.info("📝 Starting Document Chunking")
        logger.info("=" * 70)
        
        if base_metadata is None:
            base_metadata = {}
        
        # Get chunker from model manager
        chunker = self.model_manager.chunker
        
        try:
            # Generate chunks using Docling Core
            logger.info("⏳ Generating chunks...")
            chunk_iter = chunker.chunk(dl_doc=doc)
            chunks_list = list(chunk_iter)
            
            logger.info(f"✓ Generated {len(chunks_list)} raw chunks")
            
            # Process chunks with contextualization
            document_chunks = []
            current_pos = 0
            total_tokens = 0
            
            for i, chunk in enumerate(chunks_list):
                # Get contextualized text (includes heading hierarchy)
                contextualized_text = chunker.serialize(chunk=chunk)
                
                # Count actual tokens
                token_count = self.model_manager.count_tokens(contextualized_text)
                total_tokens += token_count
                
                # Create chunk metadata
                chunk_metadata = {
                    **base_metadata,
                    "chunk_index": i,
                    "total_chunks": len(chunks_list),
                    "token_count": token_count,
                    "has_context": True,
                    "processor_version": "1.0",
                }
                
                # Estimate character positions
                start_char = current_pos
                end_char = start_char + len(contextualized_text)
                
                document_chunks.append(
                    DocumentChunk(
                        content=contextualized_text.strip(),
                        index=i,
                        start_char=start_char,
                        end_char=end_char,
                        metadata=chunk_metadata,
                        token_count=token_count,
                    )
                )
                
                current_pos = end_char
                
                # Log sample chunks (first 3 and last)
                if i < 3 or i == len(chunks_list) - 1:
                    preview = contextualized_text[:150].replace('\n', ' ')
                    logger.debug(
                        f"   Chunk {i:3d} ({token_count:4d} tokens): {preview}..."
                    )
            
            # Log statistics
            avg_tokens = total_tokens / len(document_chunks) if document_chunks else 0
            min_tokens = min((c.token_count for c in document_chunks), default=0)
            max_tokens = max((c.token_count for c in document_chunks), default=0)
            
            logger.info("=" * 70)
            logger.info("✓ Chunking Complete - Statistics:")
            logger.info(f"   Total Chunks: {len(document_chunks)}")
            logger.info(f"   Total Tokens: {total_tokens:,}")
            logger.info(f"   Average Tokens/Chunk: {avg_tokens:.1f}")
            logger.info(f"   Token Range: {min_tokens} - {max_tokens}")
            logger.info("=" * 70)
            
            return document_chunks
            
        except Exception as e:
            logger.error(f"✗ Chunking failed: {e}")
            raise
    