# Step 1 - Document Conversion
import time
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
    TableFormerMode,
    AcceleratorDevice,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
import torch
from docling.chunking import HybridChunker
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer

vlm_options = VlmPipelineOptions(
    model_id="ibm-granite/granite-docling-258M",
    do_table_structure=True,
    table_mode=TableFormerMode.ACCURATE,
    generate_page_images=True,
    generate_picture_images=True,
    max_pages=7,
)

# Enable CUDA acceleration if available
if torch.cuda.is_available():
    vlm_options.accelerator_options.device = AcceleratorDevice.CUDA
    vlm_options.accelerator_options.cuda_use_flash_attention2 = True  # 2-4x speedup!

pdf_options = PdfFormatOption(
    pipeline_cls=VlmPipeline,
    pipeline_options=vlm_options,
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: pdf_options,
    }
)
source = "C:/Users/h.goian/Documents/Maseera/Finance/Financial_Statement_Analysis_POC_Crewai/knowledge/ADIB.pdf"
result = converter.convert(source=source)
confidence = result.confidence

print("Overall quality:", confidence.mean_grade)
print("Lowest area quality:", confidence.low_grade)
print("Layout score:", confidence.layout_score)
print("OCR score:", confidence.ocr_score)
print("Parse score:", confidence.parse_score)
print("Table score:", confidence.table_score) 

for page in confidence.pages:
    print(f"Page {page['page_number']} grade:", page['mean_grade'])
# Step 2 - Chunking and Embedding
embeddings_model_path = "ibm-granite/granite-embedding-30m-english"
embeddings_model = HuggingFaceEmbeddings(
    model_name=embeddings_model_path,
)

embeddings_tokenizer = AutoTokenizer.from_pretrained(embeddings_model_path)

chunker = HybridChunker(
    tokenizer=embeddings_tokenizer,
    merge_peers=True,
)


# ✅ FIXED: Pass result (Document object) instead of converter
chunk_iter = chunker.chunk(dl_doc=result)

# ✅ FIXED: Convert to list once to avoid iterator exhaustion
chunks = list(chunk_iter)

# Now iterate over the list instead of the exhausted iterator
for i, chunk in enumerate(chunks):
    print(f"=== {i} ===")
    print(f"chunk.text:\n{chunk.text[:300]}…")

    enriched_text = chunker.contextualize(chunk=chunk)
    print(f"chunker.contextualize(chunk):\n{enriched_text[:300]}…")
    print()