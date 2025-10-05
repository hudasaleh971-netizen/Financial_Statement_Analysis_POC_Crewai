"""
Enhanced DocumentConverter compatible with LangChain's DoclingLoader.
Supports VLM with RapidOCR fallback for optimal PDF processing.
"""
import torch
import os
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend  
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
    PdfPipelineOptions,
    TableFormerMode,
    AcceleratorDevice,
    RapidOcrOptions,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline


def create_enhanced_converter(
    use_vlm: bool = True,
    use_gpu: bool = None,
    max_pages: int = 20,
    table_mode: TableFormerMode = TableFormerMode.ACCURATE,
    ocr_lang: list = ["en"],
) -> DocumentConverter:
    """
    Create an enhanced DocumentConverter with VLM + RapidOCR fallback support.
    
    Args:
        use_vlm: If True, use VLM pipeline; if False, use RapidOCR
        use_gpu: Force GPU/CPU usage. None = auto-detect
        max_pages: Maximum pages to process
        table_mode: Table extraction mode (FAST or ACCURATE)
        ocr_lang: OCR languages for RapidOCR
    
    Returns:
        DocumentConverter instance ready for DoclingLoader
    
    Example:
        >>> from langchain_docling import DoclingLoader
        >>> converter = create_enhanced_converter()
        >>> loader = DoclingLoader(
        ...     file_path="document.pdf",
        ...     converter=converter
        ... )
        >>> docs = loader.load()
    """
    # Auto-detect device
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()
    
    device = AcceleratorDevice.CUDA if use_gpu else AcceleratorDevice.CPU
    
    if use_vlm:
        # VLM Pipeline (Primary - best for mixed content)
        vlm_options = VlmPipelineOptions(
            model_id="ibm-granite/granite-docling-258M",
            do_table_structure=True,
            table_mode=table_mode,
            generate_page_images=True,
            generate_picture_images=True,
            max_pages=max_pages,
        )
        vlm_options.accelerator_options.device = device
        
        if device == AcceleratorDevice.CUDA:
            vlm_options.accelerator_options.cuda_use_flash_attention2 = False
        else:
            vlm_options.accelerator_options.num_threads = os.cpu_count() // 2
        
        pdf_options = PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=vlm_options,
            backend=PyPdfiumDocumentBackend,
        )
    else:
        # RapidOCR Pipeline (Fallback - best for scanned docs with tables/numbers)
        ocr_options = PdfPipelineOptions(
            do_ocr=True,
            ocr_options=RapidOcrOptions(lang=ocr_lang),
            do_table_structure=True,
            table_structure_options=TableStructureOptions(
                do_cell_matching=True,
                table_mode=table_mode,
            ),
            force_full_page_ocr=True,
        )
        ocr_options.accelerator_options.device = device
        
        if device == AcceleratorDevice.CPU:
            ocr_options.accelerator_options.num_threads = os.cpu_count() // 2
        
        pdf_options = PdfFormatOption(
            pipeline_options=ocr_options,
            backend=PyPdfiumDocumentBackend,
        )
    
    return DocumentConverter(
        format_options={InputFormat.PDF: pdf_options}
    )


# Quick presets for common use cases
def create_vlm_converter(**kwargs) -> DocumentConverter:
    """VLM converter (best for native PDFs with mixed content)"""
    return create_enhanced_converter(use_vlm=True, **kwargs)


def create_ocr_converter(**kwargs) -> DocumentConverter:
    """RapidOCR converter (best for scanned docs, tables, numbers)"""
    return create_enhanced_converter(use_vlm=False, **kwargs)


def create_fast_converter(**kwargs) -> DocumentConverter:
    """Fast VLM converter (FAST table mode for speed)"""
    return create_enhanced_converter(
        use_vlm=True,
        table_mode=TableFormerMode.FAST,
        **kwargs
    )


# Usage Examples
if __name__ == "__main__":
    # Example 1: Basic usage with DoclingLoader
    from langchain_docling import DoclingLoader
    source_file = "C:/Users/h.goian/Documents/Maseera/Finance/Financial_Statemets_Analysis/Financial_Statement_Analysis_POC_Crewai/backend/knowledge/HSBC-11.pdf"

    converter = create_enhanced_converter()
    loader = DoclingLoader(
        file_path=source_file,
        converter=converter
    )
    docs = loader.load()
    print(docs[0])
    