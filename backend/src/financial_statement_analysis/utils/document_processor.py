"""
Document processing utilities using Docling for PDF and Excel conversion with RapidOCR fallback.
"""
import json
from pathlib import Path
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
from docling.document_converter import DocumentConverter, ConversionResult, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline
from src.financial_statement_analysis.utils.logging_config import setup_logger, log_execution_time

logger = setup_logger()


class DocumentProcessor:
    """Document processor for PDF and Excel files with OCR fallback."""
    
    def __init__(self):
        logger.info("Initializing DocumentProcessor...")
        
        # Primary VLM pipeline for PDF
        vlm_options = VlmPipelineOptions(
            model_id="ibm-granite/granite-docling-258M",
            do_table_structure=True,
            table_mode=TableFormerMode.ACCURATE,
            generate_page_images=True,
            generate_picture_images=True,
            max_pages=20,
        )

        if torch.cuda.is_available():
            vlm_options.accelerator_options.device = AcceleratorDevice.CUDA
            vlm_options.accelerator_options.cuda_use_flash_attention2 = False
            logger.info("Using CUDA for VLM pipeline")
        else:
            vlm_options.accelerator_options.device = AcceleratorDevice.CPU
            vlm_options.accelerator_options.num_threads = os.cpu_count() // 2
            logger.info(f"Using CPU with {os.cpu_count() // 2} threads for VLM pipeline")
            
        pdf_options_vlm = PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=vlm_options,
            backend=PyPdfiumDocumentBackend, 
        )

        # Secondary OCR pipeline for PDF fallback (RapidOCR)
        ocr_options = PdfPipelineOptions(
            do_ocr=True,
            ocr_options=RapidOcrOptions(lang=["en"]),
            do_table_structure=True,
            table_structure_options=TableStructureOptions(
                do_cell_matching=True,
                table_mode=TableFormerMode.ACCURATE,
            ),
            force_full_page_ocr=True,
        )
        
        if torch.cuda.is_available():
            ocr_options.accelerator_options.device = AcceleratorDevice.CUDA
        else:
            ocr_options.accelerator_options.device = AcceleratorDevice.CPU
            ocr_options.accelerator_options.num_threads = os.cpu_count() // 2

        pdf_options_ocr = PdfFormatOption(
            pipeline_options=ocr_options,
            backend=PyPdfiumDocumentBackend,
        )

        self.primary_converter = DocumentConverter(
            format_options={InputFormat.PDF: pdf_options_vlm}
        )
        self.fallback_converter = DocumentConverter(
            format_options={InputFormat.PDF: pdf_options_ocr}
        )
        
        logger.info("DocumentProcessor initialized successfully")

    @log_execution_time
    def convert_document(
        self,
        source_path: str,
        output_dir: str = "output/processed_docs/",
        save_as_markdown: bool = False,
        save_as_json: bool = False,
        save_as_html: bool = False,
    ) -> ConversionResult:
        """
        Convert document (PDF or XLSX) to DoclingDocument format.
        
        Args:
            source_path: Path to the source document
            output_dir: Directory to save outputs
            save_as_markdown: Save as markdown file
            save_as_json: Save as JSON file
            save_as_html: Save as HTML file
            
        Returns:
            ConversionResult containing the DoclingDocument (access via result.document)
        """
        source = Path(source_path)
        if not source.exists():
            logger.error(f"Document file not found: {source_path}")
            raise FileNotFoundError(f"Document file not found: {source_path}")

        ext = source.suffix.lower()
        if ext not in ['.pdf', '.xlsx']:
            logger.error(f"Unsupported file extension: {ext}")
            raise ValueError(f"Unsupported file extension: {ext}. Supported: .pdf, .xlsx")

        icon = "📄" if ext == '.pdf' else "📊"
        logger.info("=" * 70)
        logger.info(f"{icon} Converting Document: {source.name}")
        logger.info("=" * 70)

        try:
            if ext == '.xlsx':
                result = self.primary_converter.convert(source=str(source))
                logger.info("✓ Excel Conversion Complete (Native)")
            else:  # PDF
                try:
                    result = self.primary_converter.convert(str(source))
                    confidence = result.confidence
                    logger.info("✓ Primary VLM Conversion Complete - Quality Metrics:")
                    logger.info(f"   Overall Quality: {confidence.mean_grade}")
                    logger.info(f"   Lowest Area Quality: {confidence.low_grade}")
                except Exception as vlm_error:
                    logger.warning(f"VLM conversion failed: {vlm_error}")
                    logger.info("Falling back to RapidOCR for better number/table extraction")
                    result = self.fallback_converter.convert(str(source))
                    logger.info("✓ RapidOCR Fallback Complete")

            # Save outputs if requested
            if save_as_markdown or save_as_json or save_as_html:
                self._save_outputs(result, source, output_dir, 
                                 save_as_markdown, save_as_json, save_as_html)

            logger.info(f"✓ Document conversion successful: {source.name}")
            return result

        except Exception as e:
            logger.error(f"✗ Document conversion failed: {str(e)}", exc_info=True)
            raise

    def _save_outputs(self, result: ConversionResult, source: Path, 
                     output_dir: str, save_md: bool, save_json: bool, save_html: bool):
        """Helper method to save conversion outputs."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            file_stem = source.stem

            if save_md:
                md_path = output_path / f"{file_stem}.md"
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(result.document.export_to_markdown())
                logger.info(f"Saved Markdown to: {md_path}")

            if save_json:
                json_path = output_path / f"{file_stem}.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    f.write(result.document.model_dump_json(indent=4))
                logger.info(f"Saved JSON to: {json_path}")

            if save_html:
                html_path = output_path / f"{file_stem}.html"
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(result.document.export_to_html())
                logger.info(f"Saved HTML to: {html_path}")
                
        except Exception as e:
            logger.error(f"Failed to save outputs: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    processor = DocumentProcessor()
    try:
        source_file = "C:/Users/h.goian/Documents/Maseera/Finance/Financial_Statemets_Analysis/Financial_Statement_Analysis_POC_Crewai/backend/knowledge/HSBC-11.pdf"
        result = processor.convert_document(
            source_path=source_file,
            save_as_markdown=True,
            save_as_json=True,
            save_as_html=True
        )
        
        # Access the DoclingDocument
        docling_doc = result.document
        print(docling_doc)
        logger.info(f"Document has {len(docling_doc.texts)} text elements")
        
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}", exc_info=True)