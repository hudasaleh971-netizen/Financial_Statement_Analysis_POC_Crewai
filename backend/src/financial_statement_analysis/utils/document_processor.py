"""
Document processing utilities using Docling for PDF conversion.
"""
import json
from pathlib import Path
import torch
import os
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend  
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
    TableFormerMode,
    AcceleratorDevice,
)
from docling.document_converter import PdfFormatOption
from docling.document_converter import DocumentConverter, ConversionResult
from docling.pipeline.vlm_pipeline import VlmPipeline

from src.financial_statement_analysis.utils.logging_config import setup_logger, log_execution_time

logger = setup_logger()


class DocumentProcessor:
    def __init__(self):
        vlm_options = VlmPipelineOptions(
            model_id="ibm-granite/granite-docling-258M",
            do_table_structure=True,
            table_mode=TableFormerMode.ACCURATE,
            generate_page_images=True,
            generate_picture_images=True,
            max_pages=None,
        )

        if torch.cuda.is_available():
            vlm_options.accelerator_options.device = AcceleratorDevice.CUDA
            vlm_options.accelerator_options.cuda_use_flash_attention2 = True
        else:
            vlm_options.accelerator_options.device = AcceleratorDevice.CPU
            vlm_options.accelerator_options.num_threads = os.cpu_count() // 2  # Dynamic scaling
        pdf_options = PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=vlm_options,
            backend=PyPdfiumDocumentBackend, 
        )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: pdf_options,
            }
        )

    @log_execution_time
    def convert_document(
        self,
        source_path: str,
        output_dir: str = "C:/Users/h.goian/Documents/Maseera/Finance/Financial_Statemets_Analysis/Financial_Statement_Analysis_POC_Crewai/backend/src/financial_statement_analysis/output/processed_docs/",
        save_as_markdown: bool = False,
        save_as_json: bool = False,
    ) -> ConversionResult:
        """
        Convert PDF document to DoclingDocument format and optionally save the output.
        """
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"PDF file not found: {source_path}")

        logger.info("=" * 70)
        logger.info(f"📄 Converting Document: {source.name}")
        logger.info("=" * 70)

        try:
            result = self.converter.convert(source=str(source))
            confidence = result.document.confidence if hasattr(result.document, 'confidence') else result.confidence

            logger.info("✓ Conversion Complete - Quality Metrics:")

            logger.info(f"   Overall Quality: {confidence.mean_grade}")

            logger.info(f"   Lowest Area Quality: {confidence.low_grade}")

            if save_as_markdown or save_as_json:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                file_stem = source.stem

                if save_as_markdown:
                    md_path = output_path / f"{file_stem}.md"
                    with open(md_path, "w", encoding="utf-8") as f:
                        f.write(result.document.export_to_markdown())
                    logger.info(f"Saved Markdown to: {md_path}")

                if save_as_json:
                    json_path = output_path / f"{file_stem}.json"
                    with open(json_path, "w", encoding="utf-8") as f:
                        # We can use the model_dump_json method from pydantic
                        f.write(result.document.model_dump_json(indent=4))
                    logger.info(f"Saved JSON to: {json_path}")

            return result

        except Exception as e:
            logger.error(f"✗ Document conversion failed: {e}")
            raise


# # Inside __init__:
# pdf_options = PdfFormatOption(
#     pipeline_cls=PdfPipeline,  # Or VlmPipeline if hybrid
#     pipeline_options=pipeline_options,  # Your OCR/table options
#      # <-- Used here: Configures PyMuPDF as the PDF loader/parser
# )
if __name__ == "__main__":
    processor = DocumentProcessor()
    try:
        source_file = "C:/Users/h.goian/Documents/Maseera/Finance/Financial_Statemets_Analysis/Financial_Statement_Analysis_POC_Crewai/backend/knowledge/HSBC-11.pdf"
        processor.convert_document(
            source_path=source_file, save_as_markdown=True, save_as_json=True
        )
    except FileNotFoundError as e:
        logger.error(e)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")