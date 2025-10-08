import streamlit as st
import json
import tempfile
import os
from pathlib import Path

# Import your processing modules
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer

from src.financial_statement_analysis.utils.document_processor import DocumentProcessor
from src.financial_statement_analysis.utils.document_chunker import EnhancedDocumentChunker
from src.financial_statement_analysis.utils.logging_config import setup_logger
from src.financial_statement_analysis.utils.vectorstore_save import save_chunks_to_qdrant
from src.financial_statement_analysis.crew import crew

# Page configuration
st.set_page_config(
    page_title="Financial Statement Analyzer",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None
if 'confirmed' not in st.session_state:
    st.session_state.confirmed = False
if 'edited_data' not in st.session_state:
    st.session_state.edited_data = None

logger = setup_logger()

def process_document(file_path):
    """Process the document using main.py logic"""
    try:
        # Step 1: Convert document
        logger.info("Step 1: Converting document...")
        processor = DocumentProcessor()
        result = processor.convert_document(source_path=file_path)
        docling_doc = result.document

        # Step 2: Initialize tokenizer
        logger.info("Step 2: Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-embedding-english-r2")

        # Step 3: Create enhanced chunks with Gemini
        logger.info("Step 3: Creating enhanced chunks with Gemini...")
        enhanced_chunker = EnhancedDocumentChunker(
            tokenizer=tokenizer,
            model_name="gemini-2.0-flash",
            max_tokens=1024,
        )
        chunks = enhanced_chunker.chunk_document(docling_doc, file_path)
        
        logger.info(f"✓ Ready for RAG pipeline with {len(chunks)} enhanced chunks.")

        # Step 4: Save chunks to local Qdrant vector database
        logger.info("Step 4: Saving chunks to local Qdrant...")
        save_chunks_to_qdrant(chunks, file_path)    
        
        filename = os.path.basename(file_path)
        inputs = {'filename': filename}

        # Step 5: Run the crew analysis
        logger.info("Step 5: Running crew analysis...")
        crew_output = crew.kickoff(inputs=inputs)
        
        # Extract the Pydantic model from CrewOutput
        # CrewOutput has a .pydantic attribute that contains the structured output
        if hasattr(crew_output, 'pydantic'):
            result = crew_output.pydantic.model_dump()
        elif hasattr(crew_output, 'json_dict'):
            result = crew_output.json_dict
        elif hasattr(crew_output, 'dict'):
            result = crew_output.dict()
        else:
            # Fallback: try to convert to dict
            result = dict(crew_output)
        
        return result
    
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise e

st.title("📊 Financial Statement Analyzer")
st.markdown("Upload a PDF financial statement to extract and verify structured data")

# File upload section
st.header("1. Upload Financial Statement")
uploaded_file = st.file_uploader(
    "Choose a PDF file", 
    type=['pdf'],
    help="Upload a financial statement in PDF format"
)

if uploaded_file is not None:
    st.success(f"✅ File uploaded: {uploaded_file.name}")
    
    # Analyze button
    if st.button("🔍 Analyze Document", type="primary", disabled=st.session_state.confirmed):
        with st.spinner("Starting analysis... This may take several minutes."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Process the document directly
                result = process_document(tmp_path)
                
                # Store result in session state
                st.session_state.analysis_data = result
                st.session_state.edited_data = json.loads(json.dumps(result))
                st.session_state.confirmed = False
                
                # Clean up temp file
                os.unlink(tmp_path)
                
                st.success("✅ Analysis completed!")
                st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

# Display and edit analysis results
if st.session_state.edited_data is not None:
    st.header("2. Review and Edit Extracted Data")
    
    if st.session_state.confirmed:
        st.success("✅ Data confirmed and saved!")
    
    data = st.session_state.edited_data
    
    # Metadata Section
    st.subheader("📋 Metadata")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        data['metadata']['entity_name'] = st.text_input(
            "Entity Name",
            value=data['metadata']['entity_name'],
            key="entity_name",
            disabled=st.session_state.confirmed
        )
    
    with col2:
        data['metadata']['unit'] = st.text_input(
            "Unit",
            value=data['metadata']['unit'],
            key="unit",
            disabled=st.session_state.confirmed
        )
    
    with col3:
        data['metadata']['report_year'] = st.number_input(
            "Report Year",
            value=data['metadata']['report_year'],
            min_value=1900,
            max_value=2100,
            key="report_year",
            disabled=st.session_state.confirmed
        )
    
    with col4:
        data['metadata']['prev_year'] = st.number_input(
            "Previous Year",
            value=data['metadata']['prev_year'],
            min_value=1900,
            max_value=2100,
            key="prev_year",
            disabled=st.session_state.confirmed
        )
    
    years = [data['metadata']['report_year'], data['metadata']['prev_year']]
    
    # Income Statement Section
    st.subheader("💰 Income Statement")
    income_metrics = [
        ('net_interest_income', 'Net Interest Income'),
        ('non_interest_income', 'Non-Interest Income'),
        ('operating_expenses', 'Operating Expenses'),
        ('interest_expense', 'Interest Expense'),
        ('net_income', 'Net Income')
    ]
    
    for metric_key, metric_label in income_metrics:
        cols = st.columns([3] + [1] * len(years))
        cols[0].markdown(f"**{metric_label}**")
        for idx, year in enumerate(years):
            value = data['income_statement'][metric_key].get(year)
            new_value = cols[idx + 1].number_input(
                f"{year}",
                value=float(value) if value is not None else 0.0,
                format="%.2f",
                key=f"income_{metric_key}_{year}",
                label_visibility="visible",
                disabled=st.session_state.confirmed
            )
            data['income_statement'][metric_key][year] = new_value if new_value != 0.0 else None
    
    # Balance Sheet Section
    st.subheader("🏦 Balance Sheet")
    
    # Total Assets
    st.markdown("**Total Assets**")
    cols = st.columns([3] + [1] * len(years))
    cols[0].write("")
    for idx, year in enumerate(years):
        value = data['balance_sheet']['total_assets'].get(year)
        new_value = cols[idx + 1].number_input(
            f"{year}",
            value=float(value) if value is not None else 0.0,
            format="%.2f",
            key=f"total_assets_{year}",
            disabled=st.session_state.confirmed
        )
        data['balance_sheet']['total_assets'][year] = new_value if new_value != 0.0 else None
    
    # Asset Line Items
    with st.expander("📊 Asset Line Items", expanded=False):
        # Add new asset line item
        if not st.session_state.confirmed:
            col1, col2 = st.columns([3, 1])
            with col1:
                new_asset_item = st.text_input("Add new asset line item", key="new_asset_item", placeholder="Enter item name...")
            with col2:
                if st.button("➕ Add", key="add_asset_btn", use_container_width=True):
                    if new_asset_item and new_asset_item not in data['balance_sheet']['assets_line_items']:
                        data['balance_sheet']['assets_line_items'][new_asset_item] = {year: None for year in years}
                        st.rerun()
            st.markdown("---")
        
        items_to_remove = []
        for line_item, values in data['balance_sheet']['assets_line_items'].items():
            cols = st.columns([3] + [1] * len(years) + [0.5])
            cols[0].markdown(f"*{line_item}*")
            for idx, year in enumerate(years):
                value = values.get(year)
                new_value = cols[idx + 1].number_input(
                    f"{year}",
                    value=float(value) if value is not None else 0.0,
                    format="%.2f",
                    key=f"asset_{line_item}_{year}",
                    label_visibility="collapsed",
                    disabled=st.session_state.confirmed
                )
                data['balance_sheet']['assets_line_items'][line_item][year] = new_value if new_value != 0.0 else None
            
            # Delete button
            if not st.session_state.confirmed:
                if cols[-1].button("🗑️", key=f"del_asset_{line_item}", help="Delete this item"):
                    items_to_remove.append(line_item)
        
        # Remove items after iteration
        for item in items_to_remove:
            del data['balance_sheet']['assets_line_items'][item]
            st.rerun()
    
    # Total Liabilities
    st.markdown("**Total Liabilities**")
    cols = st.columns([3] + [1] * len(years))
    cols[0].write("")
    for idx, year in enumerate(years):
        value = data['balance_sheet']['total_liabilities'].get(year)
        new_value = cols[idx + 1].number_input(
            f"{year}",
            value=float(value) if value is not None else 0.0,
            format="%.2f",
            key=f"total_liabilities_{year}",
            disabled=st.session_state.confirmed
        )
        data['balance_sheet']['total_liabilities'][year] = new_value if new_value != 0.0 else None
    
    # Liability Line Items
    with st.expander("📊 Liability Line Items", expanded=False):
        for line_item, values in data['balance_sheet']['liabilities_line_items'].items():
            cols = st.columns([3] + [1] * len(years))
            cols[0].markdown(f"*{line_item}*")
            for idx, year in enumerate(years):
                value = values.get(year)
                new_value = cols[idx + 1].number_input(
                    f"{year}",
                    value=float(value) if value is not None else 0.0,
                    format="%.2f",
                    key=f"liability_{line_item}_{year}",
                    label_visibility="collapsed",
                    disabled=st.session_state.confirmed
                )
                data['balance_sheet']['liabilities_line_items'][line_item][year] = new_value if new_value != 0.0 else None
    
    # Total Equity
    st.markdown("**Total Equity**")
    cols = st.columns([3] + [1] * len(years))
    cols[0].write("")
    for idx, year in enumerate(years):
        value = data['balance_sheet']['total_equity'].get(year)
        new_value = cols[idx + 1].number_input(
            f"{year}",
            value=float(value) if value is not None else 0.0,
            format="%.2f",
            key=f"total_equity_{year}",
            disabled=st.session_state.confirmed
        )
        data['balance_sheet']['total_equity'][year] = new_value if new_value != 0.0 else None
    
    # Equity Line Items
    with st.expander("📊 Equity Line Items", expanded=False):
        for line_item, values in data['balance_sheet']['equity_line_items'].items():
            cols = st.columns([3] + [1] * len(years))
            cols[0].markdown(f"*{line_item}*")
            for idx, year in enumerate(years):
                value = values.get(year)
                new_value = cols[idx + 1].number_input(
                    f"{year}",
                    value=float(value) if value is not None else 0.0,
                    format="%.2f",
                    key=f"equity_{line_item}_{year}",
                    label_visibility="collapsed",
                    disabled=st.session_state.confirmed
                )
                data['balance_sheet']['equity_line_items'][line_item][year] = new_value if new_value != 0.0 else None
    
    # Risk and Liquidity Section
    st.subheader("⚠️ Risk and Liquidity Metrics")
    risk_metrics = [
        ('total_loans', 'Total Loans'),
        ('total_deposits', 'Total Deposits'),
        ('loan_loss_provisions', 'Loan Loss Provisions'),
        ('non_performing_loans', 'Non-Performing Loans'),
        ('regulatory_capital', 'Regulatory Capital'),
        ('risk_weighted_assets', 'Risk Weighted Assets'),
        ('high_quality_liquid_assets', 'High Quality Liquid Assets'),
        ('net_cash_outflows_over_30_days', 'Net Cash Outflows (30 days)')
    ]
    
    for metric_key, metric_label in risk_metrics:
        cols = st.columns([3] + [1] * len(years))
        cols[0].markdown(f"**{metric_label}**")
        for idx, year in enumerate(years):
            value = data['risk_and_liquidity'][metric_key].get(year)
            new_value = cols[idx + 1].number_input(
                f"{year}",
                value=float(value) if value is not None else 0.0,
                format="%.2f",
                key=f"risk_{metric_key}_{year}",
                label_visibility="visible",
                disabled=st.session_state.confirmed
            )
            data['risk_and_liquidity'][metric_key][year] = new_value if new_value != 0.0 else None
    
    # Confirm button
    st.header("3. Confirm Data")
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        if not st.session_state.confirmed:
            if st.button("✅ Confirm Data", type="primary", use_container_width=True):
                st.session_state.confirmed = True
                st.rerun()
    
    with col2:
        if st.session_state.confirmed:
            if st.button("📝 Edit Again", use_container_width=True):
                st.session_state.confirmed = False
                st.rerun()
    
    with col3:
        if st.download_button(
            label="💾 Download JSON",
            data=json.dumps(data, indent=2),
            file_name=f"financial_analysis_{data['metadata']['entity_name']}_{data['metadata']['report_year']}.json",
            mime="application/json",
            use_container_width=True
        ):
            st.success("Downloaded successfully!")

# Sidebar with instructions
with st.sidebar:
    st.header("📖 Instructions")
    st.markdown("""
    1. **Upload** a PDF financial statement
    2. Click **Analyze Document** to extract data
    3. **Review** the extracted fields
    4. **Edit** any incorrect values
    5. Click **Confirm Data** when satisfied
    6. **Download** the JSON output
    
    ---
    
    ### ℹ️ Processing Info
    - Uses direct document processing (no API)
    - Powered by CrewAI and Gemini
    - Local Qdrant vector database
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Supported Metrics")
    st.markdown("""
    - Metadata (Entity, Unit, Years)
    - Income Statement
    - Balance Sheet (Assets, Liabilities, Equity)
    - Risk & Liquidity Metrics
    """)