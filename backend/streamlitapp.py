# src/financial_statement_analysis/streamlitapp.py
import streamlit as st
import json
import tempfile
import os
from pathlib import Path
from copy import deepcopy
import pandas as pd
import io
from transformers import AutoTokenizer
from src.financial_statement_analysis.utils.document_processor import DocumentProcessor
from src.financial_statement_analysis.utils.document_chunker import EnhancedDocumentChunker
from src.financial_statement_analysis.utils.logging_config import setup_logger
from src.financial_statement_analysis.utils.langfuse_config import init_langfuse
from src.financial_statement_analysis.utils.vectorstore_save import save_chunks_to_qdrant
from src.financial_statement_analysis.crew import run_crew  # Import the new function

# Initialize Langfuse, instrumentation for tracing
langfuse = init_langfuse()  # Add this call; returns the langfuse client
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

        # Step 5: Run the crew analysis using the new function
        logger.info("Step 5: Running crew analysis...")
        crew_outputs = run_crew(filename)
        
        return crew_outputs
    
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise e

def display_editable_metadata(key, data_dict):
    df = pd.DataFrame([data_dict])
    column_config = {
        "entity_name": st.column_config.TextColumn("Entity Name"),
        "unit": st.column_config.TextColumn("Unit"),
        "report_year": st.column_config.NumberColumn("Report Year"),
        "prev_year": st.column_config.NumberColumn("Previous Year"),
        "fiscal_month_end": st.column_config.NumberColumn("Fiscal Month End"),
    }
    edited_df = st.data_editor(
        df,
        column_config=column_config,
        hide_index=True,
        num_rows="fixed",
        disabled=st.session_state.confirmed,
        key=key
    )
    return edited_df.iloc[0].to_dict() if not edited_df.empty else data_dict

def display_editable_metrics(key, metrics_list, metadata, extra_cols=None):
    if not metrics_list:
        return []
    
    extra_cols = extra_cols or []
    df = pd.DataFrame(metrics_list)
    
    # Rename value columns dynamically
    rename_map = {
        'current_year_value': f'Value {metadata["report_year"]}',
        'previous_year_value': f'Value {metadata["prev_year"]}'
    }
    df_renamed = df.rename(columns=rename_map)
    
    # Define column order
    base_cols = ['metric_name'] + [col for col in df_renamed.columns if 'Value' in col]
    all_cols = base_cols + (['category'] if 'category' in df_renamed.columns else []) + extra_cols
    df_renamed = df_renamed[all_cols]
    
    # Column configs
    column_config = {
        "metric_name": st.column_config.TextColumn("Metric Name"),
    }
    for col in extra_cols:
        if col == 'confidence':
            column_config[col] = st.column_config.NumberColumn("Confidence", min_value=0.0, max_value=1.0, step=0.01)
        elif col == 'explanation':
            column_config[col] = st.column_config.TextColumn("Explanation", width="large")
    if 'category' in all_cols:
        column_config['category'] = st.column_config.SelectboxColumn("Category", options=[e.value for e in BalanceSheetCategory] if 'Assets' in extra_cols[0] else [e.value for e in IncomeStatementCategory])
    
    edited_df = st.data_editor(
        df_renamed,
        column_config=column_config,
        hide_index=True,
        num_rows="dynamic" if not st.session_state.confirmed else "fixed",
        disabled=st.session_state.confirmed,
        key=key
    )
    
    # Rename back
    reverse_map = {v: k for k, v in rename_map.items()}
    edited_df = edited_df.rename(columns=reverse_map)
    
    return edited_df.to_dict(orient='records')

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
                st.session_state.edited_data = deepcopy(result)
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
    metadata = data['metadata']  # For easy access
    
    # Use tabs for different sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Metadata", "Balance Sheet", "Key Metrics", "Income Statement", "Income Metrics", "Risk Metrics"])
    
    with tab1:
        st.subheader("📋 Metadata Table")
        data['metadata'] = display_editable_metadata("metadata_editor", data['metadata'])
    
    with tab2:
        st.subheader("🏦 Balance Sheet Metrics")
        data['balance_sheet']['metrics'] = display_editable_metrics("bs_editor", data['balance_sheet']['metrics'], metadata, extra_cols=[])
    
    with tab3:
        st.subheader("🔑 Key Financial Metrics")
        data['key_metrics']['metrics'] = display_editable_metrics("key_metrics_editor", data['key_metrics']['metrics'], metadata, extra_cols=[])
    
    with tab4:
        st.subheader("💰 Income Statement Metrics")
        data['income_statement']['metrics'] = display_editable_metrics("is_editor", data['income_statement']['metrics'], metadata, extra_cols=[])
    
    with tab5:
        st.subheader("📈 Key Income Metrics")
        data['income_metrics']['metrics'] = display_editable_metrics("income_metrics_editor", data['income_metrics']['metrics'], metadata, extra_cols=['confidence', 'explanation'])
    
    with tab6:
        st.subheader("⚠️ Key Risk Metrics")
        data['risk_metrics']['metrics'] = display_editable_metrics("risk_editor", data['risk_metrics']['metrics'], metadata, extra_cols=['confidence', 'explanation'])
    
    # Confirm and Download section
    st.header("3. Confirm and Export Data")
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
    
    # Download JSON
    with col3:
        st.download_button(
            label="💾 Download JSON",
            data=json.dumps(data, indent=2),
            file_name=f"financial_analysis_{data['metadata']['entity_name']}_{data['metadata']['report_year']}.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Download Excel with multiple sheets
    if st.button("📑 Download Excel", use_container_width=True):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            pd.DataFrame([data['metadata']]).to_excel(writer, sheet_name='Metadata', index=False)
            pd.DataFrame(data['balance_sheet']['metrics']).to_excel(writer, sheet_name='Balance Sheet', index=False)
            pd.DataFrame(data['key_metrics']['metrics']).to_excel(writer, sheet_name='Key Metrics', index=False)
            pd.DataFrame(data['income_statement']['metrics']).to_excel(writer, sheet_name='Income Statement', index=False)
            pd.DataFrame(data['income_metrics']['metrics']).to_excel(writer, sheet_name='Income Metrics', index=False)
            pd.DataFrame(data['risk_metrics']['metrics']).to_excel(writer, sheet_name='Risk Metrics', index=False)
        output.seek(0)
        st.download_button(
            label="📑 Download Excel",
            data=output,
            file_name=f"financial_analysis_{data['metadata']['entity_name']}_{data['metadata']['report_year']}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# Sidebar with instructions (keep as is)
with st.sidebar:
    st.header("📖 Instructions")
    st.markdown("""
    1. **Upload** a PDF financial statement
    2. Click **Analyze Document** to extract data
    3. **Review** the tables in each tab
    4. **Edit** values, add/delete rows as needed
    5. Click **Confirm Data** when satisfied
    6. **Download** JSON or Excel output
    
    --- 
    
    ### ℹ️ Processing Info
    - Uses direct document processing (no API)
    - Powered by CrewAI and Gemini
    - Local Qdrant vector database
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Supported Sections")
    st.markdown("""
    - Metadata Table
    - Balance Sheet Metrics
    - Key Financial Metrics (Deposits, Loans)
    - Income Statement Metrics
    - Key Income Metrics (with confidence/explanation)
    - Key Risk Metrics (with confidence/explanation)
    """)
