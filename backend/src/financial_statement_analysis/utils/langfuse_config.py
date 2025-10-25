import os
from dotenv import load_dotenv
from langfuse import get_client
from openinference.instrumentation.crewai import CrewAIInstrumentor
from openinference.instrumentation.litellm import LiteLLMInstrumentor
from typing import List, Optional, Dict, Any

def init_langfuse():
    """
    Initialize Langfuse client and OpenInference instrumentation for CrewAI tracing. 
    This function verifies the Langfuse connection, initializes the client,
    and sets up instrumentation for CrewAI and LiteLLM to automatically capture operations.
    """
    langfuse = get_client()
    
    # Verify authentication
    if langfuse.auth_check():
        print("Langfuse client is authenticated and ready!")
    else:
        print("Authentication failed. Please check your credentials and host.")
        raise ValueError("Langfuse authentication failed.")
    
    # Initialize OpenInference instrumentation for CrewAI and LiteLLM
    CrewAIInstrumentor().instrument(skip_dep_check=True)
    LiteLLMInstrumentor().instrument()
    
    return langfuse