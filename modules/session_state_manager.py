import streamlit as st
import logging
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format=\'%(asctime)s - %(name)s - %(levelname)s - %(message)s\')
logger = logging.getLogger(__name__)

def initialize_state(key: str, default_value: Any):
    """
    Initializes a specific key in Streamlit\'s session state if it doesn\"t exist.

    Args:
        key (str): The session state key to initialize.
        default_value: The default value to set if the key is not present.
    """
    if key not in st.session_state:
        st.session_state[key] = default_value
        logger.info(f"Initialized {key} in session state")

def initialize_app_session_state():
    """
    Global session state initialization function to be called at the start of the application.
    This ensures all required session state variables are properly initialized.
    """
    # Core session state variables
    initialize_state("authenticated", False)
    initialize_state("client", None)
    initialize_state("current_page", "Home")
    initialize_state("last_activity", time.time()) # Add last_activity for timeout
    
    # File selection and metadata configuration
    initialize_state("selected_files", [])
    initialize_state("selected_folders", []) # Add selected_folders
    initialize_state("metadata_config", {
        "extraction_method": "freeform",
        "freeform_prompt": "Extract key metadata from this document.",
        "use_template": False,
        "template_id": "",
        "custom_fields": [],
        "ai_model": "azure__openai__gpt_4o_mini", # Default model
        "batch_size": 5,
        "processing_mode": "Parallel",
        "max_workers": 5,
        "use_template_cache": True,
        "auto_apply_metadata": False
    })
    
    # Results and processing state
    initialize_state("extraction_results", {})
    initialize_state("selected_result_ids", [])
    initialize_state("application_state", {
        "is_applying": False,
        "progress": 0,
        "total_files": 0,
        "errors": {},
        "status_message": "Ready to apply",
        "cancel_requested": False
    })
    initialize_state("processing_state", {
        "is_processing": False,
        "progress": 0,
        "total_files": 0,
        "current_batch": 0,
        "total_batches": 0,
        "errors": {},
        "status_message": "Ready to process",
        "cancel_requested": False
    })
    
    # Debug and feedback
    initialize_state("debug_info", {})
    initialize_state("metadata_templates", [])
    initialize_state("feedback_data", {})
    
    # Document categorization state
    initialize_state("document_categorization", {
        "is_categorizing": False,
        "progress": 0,
        "total_files": 0,
        "results": {},
        "errors": {},
        "status_message": "Ready to categorize",
        "cancel_requested": False
    })
    
    # Template cache timestamp and mapping
    # Moved initialization here from metadata_template_retrieval for centralization
    initialize_state("template_cache_timestamp", None)
    initialize_state("document_type_to_template", {})
    
    # UI Preferences
    initialize_state("ui_preferences", {
        "theme": "Light",
        "show_debug": False
    })

def get_safe_session_state(key, default_value=None):
    """
    Safely get a value from session state with a fallback default value.
    This prevents KeyError when accessing session state variables.
    
    Args:
        key (str): The session state key to access
        default_value: The default value to return if key doesn\'t exist
        
    Returns:
        The value from session state or the default value
    """
    return st.session_state.get(key, default_value)

def set_safe_session_state(key, value):
    """
    Safely set a value in session state.
    This ensures the session state is properly initialized before setting values.
    
    Args:
        key (str): The session state key to set
        value: The value to set
    """
    try:
        st.session_state[key] = value
        return True
    except Exception as e:
        logger.error(f"Error setting session state key \'{key}\': {str(e)}")
        return False

def reset_session_state():
    """
    Reset the session state to its initial values.
    This can be used as a recovery mechanism when errors occur.
    """
    # Clear specific session state variables
    keys_to_reset = [
        "authenticated",
        "client",
        "current_page",
        "selected_files",
        "selected_folders",
        "metadata_config",
        "extraction_results", 
        "selected_result_ids", 
        "application_state", 
        "processing_state",
        "debug_info",
        "metadata_templates",
        "feedback_data",
        "document_categorization",
        "template_cache_timestamp",
        "document_type_to_template",
        "ui_preferences"
    ]
    
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    
    # Re-initialize session state
    initialize_app_session_state()
    
    logger.info("Session state has been reset")
    return True

def debug_session_state():
    """
    Create a debug view of the current session state.
    This can be used to diagnose session state issues.
    
    Returns:
        dict: A dictionary containing debug information about session state
    """
    debug_info = {
        "session_state_keys": list(st.session_state.keys()),
        "has_extraction_results": "extraction_results" in st.session_state,
        "extraction_results_type": str(type(get_safe_session_state("extraction_results"))),
        "extraction_results_keys": list(get_safe_session_state("extraction_results", {}).keys()),
        "has_selected_files": "selected_files" in st.session_state,
        "selected_files_count": len(get_safe_session_state("selected_files", [])),
        "has_processing_state": "processing_state" in st.session_state,
        "has_application_state": "application_state" in st.session_state
    }
    
    # Add more details as needed
    # logger.info(f"Session state debug info: {debug_info}")
    return debug_info

