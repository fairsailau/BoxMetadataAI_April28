import streamlit as st
import os
import sys
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent))

# Import modules
from modules.authentication import authenticate
from modules.file_browser import file_browser
from modules.metadata_config import metadata_config
from modules.processing import process_files
from modules.results_viewer import view_results
from modules.direct_metadata_application_v3_fixed import apply_metadata_direct as apply_metadata
from modules.document_categorization import document_categorization
from modules.metadata_template_retrieval import get_metadata_templates, initialize_template_state
from modules.user_journey_guide import user_journey_guide, display_step_help


# Session timeout configuration
SESSION_TIMEOUT_MINUTES = 60  # Increased from default

# Centralized session state initialization
def initialize_session_state():
    """
    Initialize all session state variables in a centralized function
    to ensure consistency across the application
    """
    # Core session state variables
    if not hasattr(st.session_state, "authenticated"):
        st.session_state.authenticated = False
        logger.info("Initialized authenticated in session state")
    
    if not hasattr(st.session_state, "client"):
        st.session_state.client = None
        logger.info("Initialized client in session state")
    
    if not hasattr(st.session_state, "current_page"):
        st.session_state.current_page = "Home"
        logger.info("Initialized current_page in session state")
    
    # Session management
    if not hasattr(st.session_state, "last_activity"):
        st.session_state.last_activity = datetime.now()
        logger.info("Initialized last_activity in session state")
    
    # File selection and processing variables
    if not hasattr(st.session_state, "selected_files"):
        st.session_state.selected_files = []
        logger.info("Initialized selected_files in session state")
    
    # Folder selection
    if not hasattr(st.session_state, "selected_folders"):
        st.session_state.selected_folders = []
        logger.info("Initialized selected_folders in session state")
    
    # Metadata configuration
    if not hasattr(st.session_state, "metadata_config"):
        st.session_state.metadata_config = {
            "extraction_method": "freeform",
            "freeform_prompt": "Extract key metadata from this document including dates, names, amounts, and other important information.",
            "use_template": False,
            "template_id": "",
            "custom_fields": [],
            "ai_model": "google__gemini_2_0_flash_001",
            "batch_size": 5
        }
        logger.info("Initialized metadata_config in session state")
    
    # Extraction results
    if not hasattr(st.session_state, "extraction_results"):
        st.session_state.extraction_results = {}
        logger.info("Initialized extraction_results in session state")
    
    # Selected results for metadata application - FIXED: Use direct attribute assignment
    if not hasattr(st.session_state, "selected_result_ids"):
        st.session_state.selected_result_ids = []
        logger.info("Initialized selected_result_ids in session state")
    
    # Application state for metadata application
    if not hasattr(st.session_state, "application_state"):
        st.session_state.application_state = {
            "is_applying": False,
            "applied_files": 0,
            "total_files": 0,
            "current_batch": [],
            "results": {},
            "errors": {}
        }
        logger.info("Initialized application_state in session state")
    
    # Processing state for file processing
    if not hasattr(st.session_state, "processing_state"):
        st.session_state.processing_state = {
            "is_processing": False,
            "processed_files": 0,
            "total_files": 0,
            "current_file_index": -1,
            "current_file": "",
            "results": {},
            "errors": {},
            "retries": {},
            "max_retries": 3,
            "retry_delay": 2,
            "visualization_data": {}
        }
        logger.info("Initialized processing_state in session state")
    
    # Debug information
    if not hasattr(st.session_state, "debug_info"):
        st.session_state.debug_info = []
        logger.info("Initialized debug_info in session state")
    
    # Metadata templates
    if not hasattr(st.session_state, "metadata_templates"):
        st.session_state.metadata_templates = {}
        logger.info("Initialized metadata_templates in session state")
    
    # Feedback data
    if not hasattr(st.session_state, "feedback_data"):
        st.session_state.feedback_data = {}
        logger.info("Initialized feedback_data in session state")
    
    # Initialize document categorization state
    if not hasattr(st.session_state, "document_categorization"):
        st.session_state.document_categorization = {
            "is_categorized": False,
            "categorized_files": 0,
            "total_files": 0,
            "results": {},  # file_id -> categorization result
            "errors": {},   # file_id -> error message
            "processing_state": {
                "is_processing": False,
                "current_file_index": -1,
                "current_file": "",
                "current_batch": [],
                "batch_size": 5
            }
        }
        logger.info("Initialized document_categorization in session state")
    
    # Initialize template state
    initialize_template_state()
    
    # UI preferences
    if not hasattr(st.session_state, "ui_preferences"):
        st.session_state.ui_preferences = {
            "show_user_journey": True,
            "show_step_help": True,
            "dark_mode": False,
            "compact_view": False
        }
        logger.info("Initialized ui_preferences in session state")

# Initialize session state
initialize_session_state()

# Update last activity timestamp
def update_activity():
    st.session_state.last_activity = datetime.now()

# Check if session has timed out
def check_session_timeout():
    if not hasattr(st.session_state, "last_activity"):
        update_activity()
        return False
    
    time_since_last_activity = datetime.now() - st.session_state.last_activity
    if time_since_last_activity > timedelta(minutes=SESSION_TIMEOUT_MINUTES):
        logger.info(f"Session timed out after {time_since_last_activity}")
        return True
    
    return False

# Navigation function
def navigate_to(page):
    st.session_state.current_page = page
    update_activity()
    logger.info(f"Navigated to page: {page}")

# Sidebar navigation
with st.sidebar:
    st.title("Box AI Metadata")
    
    # Show navigation only if authenticated
    if hasattr(st.session_state, "authenticated") and st.session_state.authenticated:
        # Check for session timeout
        if check_session_timeout():
            st.warning("Your session has timed out due to inactivity. Please log in again.")
            st.session_state.authenticated = False
            st.session_state.client = None
            navigate_to("Home")
            st.rerun()
        else:
            # Update activity timestamp
            update_activity()
        
        # Display session timeout info
        remaining_time = SESSION_TIMEOUT_MINUTES - (datetime.now() - st.session_state.last_activity).total_seconds() / 60
        st.caption(f"Session timeout: {int(remaining_time)} minutes remaining")
        
        # Display user journey guide
        if st.session_state.ui_preferences.get("show_user_journey", True):
            user_journey_guide(st.session_state.current_page)
        
        st.subheader("Navigation")
        
        if st.button("Home", use_container_width=True, key="nav_home"):
            navigate_to("Home")
        
        if st.button("File Browser", use_container_width=True, key="nav_file_browser"):
            navigate_to("File Browser")
        
        if st.button("Document Categorization", use_container_width=True, key="nav_doc_cat"):
            navigate_to("Document Categorization")
            
        if st.button("Metadata Configuration", use_container_width=True, key="nav_meta_config"):
            navigate_to("Metadata Configuration")
            
        if st.button("Process Files", use_container_width=True, key="nav_process"):
            navigate_to("Process Files")
            
        if st.button("View Results", use_container_width=True, key="nav_view"):
            navigate_to("View Results")
            
        if st.button("Apply Metadata", use_container_width=True, key="nav_apply"):
            navigate_to("Apply Metadata")
        
        # Metadata Templates section
        st.subheader("Metadata Templates")
        
        # Display template count
        template_count = len(st.session_state.metadata_templates) if hasattr(st.session_state, "metadata_templates") else 0
        st.write(f"{template_count} templates loaded")
        
        # Display last update time
        if hasattr(st.session_state, "template_cache_timestamp") and st.session_state.template_cache_timestamp:
            cache_time = datetime.fromtimestamp(st.session_state.template_cache_timestamp)
            st.write(f"Last updated: {cache_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Refresh templates button
        if st.button("Refresh Templates", key="refresh_templates_btn"):
            with st.spinner("Refreshing metadata templates..."):
                templates = get_metadata_templates(st.session_state.client, force_refresh=True)
                st.session_state.template_cache_timestamp = time.time()
                st.success(f"Retrieved {len(templates)} metadata templates")
                st.rerun()
        
        # UI Settings
        with st.expander("UI Settings", expanded=False):
            st.session_state.ui_preferences["show_user_journey"] = st.checkbox(
                "Show User Journey Guide", 
                value=st.session_state.ui_preferences.get("show_user_journey", True),
                key="show_user_journey_checkbox"
            )
            
            st.session_state.ui_preferences["show_step_help"] = st.checkbox(
                "Show Step Help", 
                value=st.session_state.ui_preferences.get("show_step_help", True),
                key="show_step_help_checkbox"
            )
            
            st.session_state.ui_preferences["compact_view"] = st.checkbox(
                "Compact View", 
                value=st.session_state.ui_preferences.get("compact_view", False),
                key="compact_view_checkbox"
            )
        
        # Logout button
        if st.button("Logout", use_container_width=True, key="nav_logout"):
            st.session_state.authenticated = False
            st.session_state.client = None
            navigate_to("Home")
            st.rerun()
    
    # Show app info
    st.subheader("About")
    st.info(
        "This app connects to Box.com and uses Box AI API "
        "to extract metadata from files and apply it at scale."
    )

# Main content area
if not hasattr(st.session_state, "authenticated") or not st.session_state.authenticated:
    # Authentication page
    authenticate()
else:
    # Update activity timestamp
    update_activity()
    
    # Retrieve metadata templates if authenticated and not already cached
    if st.session_state.authenticated and st.session_state.client:
        if not st.session_state.metadata_templates:
            with st.spinner("Retrieving metadata templates..."):
                templates = get_metadata_templates(st.session_state.client)
                st.session_state.template_cache_timestamp = time.time()
                logger.info(f"Retrieved {len(templates)} metadata templates")
    
    # Display step help if enabled
    if st.session_state.ui_preferences.get("show_step_help", True):
        display_step_help(st.session_state.current_page)
    
    # Display current page based on navigation
    if not hasattr(st.session_state, "current_page") or st.session_state.current_page == "Home":
        st.title("Box AI Metadata Extraction")
        
        st.write("""
        ## Welcome to Box AI Metadata Extraction App
        
        This application helps you extract metadata from your Box files using Box AI API 
        and apply it at scale. Follow these steps to get started:
        """)
        
        # Enhanced welcome page with visual workflow
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("### Workflow Steps")
            
            workflow_steps = [
                {"icon": "📁", "title": "Select Files", "description": "Browse and select files or entire folders from Box"},
                {"icon": "🏷️", "title": "Categorize Documents", "description": "Use AI to automatically categorize your documents"},
                {"icon": "⚙️", "title": "Configure Metadata", "description": "Set up metadata extraction parameters"},
                {"icon": "🔄", "title": "Process Files", "description": "Extract metadata using Box AI"},
                {"icon": "👁️", "title": "Review Results", "description": "View and edit extracted metadata"},
                {"icon": "✅", "title": "Apply Metadata", "description": "Apply metadata to your Box files"}
            ]
            
            for i, step in enumerate(workflow_steps):
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 15px;">
                    <div style="font-size: 24px; margin-right: 10px;">{step['icon']}</div>
                    <div>
                        <strong>{i+1}. {step['title']}</strong><br>
                        <span style="font-size: 0.9em;">{step['description']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.write("### Key Features")
            
            features = [
                "🔍 AI-powered metadata extraction",
                "📊 Batch processing for efficiency",
                "📂 Folder selection support",
                "🏷️ Document categorization",
                "📝 Customizable metadata templates",
                "🔄 Automatic metadata application"
            ]
            
            for feature in features:
                st.markdown(f"- {feature}")
        
        # Quick actions
        st.subheader("Quick Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Browse Files", use_container_width=True, key="quick_browse"):
                navigate_to("File Browser")
                st.rerun()
        
        with col2:
            if st.button("Categorize Documents", use_container_width=True, key="quick_categorize"):
                navigate_to("Document Categorization")
                st.rerun()
        
        with col3:
            if st.button("Configure Metadata", use_container_width=True, key="quick_configure"):
                navigate_to("Metadata Configuration")
                st.rerun()
    
    elif st.session_state.current_page == "File Browser":
        file_browser()
    
    elif st.session_state.current_page == "Document Categorization":
        document_categorization()
    
    elif st.session_state.current_page == "Metadata Configuration":
        metadata_config()
    
    elif st.session_state.current_page == "Process Files":
        process_files()
    
    elif st.session_state.current_page == "View Results":
        view_results()
    
    elif st.session_state.current_page == "Apply Metadata":
        apply_metadata()
