import streamlit as st
import time
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import json
import concurrent.futures

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Debug mode flag
DEBUG_MODE = True

def process_files():
    """
    Process files for metadata extraction with Streamlit-compatible processing
    """
    st.title("Process Files")
    
    # Add debug information
    if "debug_info" not in st.session_state:
        st.session_state.debug_info = []
    
    # Add metadata templates
    if "metadata_templates" not in st.session_state:
        st.session_state.metadata_templates = {}
    
    # Add feedback data
    if "feedback_data" not in st.session_state:
        st.session_state.feedback_data = {}
    
    # Initialize extraction results if not exists
    if "extraction_results" not in st.session_state:
        st.session_state.extraction_results = {}
    
    try:
        if not st.session_state.authenticated or not st.session_state.client:
            st.error("Please authenticate with Box first")
            return
        
        if not st.session_state.selected_files:
            st.warning("No files selected. Please select files in the File Browser first.")
            if st.button("Go to File Browser", key="go_to_file_browser_button"):
                st.session_state.current_page = "File Browser"
                st.rerun()
            return
        
        if "metadata_config" not in st.session_state or (
            st.session_state.metadata_config["extraction_method"] == "structured" and 
            not st.session_state.metadata_config["use_template"] and 
            not st.session_state.metadata_config["custom_fields"]
        ):
            st.warning("Metadata configuration is incomplete. Please configure metadata extraction parameters.")
            if st.button("Go to Metadata Configuration", key="go_to_metadata_config_button"):
                st.session_state.current_page = "Metadata Configuration"
                st.rerun()
            return
        
        # Initialize processing state
        if "processing_state" not in st.session_state:
            st.session_state.processing_state = {
                "is_processing": False,
                "processed_files": 0,
                "total_files": len(st.session_state.selected_files),
                "current_file_index": -1,
                "current_file": "",
                "results": {},
                "errors": {},
                "retries": {},
                "max_retries": 3,
                "retry_delay": 2,  # seconds
                "visualization_data": {}
            }
        
        # Display processing information
        st.write(f"Ready to process {len(st.session_state.selected_files)} files using the configured metadata extraction parameters.")
        
        # Enhanced batch processing controls
        with st.expander("Batch Processing Controls"):
            col1, col2 = st.columns(2)
            
            with col1:
                # Batch size control
                batch_size = st.number_input(
                    "Batch Size",
                    min_value=1,
                    max_value=50,
                    value=st.session_state.metadata_config.get("batch_size", 5),
                    key="batch_size_input"
                )
                st.session_state.metadata_config["batch_size"] = batch_size
                
                # Max retries control
                max_retries = st.number_input(
                    "Max Retries",
                    min_value=0,
                    max_value=10,
                    value=st.session_state.processing_state.get("max_retries", 3),
                    key="max_retries_input"
                )
                st.session_state.processing_state["max_retries"] = max_retries
            
            with col2:
                # Retry delay control
                retry_delay = st.number_input(
                    "Retry Delay (seconds)",
                    min_value=1,
                    max_value=30,
                    value=st.session_state.processing_state.get("retry_delay", 2),
                    key="retry_delay_input"
                )
                st.session_state.processing_state["retry_delay"] = retry_delay
                
                # Processing mode
                processing_mode = st.selectbox(
                    "Processing Mode",
                    options=["Sequential", "Parallel"],
                    index=0,
                    key="processing_mode_input"
                )
                st.session_state.processing_state["processing_mode"] = processing_mode
        
        # Template management
        with st.expander("Metadata Template Management"):
            st.write("#### Save Current Configuration as Template")
            template_name = st.text_input("Template Name", key="template_name_input")
            
            if st.button("Save Template", key="save_template_button"):
                if template_name:
                    st.session_state.metadata_templates[template_name] = st.session_state.metadata_config.copy()
                    st.success(f"Template '{template_name}' saved successfully!")
                else:
                    st.warning("Please enter a template name")
            
            st.write("#### Load Template")
            if st.session_state.metadata_templates:
                template_options = list(st.session_state.metadata_templates.keys())
                selected_template = st.selectbox(
                    "Select Template",
                    options=template_options,
                    key="load_template_select"
                )
                
                if st.button("Load Template", key="load_template_button"):
                    st.session_state.metadata_config = st.session_state.metadata_templates[selected_template].copy()
                    st.success(f"Template '{selected_template}' loaded successfully!")
            else:
                st.info("No saved templates yet")
        
        # Display configuration summary
        with st.expander("Configuration Summary"):
            st.write("#### Extraction Method")
            st.write(f"Method: {st.session_state.metadata_config['extraction_method'].capitalize()}")
            
            if st.session_state.metadata_config["extraction_method"] == "structured":
                if st.session_state.metadata_config["use_template"]:
                    st.write(f"Using template: Template ID {st.session_state.metadata_config['template_id']}")
                else:
                    st.write(f"Using {len(st.session_state.metadata_config['custom_fields'])} custom fields")
                    for i, field in enumerate(st.session_state.metadata_config['custom_fields']):
                        st.write(f"- {field.get('display_name', field.get('name', ''))} ({field.get('type', 'string')})")
            else:
                st.write("Freeform prompt:")
                st.write(f"> {st.session_state.metadata_config['freeform_prompt']}")
            
            st.write(f"AI Model: {st.session_state.metadata_config['ai_model']}")
            st.write(f"Batch Size: {st.session_state.metadata_config['batch_size']}")
        
        # Display selected files
        with st.expander("Selected Files"):
            for file in st.session_state.selected_files:
                st.write(f"- {file['name']} (Type: {file['type']})")
        
        # Process files button
        col1, col2 = st.columns(2)
        
        with col1:
            start_button = st.button(
                "Start Processing",
                disabled=st.session_state.processing_state["is_processing"],
                use_container_width=True,
                key="start_processing_button"
            )
        
        with col2:
            cancel_button = st.button(
                "Cancel Processing",
                disabled=not st.session_state.processing_state["is_processing"],
                use_container_width=True,
                key="cancel_processing_button"
            )
        
        # Progress tracking
        progress_container = st.container()
        
        # Process files
        if start_button:
            # Reset processing state
            st.session_state.processing_state = {
                "is_processing": True,
                "processed_files": 0,
                "total_files": len(st.session_state.selected_files),
                "current_file_index": -1,
                "current_file": "",
                "results": {},
                "errors": {},
                "retries": {},
                "max_retries": max_retries,
                "retry_delay": retry_delay,
                "processing_mode": processing_mode,
                "visualization_data": {}
            }
            
            # Reset extraction results
            st.session_state.extraction_results = {}
            
            # Get metadata extraction functions
            extraction_functions = get_extraction_functions()
            
            # Process files with progress tracking
            process_files_with_progress(
                st.session_state.selected_files,
                extraction_functions,
                batch_size=batch_size,
                processing_mode=processing_mode
            )
        
        # Cancel processing
        if cancel_button and st.session_state.processing_state.get("is_processing", False):
            st.session_state.processing_state["is_processing"] = False
            st.warning("Processing cancelled")
        
        # Display processing progress
        if st.session_state.processing_state.get("is_processing", False):
            with progress_container:
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Update progress
                processed_files = st.session_state.processing_state["processed_files"]
                total_files = st.session_state.processing_state["total_files"]
                current_file = st.session_state.processing_state["current_file"]
                
                # Calculate progress
                progress = processed_files / total_files if total_files > 0 else 0
                
                # Update progress bar
                progress_bar.progress(progress)
                
                # Update status text
                if current_file:
                    status_text.text(f"Processing {current_file}... ({processed_files}/{total_files})")
                else:
                    status_text.text(f"Processed {processed_files}/{total_files} files")
        
        # Display processing results
        if "results" in st.session_state.processing_state and st.session_state.processing_state["results"]:
            st.write("### Processing Results")
            
            # Display success message
            processed_files = len(st.session_state.processing_state["results"])
            error_files = len(st.session_state.processing_state["errors"]) if "errors" in st.session_state.processing_state else 0
            
            if error_files == 0:
                st.success(f"Processing complete! Successfully processed {processed_files} files.")
            else:
                st.warning(f"Processing complete! Successfully processed {processed_files} files with {error_files} errors.")
            
            # Display errors if any
            if "errors" in st.session_state.processing_state and st.session_state.processing_state["errors"]:
                st.write("### Errors")
                
                for file_id, error in st.session_state.processing_state["errors"].items():
                    # Find file name
                    file_name = ""
                    for file in st.session_state.selected_files:
                        if file["id"] == file_id:
                            file_name = file["name"]
                            break
                    
                    st.error(f"{file_name}: {error}")
            
            # Continue button
            st.write("---")
            if st.button("Continue to View Results", key="continue_to_results_button", use_container_width=True):
                st.session_state.current_page = "View Results"
                st.rerun()
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        logger.error(f"Error in process_files: {str(e)}")

# Helper function to extract structured data from API response
def extract_structured_data_from_response(response):
    """
    Extract structured data from various possible response structures
    
    Args:
        response (dict): API response
        
    Returns:
        dict: Extracted structured data (key-value pairs)
    """
    structured_data = {}
    extracted_text = ""
    
    # Log the response structure for debugging
    logger.info(f"Response structure: {json.dumps(response, indent=2) if isinstance(response, dict) else str(response)}")
    
    if isinstance(response, dict):
        # Check for answer field (contains structured data in JSON format)
        if "answer" in response and isinstance(response["answer"], dict):
            structured_data = response["answer"]
            logger.info(f"Found structured data in 'answer' field: {structured_data}")
            return structured_data
        
        # Check for answer field as string (JSON string)
        if "answer" in response and isinstance(response["answer"], str):
            try:
                answer_data = json.loads(response["answer"])
                if isinstance(answer_data, dict):
                    structured_data = answer_data
                    logger.info(f"Found structured data in 'answer' field (JSON string): {structured_data}")
                    return structured_data
            except json.JSONDecodeError:
                logger.warning(f"Could not parse 'answer' field as JSON: {response['answer']}")
        
        # Check for key-value pairs directly in response
        for key, value in response.items():
            if key not in ["error", "items", "response", "item_collection", "entries", "type", "id", "sequence_id"]:
                structured_data[key] = value
        
        # Check in response field
        if "response" in response and isinstance(response["response"], dict):
            response_obj = response["response"]
            if "answer" in response_obj and isinstance(response_obj["answer"], dict):
                structured_data = response_obj["answer"]
                logger.info(f"Found structured data in 'response.answer' field: {structured_data}")
                return structured_data
        
        # Check in items array
        if "items" in response and isinstance(response["items"], list) and len(response["items"]) > 0:
            item = response["items"][0]
            if isinstance(item, dict):
                if "answer" in item and isinstance(item["answer"], dict):
                    structured_data = item["answer"]
                    logger.info(f"Found structured data in 'items[0].answer' field: {structured_data}")
                    return structured_data
    
    # If we couldn't find structured data, return empty dict
    if not structured_data:
        logger.warning("Could not find structured data in response")
    
    return structured_data

def process_files_with_progress(files, extraction_functions, batch_size=5, processing_mode="Sequential"):
    """
    Process files with progress tracking
    
    Args:
        files: List of files to process
        extraction_functions: Dictionary of extraction functions
        batch_size: Number of files to process in parallel
        processing_mode: Processing mode (Sequential or Parallel)
    """
    # Check if already processing
    if not st.session_state.processing_state.get("is_processing", False):
        return
    
    # Get total files
    total_files = len(files)
    st.session_state.processing_state["total_files"] = total_files
    
    # Process files
    if processing_mode == "Parallel":
        # Process files in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            # Submit tasks
            future_to_file = {}
            for file in files:
                future = executor.submit(process_file, file, extraction_functions)
                future_to_file[future] = file
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_file):
                file = future_to_file[future]
                
                try:
                    result = future.result()
                    
                    # Update processing state
                    st.session_state.processing_state["processed_files"] += 1
                    st.session_state.processing_state["current_file"] = ""
                    
                    # Store result
                    if result["success"]:
                        st.session_state.processing_state["results"][file["id"]] = result["data"]
                        st.session_state.extraction_results[file["id"]] = result["data"]
                    else:
                        st.session_state.processing_state["errors"][file["id"]] = result["error"]
                
                except Exception as e:
                    # Update processing state
                    st.session_state.processing_state["processed_files"] += 1
                    st.session_state.processing_state["current_file"] = ""
                    
                    # Store error
                    st.session_state.processing_state["errors"][file["id"]] = str(e)
    else:
        # Process files sequentially
        for i, file in enumerate(files):
            # Check if processing was cancelled
            if not st.session_state.processing_state.get("is_processing", False):
                break
            
            # Update processing state
            st.session_state.processing_state["current_file_index"] = i
            st.session_state.processing_state["current_file"] = file["name"]
            
            try:
                # Process file
                result = process_file(file, extraction_functions)
                
                # Update processing state
                st.session_state.processing_state["processed_files"] += 1
                
                # Store result
                if result["success"]:
                    st.session_state.processing_state["results"][file["id"]] = result["data"]
                    st.session_state.extraction_results[file["id"]] = result["data"]
                else:
                    st.session_state.processing_state["errors"][file["id"]] = result["error"]
            
            except Exception as e:
                # Update processing state
                st.session_state.processing_state["processed_files"] += 1
                
                # Store error
                st.session_state.processing_state["errors"][file["id"]] = str(e)
    
    # Mark processing as complete
    st.session_state.processing_state["is_processing"] = False
    st.session_state.processing_state["current_file"] = ""
    
    # Rerun to update UI
    st.rerun()

def get_document_type_for_file(file_id):
    """
    Get document type for a file from categorization results
    
    Args:
        file_id: File ID
        
    Returns:
        str: Document type or None if not categorized
    """
    # Check if document categorization has been performed
    if (
        hasattr(st.session_state, "document_categorization") and 
        st.session_state.document_categorization.get("is_categorized", False) and
        file_id in st.session_state.document_categorization["results"]
    ):
        return st.session_state.document_categorization["results"][file_id]["document_type"]
    
    return None

def process_file(file, extraction_functions):
    """
    Process a single file
    
    Args:
        file: File to process
        extraction_functions: Dictionary of extraction functions
        
    Returns:
        dict: Processing result
    """
    try:
        file_id = file["id"]
        file_name = file["name"]
        
        logger.info(f"Processing file: {file_name} (ID: {file_id})")
        
        # Check if we have feedback data for this file
        feedback_key = f"{file_id}_{st.session_state.metadata_config['extraction_method']}"
        has_feedback = feedback_key in st.session_state.feedback_data
        
        if has_feedback:
            logger.info(f"Using feedback data for file: {file_name}")
        
        # Get document type for this file (if categorized)
        document_type = get_document_type_for_file(file_id)
        if document_type:
            logger.info(f"File {file_name} has document type: {document_type}")
        
        # Determine extraction method
        if st.session_state.metadata_config["extraction_method"] == "structured":
            # Structured extraction
            if st.session_state.metadata_config["use_template"]:
                # Get template ID based on document type if available
                template_id = None
                
                # Check if we have a document type and a mapping for it
                if document_type and hasattr(st.session_state, "document_type_to_template"):
                    mapped_template_id = st.session_state.document_type_to_template.get(document_type)
                    if mapped_template_id:
                        template_id = mapped_template_id
                        logger.info(f"Using document type specific template for {document_type}: {template_id}")
                
                # If no document type specific template, use the general one
                if not template_id:
                    template_id = st.session_state.metadata_config["template_id"]
                    logger.info(f"Using general template: {template_id}")
                
                # Parse the template ID to extract the correct components
                # Format is typically: scope_id_templateKey (e.g., enterprise_336904155_financialReport)
                parts = template_id.split('_')
                
                # Extract the scope and enterprise ID
                scope = parts[0]  # e.g., "enterprise"
                enterprise_id = parts[1] if len(parts) > 1 else ""
                
                # Extract the actual template key (last part)
                template_key = parts[-1] if len(parts) > 2 else template_id
                
                # Create metadata template reference with correct format according to Box API documentation
                metadata_template = {
                    "template_key": template_key,
                    "type": "metadata_template",
                    "scope": f"{scope}_{enterprise_id}"
                }
                
                logger.info(f"Using template-based extraction with template ID: {template_id}")
                
                # Use real API call
                api_result = extraction_functions["extract_structured_metadata"](
                    file_id=file_id,
                    metadata_template=metadata_template,
                    ai_model=st.session_state.metadata_config["ai_model"]
                )
                
                # Create a clean result object with the extracted data
                result = {}
                
                # Copy fields from API result to our result object
                if isinstance(api_result, dict):
                    for key, value in api_result.items():
                        if key not in ["error", "items", "response"]:
                            result[key] = value
                
                # Apply feedback if available
                if has_feedback:
                    feedback = st.session_state.feedback_data[feedback_key]
                    # Merge feedback with result, prioritizing feedback
                    for key, value in feedback.items():
                        result[key] = value
            else:
                # Custom fields extraction
                logger.info(f"Using custom fields extraction with {len(st.session_state.metadata_config['custom_fields'])} fields")
                
                # Use real API call
                api_result = extraction_functions["extract_structured_metadata"](
                    file_id=file_id,
                    fields=st.session_state.metadata_config["custom_fields"],
                    ai_model=st.session_state.metadata_config["ai_model"]
                )
                
                # Create a clean result object with the extracted data
                result = {}
                
                # Copy fields from API result to our result object
                if isinstance(api_result, dict):
                    for key, value in api_result.items():
                        if key not in ["error", "items", "response"]:
                            result[key] = value
                
                # Apply feedback if available
                if has_feedback:
                    feedback = st.session_state.feedback_data[feedback_key]
                    # Merge feedback with result, prioritizing feedback
                    for key, value in feedback.items():
                        result[key] = value
        else:
            # Freeform extraction
            # Get prompt based on document type if available
            prompt = st.session_state.metadata_config["freeform_prompt"]
            
            # Check if we have a document type and a specific prompt for it
            if document_type and "document_type_prompts" in st.session_state.metadata_config:
                document_type_prompt = st.session_state.metadata_config["document_type_prompts"].get(document_type)
                if document_type_prompt:
                    prompt = document_type_prompt
                    logger.info(f"Using document type specific prompt for {document_type}")
                else:
                    logger.info(f"No specific prompt found for document type {document_type}, using general prompt")
            
            logger.info(f"Using freeform extraction with prompt: {prompt[:30]}...")
            
            # Use real API call
            api_result = extraction_functions["extract_freeform_metadata"](
                file_id=file_id,
                prompt=prompt,
                ai_model=st.session_state.metadata_config["ai_model"]
            )
            
            # Extract structured data from the API response
            structured_data = extract_structured_data_from_response(api_result)
            
            # Create a clean result object with the structured data
            result = structured_data
            
            # If no structured data was found, include the raw response for debugging
            if not structured_data and isinstance(api_result, dict):
                result["_raw_response"] = api_result
            
            # Apply feedback if available
            if has_feedback:
                feedback = st.session_state.feedback_data[feedback_key]
                # For freeform, we might have feedback on key-value pairs
                for key, value in feedback.items():
                    result[key] = value
        
        # Check for errors
        if isinstance(api_result, dict) and "error" in api_result:
            logger.error(f"Error processing file {file_name}: {api_result['error']}")
            return {
                "success": False,
                "error": api_result["error"]
            }
        
        logger.info(f"Successfully processed file: {file_name}")
        return {
            "success": True,
            "data": result
        }
    
    except Exception as e:
        # Log error
        logger.error(f"Error processing file {file['name']} ({file['id']}): {str(e)}")
        
        # Return error
        return {
            "success": False,
            "error": str(e)
        }

def get_extraction_functions():
    """
    Get extraction functions based on configuration
    
    Returns:
        dict: Dictionary of extraction functions
    """
    try:
        # Import metadata extraction function
        from modules.metadata_extraction import metadata_extraction
        
        # Get extraction functions
        extraction_functions = metadata_extraction()
        
        # Return functions
        return extraction_functions
    except ImportError as e:
        logger.error(f"Error importing extraction functions: {str(e)}")
        st.error(f"Error importing extraction functions: {str(e)}")
        return {
            "extract_freeform_metadata": lambda file_id, **kwargs: {"error": "Extraction function not available"},
            "extract_structured_metadata": lambda file_id, **kwargs: {"error": "Extraction function not available"}
        }
