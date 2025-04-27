import streamlit as st
import time
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
import json
import concurrent.futures
from modules.api_client import BoxAPIClient # Assuming api_client is in the same directory

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
        
        # Ensure metadata_config exists and is minimally valid
        if "metadata_config" not in st.session_state:
             st.session_state.metadata_config = {
                 "extraction_method": "freeform", # Default
                 "freeform_prompt": "Extract key details.", # Default
                 "ai_model": "google__gemini_2_0_flash_001", # Default
                 "batch_size": 5, # Default
                 "use_template": False,
                 "template_id": None,
                 "custom_fields": []
             }

        if st.session_state.metadata_config["extraction_method"] == "structured" and \
           not st.session_state.metadata_config.get("use_template") and \
           not st.session_state.metadata_config.get("custom_fields"):
            st.warning("Structured metadata configuration is incomplete. Please configure a template or custom fields.")
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
                "current_batch": 0,
                "total_batches": 0,
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
                # Batch size control (Max 50 for Box AI batch endpoints)
                batch_size = st.number_input(
                    "Batch Size (Max 50)",
                    min_value=1,
                    max_value=50, # Box AI batch limit
                    value=st.session_state.metadata_config.get("batch_size", 10),
                    key="batch_size_input"
                )
                st.session_state.metadata_config["batch_size"] = batch_size
                
                # Template caching control
                use_template_cache = st.checkbox(
                    "Use Template Schema Cache",
                    value=True,
                    help="Cache template schemas to reduce API calls",
                    key="use_template_cache_checkbox"
                )
                st.session_state.metadata_config["use_template_cache"] = use_template_cache
            
            with col2:
                # Processing mode (Parallel for batches)
                processing_mode = st.selectbox(
                    "Processing Mode",
                    options=["Sequential", "Parallel"], # Controls batch submission
                    index=1, # Default to Parallel
                    key="processing_mode_input"
                )
                st.session_state.processing_state["processing_mode"] = processing_mode
                max_workers = st.number_input(
                    "Max Parallel Workers",
                    min_value=1,
                    max_value=10,
                    value=5,
                    key="max_workers_input",
                    help="Number of batches to process concurrently in Parallel mode."
                )
                st.session_state.processing_state["max_workers"] = max_workers

        
        # Auto-apply metadata option
        auto_apply_metadata = st.checkbox(
            "Automatically apply metadata after extraction",
            value=True,
            help="If checked, extracted metadata will be automatically applied to files after processing",
            key="auto_apply_metadata_checkbox"
        )
        st.session_state.processing_state["auto_apply_metadata"] = auto_apply_metadata
        
        # Template management (No changes needed here)
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
                    # Update batch size from loaded template if present
                    batch_size = st.session_state.metadata_config.get("batch_size", 10)
                    st.success(f"Template '{selected_template}' loaded successfully!")
                    st.rerun() # Rerun to reflect loaded config in UI
            else:
                st.info("No saved templates yet")
        
        # Display configuration summary (No changes needed here)
        with st.expander("Configuration Summary"):
            st.write("#### Extraction Method")
            st.write(f"Method: {st.session_state.metadata_config.get('extraction_method', 'freeform').capitalize()}")
            
            if st.session_state.metadata_config.get("extraction_method") == "structured":
                if st.session_state.metadata_config.get("use_template"):
                    st.write(f"Using template: Template ID {st.session_state.metadata_config.get('template_id')}")
                elif st.session_state.metadata_config.get("custom_fields"):
                    st.write(f"Using {len(st.session_state.metadata_config['custom_fields'])} custom fields")
                    for i, field in enumerate(st.session_state.metadata_config['custom_fields']):
                        st.write(f"- {field.get('display_name', field.get('name', ''))} ({field.get('type', 'string')})")
                else:
                     st.warning("Structured method selected but no template or custom fields configured.")
            else: # Freeform
                st.write("Freeform prompt:")
                st.write(f"> {st.session_state.metadata_config.get('freeform_prompt')}")
            
            st.write(f"AI Model: {st.session_state.metadata_config.get('ai_model')}")
            st.write(f"Batch Size: {st.session_state.metadata_config.get('batch_size')}")
            st.write(f"Template Caching: {'Enabled' if st.session_state.metadata_config.get('use_template_cache', True) else 'Disabled'}")
            st.write(f"Processing Mode: {st.session_state.processing_state.get('processing_mode', 'Parallel')}")
            if st.session_state.processing_state.get("processing_mode") == "Parallel":
                 st.write(f"Max Parallel Workers: {st.session_state.processing_state.get('max_workers', 5)}")

        
        # Display selected files (No changes needed here)
        with st.expander("Selected Files"):
            for file in st.session_state.selected_files:
                st.write(f"- {file['name']} (Type: {file['type']})")
        
        # Template cache management
        with st.expander("Template Cache Management"):
            st.write("#### Template Schema Cache")
            
            # Get API client instance for cache operations
            api_client = BoxAPIClient(st.session_state.client)
            
            # Display cache status
            cache_size = len(getattr(api_client, 'template_cache', {}))
            st.write(f"Current cache size: {cache_size} templates")
            
            # Clear cache button
            if st.button("Clear Template Cache", key="clear_template_cache_button"):
                api_client.clear_template_cache()
                st.success("Template cache cleared successfully!")
        
        # Process files button
        col1, col2 = st.columns(2)
        
        with col1:
            start_button = st.button(
                "Start Processing",
                disabled=st.session_state.processing_state.get("is_processing", False),
                use_container_width=True,
                key="start_processing_button"
            )
        
        with col2:
            cancel_button = st.button(
                "Cancel Processing",
                disabled=not st.session_state.processing_state.get("is_processing", False),
                use_container_width=True,
                key="cancel_processing_button"
            )
        
        # Progress tracking
        progress_container = st.container()
        
        # Process files
        if start_button:
            # Reset processing state
            total_files_to_process = len(st.session_state.selected_files)
            total_batches = (total_files_to_process + batch_size - 1) // batch_size
            st.session_state.processing_state = {
                "is_processing": True,
                "processed_files": 0,
                "total_files": total_files_to_process,
                "current_batch": 0,
                "total_batches": total_batches,
                "results": {},
                "errors": {},
                # Retries handled by API client
                "processing_mode": processing_mode,
                "max_workers": max_workers,
                "auto_apply_metadata": auto_apply_metadata,
                "metadata_applied": False, # Reset applied flag
                "visualization_data": {}
            }
            
            # Reset extraction results
            st.session_state.extraction_results = {}
            
            # Get API client instance
            api_client = BoxAPIClient(st.session_state.client)
            
            # Process files with progress tracking
            process_files_in_batches(
                api_client,
                st.session_state.selected_files,
                st.session_state.metadata_config,
                batch_size=batch_size,
                processing_mode=processing_mode,
                max_workers=max_workers
            )
            # Note: process_files_in_batches will call st.rerun() internally on completion/error
        
        # Cancel processing
        if cancel_button and st.session_state.processing_state.get("is_processing", False):
            st.session_state.processing_state["is_processing"] = False
            # TODO: Implement cancellation mechanism for ongoing futures if parallel
            st.warning("Processing cancellation requested (may take a moment to stop)...")
            # We might need a more robust cancellation mechanism, e.g., a flag checked by workers
        
        # Display processing progress
        if st.session_state.processing_state.get("is_processing", False):
            with progress_container:
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Update progress
                processed_files = st.session_state.processing_state["processed_files"]
                total_files = st.session_state.processing_state["total_files"]
                current_batch = st.session_state.processing_state["current_batch"]
                total_batches = st.session_state.processing_state["total_batches"]
                
                # Calculate progress
                progress = processed_files / total_files if total_files > 0 else 0
                
                # Update progress bar
                progress_bar.progress(progress)
                
                # Update status text
                status_text.text(f"Processing Batch {current_batch}/{total_batches}... Processed Files: {processed_files}/{total_files}")

        
        # Display processing results (after processing is complete)
        if not st.session_state.processing_state.get("is_processing", False) and \
           ("results" in st.session_state.processing_state or "errors" in st.session_state.processing_state) and \
           (st.session_state.processing_state["results"] or st.session_state.processing_state["errors"]):
            
            st.write("### Processing Results")
            
            # Display success/error summary
            success_files = len(st.session_state.processing_state.get("results", {}))
            error_files = len(st.session_state.processing_state.get("errors", {}))
            total_processed = success_files + error_files
            
            if error_files == 0 and total_processed > 0:
                st.success(f"Extraction complete! Successfully processed {success_files} files.")
            elif total_processed > 0:
                st.warning(f"Extraction complete! Successfully processed {success_files} files with {error_files} errors.")
            else:
                st.info("No files were processed or processing was cancelled.")

            # Display errors if any
            if st.session_state.processing_state.get("errors"):
                st.write("### Errors During Extraction")
                errors_df_data = []
                for file_id, error_info in st.session_state.processing_state["errors"].items():
                    file_name = next((f["name"] for f in st.session_state.selected_files if f["id"] == file_id), f"ID: {file_id}")
                    error_message = str(error_info.get("error", error_info)) if isinstance(error_info, dict) else str(error_info)
                    errors_df_data.append({"File Name": file_name, "File ID": file_id, "Error": error_message})
                
                if errors_df_data:
                    errors_df = pd.DataFrame(errors_df_data)
                    st.dataframe(errors_df, use_container_width=True)
            
            # Auto-apply metadata if enabled and results exist
            if st.session_state.processing_state.get("auto_apply_metadata", False) and \
               not st.session_state.processing_state.get("metadata_applied", False) and \
               st.session_state.processing_state.get("results"):
                
                st.write("### Applying Metadata")
                st.info("Automatically applying extracted metadata to files...")
                
                # Import apply_metadata function
                from modules.direct_metadata_application_enhanced_fixed import apply_metadata_to_file_direct
                
                # Get API client instance (reuse if possible)
                api_client = BoxAPIClient(st.session_state.client)
                
                # Create a progress bar for metadata application
                apply_progress_bar = st.progress(0)
                apply_status_text = st.empty()
                
                # Initialize counters
                apply_success_count = 0
                apply_error_count = 0
                apply_errors = {}
                
                # Prepare data for application
                results_to_apply = st.session_state.processing_state.get("results", {})
                total_files_to_apply = len(results_to_apply)
                apply_items = []
                for file_id, extracted_data in results_to_apply.items():
                    file_name = next((f["name"] for f in st.session_state.selected_files if f["id"] == file_id), f"ID: {file_id}")
                    apply_items.append({"file_id": file_id, "file_name": file_name, "metadata": extracted_data})

                # --- Apply metadata (can also be parallelized) ---
                # Using ThreadPoolExecutor for applying metadata in parallel
                apply_mode = st.session_state.processing_state.get("processing_mode", "Parallel")
                max_apply_workers = st.session_state.processing_state.get("max_workers", 5)
                
                if apply_mode == "Parallel" and total_files_to_apply > 1:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_apply_workers) as executor:
                        future_to_item = {executor.submit(apply_metadata_single, api_client, item): item for item in apply_items}
                        
                        for i, future in enumerate(concurrent.futures.as_completed(future_to_item)):
                            item = future_to_item[future]
                            file_name = item["file_name"]
                            apply_status_text.text(f"Applying metadata to {file_name}... ({i+1}/{total_files_to_apply})")
                            try:
                                result = future.result()
                                if result["success"]:
                                    apply_success_count += 1
                                else:
                                    apply_error_count += 1
                                    apply_errors[item["file_id"]] = result.get("error", "Unknown application error")
                            except Exception as exc:
                                logger.error(f"Error applying metadata to {item['file_id']}: {exc}")
                                apply_error_count += 1
                                apply_errors[item["file_id"]] = str(exc)
                            apply_progress_bar.progress((i + 1) / total_files_to_apply)
                else: # Sequential application
                     for i, item in enumerate(apply_items):
                        file_name = item["file_name"]
                        apply_status_text.text(f"Applying metadata to {file_name}... ({i+1}/{total_files_to_apply})")
                        result = apply_metadata_single(api_client, item)
                        if result["success"]:
                            apply_success_count += 1
                        else:
                            apply_error_count += 1
                            apply_errors[item["file_id"]] = result.get("error", "Unknown application error")
                        apply_progress_bar.progress((i + 1) / total_files_to_apply)

                # Clear progress indicators
                apply_progress_bar.empty()
                apply_status_text.empty()
                
                # Display results
                if apply_error_count == 0:
                    st.success(f"Successfully applied metadata to all {apply_success_count} files!")
                else:
                    st.warning(f"Applied metadata to {apply_success_count} files with {apply_error_count} errors.")
                    # Display application errors
                    st.write("### Errors During Metadata Application")
                    apply_errors_df_data = []
                    for file_id, error_msg in apply_errors.items():
                        file_name = next((f["name"] for f in st.session_state.selected_files if f["id"] == file_id), f"ID: {file_id}")
                        apply_errors_df_data.append({"File Name": file_name, "File ID": file_id, "Error": str(error_msg)})
                    if apply_errors_df_data:
                        apply_errors_df = pd.DataFrame(apply_errors_df_data)
                        st.dataframe(apply_errors_df, use_container_width=True)

                # Mark metadata as applied
                st.session_state.processing_state["metadata_applied"] = True
            
            # Continue button
            st.write("---")
            if st.button("Continue to View Results", key="continue_to_results_button", use_container_width=True):
                st.session_state.current_page = "View Results"
                st.rerun()
    
    except Exception as e:
        st.error(f"An unexpected error occurred in the processing page: {str(e)}")
        logger.exception("Error in process_files UI rendering")

# --- Batch Processing Logic ---

def process_files_in_batches(api_client: BoxAPIClient, files: List[Dict], config: Dict, batch_size: int, processing_mode: str, max_workers: int):
    """
    Processes files using batch API calls.
    Updates st.session_state.processing_state with results and errors.
    """
    total_files = len(files)
    batches = [files[i:i + batch_size] for i in range(0, total_files, batch_size)]
    total_batches = len(batches)
    st.session_state.processing_state["total_batches"] = total_batches
    st.session_state.processing_state["processed_files"] = 0 # Reset count
    st.session_state.processing_state["current_batch"] = 0 # Reset count

    logger.info(f"Starting batch processing for {total_files} files in {total_batches} batches of size {batch_size}. Mode: {processing_mode}")

    all_results = {}
    all_errors = {}

    def _update_progress(processed_in_batch=0):
        st.session_state.processing_state["processed_files"] += processed_in_batch
        st.session_state.processing_state["current_batch"] += 1
        # Use a callback or st.experimental_rerun if needed for smoother UI updates
        # For now, rely on the final rerun

    if processing_mode == "Parallel" and total_batches > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch_index = {executor.submit(_process_batch_extraction, api_client, batch, config): i for i, batch in enumerate(batches)}
            
            for future in concurrent.futures.as_completed(future_to_batch_index):
                batch_index = future_to_batch_index[future]
                batch_files = batches[batch_index]
                try:
                    batch_results, batch_errors = future.result()
                    all_results.update(batch_results)
                    all_errors.update(batch_errors)
                    _update_progress(len(batch_files))
                    logger.info(f"Completed batch {batch_index + 1}/{total_batches}. Success: {len(batch_results)}, Errors: {len(batch_errors)}")
                except Exception as exc:
                    logger.error(f"Batch {batch_index + 1} failed entirely: {exc}")
                    # Mark all files in this batch as errors
                    for file in batch_files:
                        all_errors[file["id"]] = f"Batch processing error: {exc}"
                    _update_progress(len(batch_files))
    else: # Sequential processing of batches
        for i, batch in enumerate(batches):
            if not st.session_state.processing_state.get("is_processing", False):
                logger.warning("Processing cancelled during sequential batch execution.")
                break
            st.session_state.processing_state["current_batch"] = i + 1
            logger.info(f"Processing batch {i + 1}/{total_batches} sequentially...")
            try:
                batch_results, batch_errors = _process_batch_extraction(api_client, batch, config)
                all_results.update(batch_results)
                all_errors.update(batch_errors)
                st.session_state.processing_state["processed_files"] += len(batch)
                logger.info(f"Completed batch {i + 1}/{total_batches}. Success: {len(batch_results)}, Errors: {len(batch_errors)}")
            except Exception as exc:
                logger.error(f"Batch {i + 1} failed entirely: {exc}")
                for file in batch:
                    all_errors[file["id"]] = f"Batch processing error: {exc}"
                st.session_state.processing_state["processed_files"] += len(batch)

    # Update final state
    st.session_state.processing_state["results"] = all_results
    st.session_state.processing_state["errors"] = all_errors
    st.session_state.processing_state["is_processing"] = False
    st.session_state.extraction_results = all_results # Update main results store
    logger.info(f"Batch processing finished. Total Success: {len(all_results)}, Total Errors: {len(all_errors)}")
    st.rerun() # Rerun UI to show final results

def _process_batch_extraction(api_client: BoxAPIClient, batch_files: List[Dict], config: Dict) -> tuple:
    """
    Processes a single batch of files using the appropriate Box AI batch API.
    
    Returns:
        tuple: (batch_results, batch_errors)
    """
    batch_results = {}
    batch_errors = {}
    items = []
    file_id_map = {}
    for file in batch_files:
        # FIX: Check if the item has an 'id' and is NOT a 'folder'
        # Box API expects 'type': 'file' in the request body, regardless of the actual file type (pdf, docx etc.)
        if file.get("id") and file.get("type") != "folder":
             # Always set type to 'file' for the API call
             items.append({"id": file["id"], "type": "file"})
             file_id_map[file["id"]] = file.get("name", f"ID: {file['id']}") # Use get for name too
        else:
             logger.warning(f"Skipping invalid or non-file item in batch: {file}")

    if not items:
        logger.warning("Empty batch after filtering, skipping API call.")
        return {}, {}

    extraction_method = config.get("extraction_method", "freeform")
    ai_model = config.get("ai_model", "google__gemini_2_0_flash_001")
    use_template_cache = config.get("use_template_cache", True)
    api_response = None

    try:
        if extraction_method == "structured":
            if config.get("use_template") and config.get("template_id"):
                template_id = config["template_id"]
                # Parse template ID (assuming format scope_id_templateKey)
                parts = template_id.split("_")
                scope = parts[0]
                enterprise_id = parts[1] if len(parts) > 1 else ""
                template_key = parts[-1] if len(parts) > 2 else template_id
                # Use the actual scope (e.g., enterprise_12345) for the API call
                template_scope_for_api = f"{scope}_{enterprise_id}" if enterprise_id else scope
                
                # Fetch schema once (API client handles caching)
                schema_response = api_client.get_metadata_template_schema(
                    scope=template_scope_for_api, 
                    template_key=template_key,
                    use_cache=use_template_cache
                )
                
                if "error" in schema_response:
                     raise ValueError(f"Failed to get template schema for {template_id}: {schema_response['error']}")
                
                logger.info(f"Calling batch structured extraction for {len(items)} items using template {template_scope_for_api}.{template_key}")
                api_response = api_client.batch_extract_metadata_structured(
                    items=items,
                    template_scope=template_scope_for_api,
                    template_key=template_key,
                    ai_model=ai_model
                )
            # Note: Batch extraction based on custom fields is not directly supported by Box AI API AFAIK.
            # Need to fall back to individual calls or use template-based for batch.
            # For now, raise error if structured without template is chosen for batch.
            elif config.get("custom_fields"):
                 raise NotImplementedError("Batch extraction with custom fields is not supported. Please use a template or freeform extraction.")
            else:
                 raise ValueError("Structured extraction selected but no template ID provided in config.")
        else: # Freeform
            prompt = config.get("freeform_prompt", "Extract key details.")
            logger.info(f"Calling batch freeform extraction for {len(items)} items.")
            api_response = api_client.batch_extract_metadata_freeform(
                items=items,
                prompt=prompt,
                ai_model=ai_model
            )

        # Process the batch response
        if not api_response or "error" in api_response:
            error_msg = api_response.get("error", "Unknown API error during batch extraction") if isinstance(api_response, dict) else "Invalid API response"
            logger.error(f"Batch extraction API call failed: {error_msg}")
            # Mark all files in this batch as errors
            for item in items:
                batch_errors[item["id"]] = error_msg
        elif "entries" in api_response and isinstance(api_response["entries"], list):
            # Successful batch response
            for entry in api_response["entries"]:
                file_id = entry.get("item", {}).get("id")
                if not file_id:
                    logger.warning(f"Skipping entry with missing item ID: {entry}")
                    continue
                
                if entry.get("status") == "error":
                    error_detail = entry.get("error", "Unknown error for this item in batch")
                    logger.warning(f"Error processing file ID {file_id} in batch: {error_detail}")
                    batch_errors[file_id] = error_detail
                elif entry.get("status") == "success":
                    # Extract the actual metadata payload
                    metadata = entry.get("answer", {}) # Structured often in answer
                    if not metadata and isinstance(entry.get("results"), dict):
                         metadata = entry["results"] # Freeform might be here
                    elif not metadata and isinstance(entry.get("results"), str): # Handle JSON string case
                         try:
                              parsed_results = json.loads(entry["results"])
                              if isinstance(parsed_results, dict):
                                   metadata = parsed_results
                         except json.JSONDecodeError:
                              metadata = {"extracted_text": entry["results"]} # Fallback
                    elif not metadata:
                         metadata = entry # Fallback to whole entry if no clear payload

                    # Apply feedback if available (still relevant per file)
                    feedback_key = f"{file_id}_{extraction_method}"
                    if feedback_key in st.session_state.get("feedback_data", {}):
                        feedback = st.session_state.feedback_data[feedback_key]
                        logger.info(f"Applying feedback data for file: {file_id_map.get(file_id, file_id)}")
                        if isinstance(metadata, dict):
                             metadata.update(feedback) # Merge feedback, prioritizing feedback
                        else:
                             metadata = feedback # Overwrite if original wasn't dict

                    batch_results[file_id] = metadata
                else:
                    logger.warning(f"Unknown status for file ID {file_id} in batch: {entry.get('status')}")
                    batch_errors[file_id] = f"Unknown status: {entry.get('status')}"
        else:
            logger.error(f"Unexpected batch API response format: {api_response}")
            for item in items:
                batch_errors[item["id"]] = "Unexpected API response format"

    except Exception as e:
        logger.exception(f"Error processing batch: {e}")
        # Mark all files in this batch as errors
        for item in items:
            batch_errors[item["id"]] = f"Error during batch processing: {e}"
            
    return batch_results, batch_errors

# --- Helper for applying metadata (can be parallelized) ---
def apply_metadata_single(api_client: BoxAPIClient, item: Dict) -> Dict:
    """
    Applies metadata to a single file.
    Designed to be called in parallel.
    """
    file_id = item["file_id"]
    metadata = item["metadata"]
    
    # Determine template scope/key (assuming it's consistent or derivable)
    # This part might need refinement based on how templates are managed/selected
    template_scope = "enterprise" # Default or get from config/context
    template_key = None
    if st.session_state.metadata_config.get("extraction_method") == "structured" and \
       st.session_state.metadata_config.get("use_template") and \
       st.session_state.metadata_config.get("template_id"):
        
        template_id = st.session_state.metadata_config["template_id"]
        parts = template_id.split("_")
        scope_part = parts[0]
        enterprise_id = parts[1] if len(parts) > 1 else ""
        template_key = parts[-1] if len(parts) > 2 else template_id
        template_scope = f"{scope_part}_{enterprise_id}" if enterprise_id else scope_part

    if not template_key:
         logger.warning(f"Cannot apply metadata for file {file_id}: Template key not determined.")
         return {"success": False, "error": "Template key not determined for application"}

    # Clean metadata - Box API often expects string values
    cleaned_metadata = {}
    for k, v in metadata.items():
        # Basic cleaning: convert non-strings/numbers/booleans to string
        if not isinstance(v, (str, int, float, bool)):
            cleaned_metadata[k] = str(v)
        else:
            cleaned_metadata[k] = v
            
    logger.info(f"Applying metadata to file {file_id} using template {template_scope}.{template_key}")
    result = api_client.apply_metadata(
        file_id=file_id,
        metadata=cleaned_metadata,
        scope=template_scope,
        template_key=template_key
    )
    
    if "error" in result:
        logger.error(f"Failed to apply metadata to {file_id}: {result['error']}")
        return {"success": False, "error": result.get("error", "Failed to apply metadata")}
    else:
        logger.info(f"Successfully applied metadata to {file_id}")
        return {"success": True}


# --- Helper function to extract structured data from API response ---
def extract_structured_data_from_response(response):
    """
    Extract structured data from various possible response structures
    (Used for parsing results within batch or individual responses)
    
    Args:
        response (dict): API response or entry from batch response
        
    Returns:
        dict: Extracted structured data (key-value pairs)
    """
    structured_data = {}
    
    # Log the response structure for debugging
    # logger.debug(f"Parsing response structure: {json.dumps(response, indent=2) if isinstance(response, dict) else str(response)}")
    
    if isinstance(response, dict):
        # Check for answer field (contains structured data in JSON format)
        if "answer" in response and isinstance(response["answer"], dict):
            structured_data = response["answer"]
            # logger.debug(f"Found structured data in 'answer' field: {structured_data}")
            return structured_data
        
        # Check for answer field as string (JSON string)
        if "answer" in response and isinstance(response["answer"], str):
            try:
                answer_data = json.loads(response["answer"])
                if isinstance(answer_data, dict):
                    structured_data = answer_data
                    # logger.debug(f"Found structured data in 'answer' field (JSON string): {structured_data}")
                    return structured_data
            except json.JSONDecodeError:
                logger.warning(f"Could not parse 'answer' field as JSON: {response['answer']}")
        
        # Check for results field (used in some AI responses)
        if "results" in response and isinstance(response["results"], dict):
             structured_data = response["results"]
             # logger.debug(f"Found structured data in 'results' field: {structured_data}")
             return structured_data
        elif "results" in response and isinstance(response["results"], str):
             try:
                  results_data = json.loads(response["results"])
                  if isinstance(results_data, dict):
                       structured_data = results_data
                       # logger.debug(f"Found structured data in 'results' field (JSON string): {structured_data}")
                       return structured_data
             except json.JSONDecodeError:
                  logger.warning(f"Could not parse 'results' field as JSON: {response['results']}")
                  # Fallback: Treat the string as a single value if parsing fails
                  structured_data = {"extracted_text": response["results"]}
                  return structured_data

        # Fallback: Check for key-value pairs directly in response (less reliable)
        # Avoid common metadata keys
        avoid_keys = ["error", "items", "response", "item_collection", "entries", "type", "id", "sequence_id", "status", "item", "completion_reason"]
        temp_data = {}
        for key, value in response.items():
            if key not in avoid_keys:
                temp_data[key] = value
        if temp_data: # Only use if we found something
             structured_data = temp_data
             # logger.debug(f"Found potential structured data directly in response keys: {structured_data}")
             return structured_data

    # If we couldn't find structured data, return empty dict
    if not structured_data:
        logger.warning("Could not find structured data in response")
    
    return structured_data

# --- Get Extraction Functions (Placeholder - Adapt if needed) ---
def get_extraction_functions():
    """
    Returns a dictionary of extraction functions (placeholder).
    In the batch implementation, the logic is mostly within _process_batch_extraction.
    This function might become less relevant or removed.
    """
    # This function might need to be adapted or removed as the core logic
    # now resides in _process_batch_extraction which directly uses the api_client.
    return {
        # "extract_structured_metadata": api_client.some_structured_method, # Example
        # "extract_freeform_metadata": api_client.some_freeform_method  # Example
    }
