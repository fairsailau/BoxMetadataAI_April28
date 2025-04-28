import streamlit as st
import time
import logging
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

from modules.api_client import BoxAPIClient
# FIX: Import the specific function needed, not the non-existent class
from modules.session_state_manager import initialize_state

# Configure logging
logger = logging.getLogger(__name__)

# Initialize session state keys related to processing
# FIX: Call the imported function directly
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
initialize_state("extraction_results", {})
initialize_state("application_state", {
    "is_applying": False,
    "progress": 0,
    "total_files": 0,
    "errors": {},
    "status_message": "Ready to apply",
    "cancel_requested": False
})

def display_processing_controls():
    """Displays UI controls for batch processing settings."""
    with st.expander("Batch Processing Controls", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            batch_size = st.number_input(
                "Batch Size (Max 50)", 
                min_value=1, 
                max_value=50, 
                value=st.session_state.metadata_config.get("batch_size", 5), 
                step=1,
                key="batch_size_input",
                help="Number of files to process in each batch API call (max 50 for Box AI)."
            )
            st.session_state.metadata_config["batch_size"] = batch_size
            
            use_schema_cache = st.checkbox(
                "Use Template Schema Cache", 
                value=st.session_state.metadata_config.get("use_template_cache", True),
                key="use_schema_cache_input",
                help="Cache template schemas to speed up processing. Disable if schemas change frequently."
            )
            st.session_state.metadata_config["use_template_cache"] = use_schema_cache

        with col2:
            processing_mode = st.selectbox(
                "Processing Mode", 
                ["Parallel", "Sequential"], 
                index=0 if st.session_state.metadata_config.get("processing_mode", "Parallel") == "Parallel" else 1,
                key="processing_mode_input",
                help="Parallel mode processes batches concurrently (faster but uses more resources). Sequential processes one batch at a time."
            )
            st.session_state.metadata_config["processing_mode"] = processing_mode
            
            max_workers = st.number_input(
                "Max Parallel Workers", 
                min_value=1, 
                max_value=10, 
                value=st.session_state.metadata_config.get("max_workers", 5), 
                step=1,
                key="max_workers_input",
                help="Maximum number of parallel threads for batch processing or metadata application.",
                disabled=(processing_mode == "Sequential")
            )
            st.session_state.metadata_config["max_workers"] = max_workers
            
        auto_apply = st.checkbox(
            "Automatically apply metadata after extraction",
            value=st.session_state.metadata_config.get("auto_apply_metadata", False),
            key="auto_apply_metadata_input",
            help="If checked, metadata will be automatically applied to files after successful extraction."
        )
        st.session_state.metadata_config["auto_apply_metadata"] = auto_apply

def display_processing_ui(api_client: BoxAPIClient):
    """Displays the main UI for processing files."""
    st.subheader("Process Files")
    
    files_to_process = st.session_state.get("selected_files", [])
    folders_to_process = st.session_state.get("selected_folders", [])
    total_items = len(files_to_process) + len(folders_to_process)
    
    if not files_to_process and not folders_to_process:
        st.warning("Please select files or folders in the File Selection step.")
        return
        
    # --- Configuration Summary (Read-only) ---
    with st.expander("Configuration Summary"):
        st.write("**Extraction Method:**", st.session_state.metadata_config.get("extraction_method", "Not Set"))
        if st.session_state.metadata_config.get("extraction_method") == "structured":
            if st.session_state.metadata_config.get("use_template"): 
                st.write("**Using Template:**", st.session_state.metadata_config.get("template_id", "Not Set"))
            else:
                st.write("**Using Custom Fields:**", "Yes" if st.session_state.metadata_config.get("custom_fields") else "No")
        else:
            st.write("**Freeform Prompt:**", st.session_state.metadata_config.get("freeform_prompt", "Not Set"))
        st.write("**AI Model:**", st.session_state.metadata_config.get("ai_model", "Not Set"))
        st.write("**Batch Size:**", st.session_state.metadata_config.get("batch_size", 5))
        st.write("**Processing Mode:**", st.session_state.metadata_config.get("processing_mode", "Parallel"))
        if st.session_state.metadata_config.get("processing_mode", "Parallel") == "Parallel":
            st.write("**Max Workers:**", st.session_state.metadata_config.get("max_workers", 5))
        st.write("**Use Schema Cache:**", "Yes" if st.session_state.metadata_config.get("use_template_cache", True) else "No")
        st.write("**Auto Apply Metadata:**", "Yes" if st.session_state.metadata_config.get("auto_apply_metadata", False) else "No")

    # --- Selected Files/Folders Summary ---
    with st.expander("Selected Files/Folders"):
        if files_to_process:
            st.write("**Selected Files:**")
            # Use a DataFrame for better display
            # FIX: Fixed nested f-string syntax by using proper escaping
            file_names = [f.get("name", f"ID: {f.get('id', 'unknown')}") for f in files_to_process]
            st.dataframe(file_names, hide_index=True, column_config={"value": "File Name"})
        if folders_to_process:
            st.write("**Selected Folders:**")
            # FIX: Fixed nested f-string syntax by using proper escaping
            folder_names = [f.get("name", f"ID: {f.get('id', 'unknown')}") for f in folders_to_process]
            st.dataframe(folder_names, hide_index=True, column_config={"value": "Folder Name"})

    # --- Template Cache Management ---
    with st.expander("Template Cache Management"):
        if st.button("Clear Template Schema Cache"):
            api_client.clear_template_cache()
            st.success("Template schema cache cleared.")
            # Optionally clear related session state if needed
            if "document_type_to_template" in st.session_state:
                del st.session_state.document_type_to_template
            st.rerun()

    st.info(f"Ready to process {len(files_to_process)} selected files using the configured metadata extraction parameters.")

    # --- Processing Controls ---
    display_processing_controls()

    # --- Start/Cancel Buttons ---
    col1, col2 = st.columns(2)
    with col1:
        start_button_disabled = st.session_state.processing_state["is_processing"]
        if st.button("Start Processing", disabled=start_button_disabled):
            # Reset state before starting
            initialize_state("processing_state", {
                "is_processing": True,
                "progress": 0,
                "total_files": 0, # Will be updated after folder expansion
                "current_batch": 0,
                "total_batches": 0,
                "errors": {},
                "status_message": "Initializing...",
                "cancel_requested": False
            })
            initialize_state("extraction_results", {}) # Clear previous results
            initialize_state("application_state", {
                "is_applying": False,
                "progress": 0,
                "total_files": 0,
                "errors": {},
                "status_message": "Ready to apply",
                "cancel_requested": False
            })
            logger.info("Processing started by user.")
            # Trigger the processing in the background
            # We use st.rerun() to update the UI immediately, 
            # and the processing logic runs on the next script run if is_processing is True
            st.rerun()
            
    with col2:
        cancel_button_disabled = not st.session_state.processing_state["is_processing"]
        if st.button("Cancel Processing", disabled=cancel_button_disabled):
            st.session_state.processing_state["cancel_requested"] = True
            st.session_state.processing_state["status_message"] = "Cancellation requested..."
            logger.warning("Processing cancellation requested by user.")
            st.rerun()

    # --- Processing Status Display ---
    if st.session_state.processing_state["is_processing"]:
        progress = st.session_state.processing_state["progress"]
        total_files = st.session_state.processing_state["total_files"]
        status_message = st.session_state.processing_state["status_message"]
        
        if total_files > 0:
            progress_percent = int((progress / total_files) * 100)
            st.progress(progress_percent / 100, text=f"{status_message} ({progress}/{total_files} files processed)")
        else:
            st.progress(0, text=f"{status_message}")
        
        # This is where the actual processing logic is called if state indicates processing
        # This ensures processing continues across reruns until completion or cancellation
        run_batch_processing(api_client, files_to_process, folders_to_process)
        
    # --- Display Results After Processing --- 
    elif st.session_state.extraction_results or st.session_state.processing_state["errors"]:
        display_processing_results()
        
        # --- Auto-Apply Logic ---
        if st.session_state.metadata_config.get("auto_apply_metadata", False) and \
           not st.session_state.application_state["is_applying"] and \
           st.session_state.extraction_results and \
           not st.session_state.processing_state["is_processing"]: # Ensure extraction is fully done
            
            files_to_apply = []
            for file_id, metadata in st.session_state.extraction_results.items():
                 # Check if metadata is valid (not an error message)
                 if isinstance(metadata, dict) and "error" not in metadata:
                      files_to_apply.append({"file_id": file_id, "metadata": metadata})
                 else:
                      logger.warning(f"Skipping auto-apply for file {file_id} due to invalid/error metadata: {metadata}")
            
            if files_to_apply:
                logger.info(f"Starting automatic metadata application for {len(files_to_apply)} files.")
                initialize_state("application_state", {
                    "is_applying": True,
                    "progress": 0,
                    "total_files": len(files_to_apply),
                    "errors": {},
                    "status_message": "Starting automatic application...",
                    "cancel_requested": False
                })
                # Trigger application process
                run_metadata_application(api_client, files_to_apply)
                st.rerun() # Rerun to show application progress
            else:
                 logger.info("No valid results to automatically apply.")
                 # Reset auto-apply flag in config if nothing to apply? Or just log?
                 # st.session_state.metadata_config["auto_apply_metadata"] = False # Optional: prevent re-trigger

    # --- Display Application Status ---
    if st.session_state.application_state["is_applying"]:
        app_progress = st.session_state.application_state["progress"]
        app_total = st.session_state.application_state["total_files"]
        app_status = st.session_state.application_state["status_message"]
        
        if app_total > 0:
            app_progress_percent = int((app_progress / app_total) * 100)
            st.progress(app_progress_percent / 100, text=f"Applying Metadata: {app_status} ({app_progress}/{app_total} files)")
        else:
            st.progress(0, text=f"Applying Metadata: {app_status}")
            
        # Continue application process if needed (similar pattern to extraction)
        # This assumes run_metadata_application handles state updates and completion
        # If run_metadata_application is blocking, this might not be necessary
        # If it uses threads, we need to check completion status here.
        # For simplicity, assuming run_metadata_application updates state and we just display it.
        # Check if application finished in the last run
        if app_progress == app_total and app_total > 0:
             st.session_state.application_state["is_applying"] = False
             st.session_state.application_state["status_message"] = "Application complete."
             st.success(f"Metadata application complete. Success: {app_total - len(st.session_state.application_state['errors'])}. Errors: {len(st.session_state.application_state['errors'])}.")
             if st.session_state.application_state["errors"]:
                  with st.expander("Errors During Application"):
                       st.error("Some errors occurred during metadata application:")
                       st.json(st.session_state.application_state["errors"])
             st.rerun() # Rerun one last time to clear progress bar

def display_processing_results():
    """Displays the results of the metadata extraction process."""
    st.subheader("Processing Results")
    
    results = st.session_state.extraction_results
    errors = st.session_state.processing_state["errors"]
    total_processed = len(results) + len(errors)
    success_count = len(results)
    error_count = len(errors)
    
    if success_count > 0 and error_count > 0:
        st.warning(f"Extraction complete! Successfully processed {success_count} files with {error_count} errors.")
    elif success_count > 0:
        st.success(f"Extraction complete! Successfully processed {success_count} files.")
    elif error_count > 0:
        st.error(f"Extraction complete! Failed to process {error_count} files.")
    else:
        st.info("No extraction results or errors to display.")

    if errors:
        with st.expander("Errors During Extraction", expanded=True):
            error_list = []
            # Assuming errors dict maps file_id to error message/details
            # Need file names - requires mapping IDs back to names if possible
            # Let's try to get names from selected_files if available
            file_id_to_name = {f["id"]: f.get("name", f"ID: {f.get('id', 'unknown')}") for f in st.session_state.get("selected_files", [])}
            
            for file_id, error_details in errors.items():
                file_name = file_id_to_name.get(file_id, f"ID: {file_id}")
                error_message = str(error_details) # Ensure it's a string
                # Truncate long error messages for display
                if len(error_message) > 200:
                     error_message = error_message[:200] + "..."
                error_list.append({"File Name": file_name, "File ID": file_id, "Error": error_message})
            
            if error_list:
                st.dataframe(error_list, hide_index=True)
            else:
                st.write("No specific error details available.")

    # Optionally display success results (can be large)
    # Add a button or expander to view detailed success results if needed
    if results:
        if st.button("Show Successful Extraction Results"):
             st.session_state["show_results_detail"] = True
        
        if st.session_state.get("show_results_detail", False):
             with st.expander("Successful Extraction Details", expanded=True):
                  st.json(results) # Display raw JSON results
                  if st.button("Hide Results"):
                       st.session_state["show_results_detail"] = False
                       st.rerun()

def expand_folders(api_client: BoxAPIClient, folder_ids: List[str]) -> List[Dict]:
    """Recursively fetches all files within the given folder IDs."""
    all_files = []
    folders_to_scan = list(folder_ids)
    processed_folders = set()

    while folders_to_scan:
        current_folder_id = folders_to_scan.pop(0)
        if current_folder_id in processed_folders:
            continue
        processed_folders.add(current_folder_id)
        
        logger.info(f"Scanning folder ID: {current_folder_id}")
        st.session_state.processing_state["status_message"] = f"Scanning folder {current_folder_id}..."
        st.rerun() # Update UI to show scanning status
        time.sleep(0.1) # Small delay to allow UI update

        offset = 0
        limit = 100 # Box API limit per request
        while True:
            try:
                # Check for cancellation
                if st.session_state.processing_state.get("cancel_requested", False):
                    logger.warning("Folder expansion cancelled.")
                    return []
                    
                items = api_client.get_folder_items(current_folder_id, limit=limit, offset=offset, fields=["id", "type", "name"])
                
                if "error" in items:
                    logger.error(f"Error fetching items for folder {current_folder_id}: {items['error']}")
                    # Store error associated with the folder itself? Or just log?
                    st.session_state.processing_state["errors"][f"folder_{current_folder_id}"] = items["error"]
                    break # Stop processing this folder on error

                entries = items.get("entries", [])
                if not entries:
                    break # No more items in this folder

                for item in entries:
                    if item["type"] == "file":
                        # Check if file already added (e.g., selected individually)
                        if not any(f["id"] == item["id"] for f in all_files) and not any(f["id"] == item["id"] for f in st.session_state.get("selected_files", [])):
                             all_files.append(item)
                    elif item["type"] == "folder":
                        if item["id"] not in processed_folders:
                            folders_to_scan.append(item["id"])
                
                offset += len(entries)
                if offset >= items.get("total_count", 0):
                    break # Reached the end

            except Exception as e:
                logger.exception(f"Exception fetching items for folder {current_folder_id}: {e}")
                st.session_state.processing_state["errors"][f"folder_{current_folder_id}"] = str(e)
                break # Stop processing this folder on error
                
    logger.info(f"Found {len(all_files)} files in selected folders.")
    return all_files

def run_batch_processing(api_client: BoxAPIClient, selected_files: List[Dict], selected_folders: List[Dict]):
    """Manages the overall batch processing workflow."""
    
    # --- Folder Expansion Phase ---
    if selected_folders and st.session_state.processing_state["total_files"] == 0:
        folder_ids = [f["id"] for f in selected_folders]
        try:
            expanded_files = expand_folders(api_client, folder_ids)
            # Combine selected files and expanded files, ensuring uniqueness
            combined_files = list(selected_files)
            added_ids = {f["id"] for f in combined_files}
            for f in expanded_files:
                if f["id"] not in added_ids:
                    combined_files.append(f)
                    added_ids.add(f["id"])
            
            files_to_process = combined_files
            st.session_state.processing_state["total_files"] = len(files_to_process)
            if not files_to_process:
                 st.warning("No files found to process after folder expansion.")
                 st.session_state.processing_state["is_processing"] = False
                 st.session_state.processing_state["status_message"] = "No files found."
                 st.rerun()
                 return
        except Exception as e:
            logger.exception(f"Error during folder expansion: {e}")
            st.error(f"Error during folder expansion: {e}")
            st.session_state.processing_state["is_processing"] = False
            st.session_state.processing_state["status_message"] = "Error expanding folders."
            st.rerun()
            return
    elif not selected_folders and st.session_state.processing_state["total_files"] == 0:
        # Only individually selected files
        files_to_process = list(selected_files)
        st.session_state.processing_state["total_files"] = len(files_to_process)
        if not files_to_process:
             st.warning("No files selected for processing.")
             st.session_state.processing_state["is_processing"] = False
             st.session_state.processing_state["status_message"] = "No files selected."
             st.rerun()
             return
    else:
        # Processing already started, use the total_files count from state
        # Reconstruct files_to_process if needed, or assume it's handled
        # For simplicity, we assume the process picks up batches correctly
        # If state needs to be fully restored, load files_to_process from somewhere
        pass 

    # --- Batching and Processing Phase ---
    config = st.session_state.metadata_config
    batch_size = config.get("batch_size", 5)
    processing_mode = config.get("processing_mode", "Parallel")
    max_workers = config.get("max_workers", 5)
    
    # Re-fetch files if not already done (e.g., after rerun)
    # This might be inefficient if large number of files. Consider storing IDs and fetching names later.
    if "files_to_process_list" not in st.session_state:
         # Reconstruct the list based on selected + expanded (if applicable)
         if selected_folders:
              folder_ids = [f["id"] for f in selected_folders]
              expanded_files = expand_folders(api_client, folder_ids) # Need to re-expand or store result
              combined_files = list(selected_files)
              added_ids = {f["id"] for f in combined_files}
              for f in expanded_files:
                   if f["id"] not in added_ids:
                        combined_files.append(f)
                        added_ids.add(f["id"])
              st.session_state.files_to_process_list = combined_files
         else:
              st.session_state.files_to_process_list = list(selected_files)
              
    files_to_process = st.session_state.files_to_process_list
    total_files = len(files_to_process)
    st.session_state.processing_state["total_files"] = total_files # Ensure it's up-to-date
    
    total_batches = (total_files + batch_size - 1) // batch_size
    st.session_state.processing_state["total_batches"] = total_batches

    all_results = st.session_state.extraction_results
    all_errors = st.session_state.processing_state["errors"]
    processed_count = len(all_results) + len(all_errors)
    start_batch_index = processed_count // batch_size

    logger.info(f"Starting batch processing for {total_files} files in {total_batches} batches of size {batch_size}. Mode: {processing_mode}")

    if processing_mode == "Parallel":
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            current_batch_num = start_batch_index + 1
            for i in range(start_batch_index * batch_size, total_files, batch_size):
                 # Check for cancellation before submitting new batch
                 if st.session_state.processing_state.get("cancel_requested", False):
                      logger.warning("Cancellation detected, stopping submission of new batches.")
                      break
                      
                 batch_files = files_to_process[i:min(i + batch_size, total_files)]
                 st.session_state.processing_state["current_batch"] = current_batch_num
                 st.session_state.processing_state["status_message"] = f"Submitting batch {current_batch_num}/{total_batches}..."
                 logger.info(f"Submitting batch {current_batch_num}/{total_batches} for processing.")
                 # Submit batch processing task
                 future = executor.submit(_process_batch_extraction, api_client, batch_files, config)
                 futures[future] = current_batch_num
                 current_batch_num += 1
                 # Short sleep to allow UI updates and prevent overwhelming the system
                 time.sleep(0.1)
                 st.rerun() # Rerun to update UI status

            # Process completed futures as they finish
            for future in as_completed(futures):
                 batch_num = futures[future]
                 st.session_state.processing_state["status_message"] = f"Processing batch {batch_num}/{total_batches}..."
                 st.rerun() # Update UI
                 
                 try:
                      batch_results, batch_errors = future.result()
                      all_results.update(batch_results)
                      all_errors.update(batch_errors)
                      processed_count = len(all_results) + len(all_errors)
                      st.session_state.processing_state["progress"] = processed_count
                      logger.info(f"Completed batch {batch_num}/{total_batches}. Success: {len(batch_results)}, Errors: {len(batch_errors)}")
                 except Exception as exc:
                      logger.error(f"Batch {batch_num} generated an exception: {exc}")
                      # Mark all files in the conceptual batch as errors
                      # This requires knowing which files were in batch_num - complex state needed
                      # Simplified: Log a general error for the batch
                      all_errors[f"batch_{batch_num}_exception"] = str(exc)
                 finally:
                      # Update progress and rerun UI regardless of success/failure
                      st.session_state.processing_state["progress"] = len(all_results) + len(all_errors)
                      st.rerun()
                      
    else: # Sequential processing
        for i in range(start_batch_index, total_batches):
            # Check for cancellation before starting batch
            if st.session_state.processing_state.get("cancel_requested", False):
                logger.warning("Cancellation detected, stopping sequential processing.")
                break
                
            batch_num = i + 1
            st.session_state.processing_state["current_batch"] = batch_num
            st.session_state.processing_state["status_message"] = f"Processing batch {batch_num}/{total_batches}..."
            logger.info(f"Processing batch {batch_num}/{total_batches} sequentially...")
            st.rerun() # Update UI
            
            start_index = i * batch_size
            end_index = min(start_index + batch_size, total_files)
            batch_files = files_to_process[start_index:end_index]
            
            try:
                batch_results, batch_errors = _process_batch_extraction(api_client, batch_files, config)
                all_results.update(batch_results)
                all_errors.update(batch_errors)
                processed_count = len(all_results) + len(all_errors)
                st.session_state.processing_state["progress"] = processed_count
                logger.info(f"Completed batch {batch_num}/{total_batches}. Success: {len(batch_results)}, Errors: {len(batch_errors)}")
            except Exception as e:
                 logger.error(f"Batch {batch_num} generated an exception during sequential processing: {e}")
                 # Mark files in this batch as errors
                 for file in batch_files:
                      if file.get("id"):
                           all_errors[file["id"]] = f"Batch exception: {e}"
                 st.session_state.processing_state["progress"] = len(all_results) + len(all_errors)
            
            # Rerun after each batch in sequential mode too
            st.rerun()

    # --- Finalization ---
    # Check if cancellation happened during processing
    if st.session_state.processing_state.get("cancel_requested", False):
         st.session_state.processing_state["status_message"] = "Processing cancelled."
         st.warning("Processing was cancelled.")
    else:
         st.session_state.processing_state["status_message"] = "Extraction complete."
         st.success("Metadata extraction process finished.")

    # Update final state
    st.session_state.processing_state["progress"] = len(all_results) + len(all_errors)
    st.session_state.processing_state["errors"] = all_errors
    st.session_state.processing_state["is_processing"] = False
    st.session_state.extraction_results = all_results # Update main results store
    logger.info(f"Batch processing finished. Total Success: {len(all_results)}, Total Errors: {len(all_errors)}")
    # Clear the temporary list
    if "files_to_process_list" in st.session_state:
         del st.session_state.files_to_process_list
    st.rerun() # Rerun UI to show final results and potentially trigger auto-apply

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
             # FIX: Fixed nested f-string syntax by using proper escaping
             file_id_map[file["id"]] = file.get("name", f"ID: {file.get('id', 'unknown')}") # Use get for name too
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
                
                # FIX: Fetch schema to get fields
                schema_response = api_client.get_metadata_template_schema(
                    scope=template_scope_for_api, 
                    template_key=template_key,
                    use_cache=use_template_cache
                )
                
                if "error" in schema_response:
                     # Mark all files in batch as error if schema fails
                     error_msg = f"Failed to get template schema for {template_id}: {schema_response['error']}"
                     logger.error(error_msg)
                     for item in items:
                          batch_errors[item["id"]] = error_msg
                     return {}, batch_errors # Return immediately
                
                # FIX: Extract fields from schema for the API call
                template_fields = schema_response.get("fields", [])
                if not template_fields:
                     # Mark all files in batch as error if schema has no fields
                     error_msg = f"Template schema for {template_id} has no fields defined."
                     logger.error(error_msg)
                     for item in items:
                          batch_errors[item["id"]] = error_msg
                     return {}, batch_errors # Return immediately

                logger.info(f"Calling batch structured extraction for {len(items)} items using template {template_scope_for_api}.{template_key}")
                api_response = api_client.batch_extract_metadata_structured(
                    items=items,
                    template_scope=template_scope_for_api,
                    template_key=template_key,
                    fields=template_fields, # FIX: Pass the extracted fields
                    ai_model=ai_model
                )
            # Note: Batch extraction based on custom fields is not directly supported by Box AI API AFAIK.
            # Need to fall back to individual calls or use template-based for batch.
            # For now, raise error if structured without template is chosen for batch.
            elif config.get("custom_fields"):
                 error_msg = "Batch extraction with custom fields is not supported. Please use a template or freeform extraction."
                 logger.error(error_msg)
                 for item in items:
                      batch_errors[item["id"]] = error_msg
                 return {}, batch_errors # Return immediately
                 # raise NotImplementedError(error_msg) # Or raise
            else:
                 error_msg = "Structured extraction selected but no template ID provided in config."
                 logger.error(error_msg)
                 for item in items:
                      batch_errors[item["id"]] = error_msg
                 return {}, batch_errors # Return immediately
                 # raise ValueError(error_msg) # Or raise
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
            # Include more details if available (e.g., from Box error response)
            if isinstance(api_response, dict):
                 details = api_response.get("message") or api_response.get("code") or api_response.get("request_id")
                 if details:
                      error_msg += f" (Details: {details})"
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
                    # Include Box error details if present
                    box_error_info = entry.get("error_details") # Assuming Box might add this
                    if box_error_info:
                         error_detail += f" (Details: {box_error_info})"
                    logger.warning(f"Error processing file ID {file_id} in batch: {error_detail}")
                    batch_errors[file_id] = error_detail
                elif entry.get("status") == "success":
                    # Extract the actual metadata payload
                    metadata = extract_structured_data_from_response(entry) # Use helper function

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
            # Handle lists/dicts specifically? Box might support JSON types for some fields.
            # For now, simple string conversion.
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

def run_metadata_application(api_client: BoxAPIClient, files_to_apply: List[Dict]):
    """Applies metadata to the given files, potentially in parallel."""
    
    total_files = len(files_to_apply)
    if total_files == 0:
        st.session_state.application_state["is_applying"] = False
        st.session_state.application_state["status_message"] = "No files to apply metadata to."
        logger.info("No files provided for metadata application.")
        st.rerun()
        return

    st.session_state.application_state["total_files"] = total_files
    st.session_state.application_state["progress"] = 0
    st.session_state.application_state["errors"] = {}
    st.session_state.application_state["status_message"] = "Applying metadata..."
    st.rerun() # Update UI

    config = st.session_state.metadata_config
    processing_mode = config.get("processing_mode", "Parallel") # Use same mode for consistency
    max_workers = config.get("max_workers", 5)
    
    application_errors = {}
    completed_count = 0

    if processing_mode == "Parallel":
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(apply_metadata_single, api_client, item): item["file_id"] for item in files_to_apply}
            
            for future in as_completed(futures):
                file_id = futures[future]
                try:
                    result = future.result()
                    if not result.get("success"): 
                        application_errors[file_id] = result.get("error", "Unknown application error")
                except Exception as exc:
                    logger.error(f"Metadata application for file {file_id} generated an exception: {exc}")
                    application_errors[file_id] = str(exc)
                finally:
                    completed_count += 1
                    st.session_state.application_state["progress"] = completed_count
                    st.session_state.application_state["status_message"] = f"Applied {completed_count}/{total_files}..."
                    st.rerun() # Update progress
    else: # Sequential
        for item in files_to_apply:
            file_id = item["file_id"]
            st.session_state.application_state["status_message"] = f"Applying to {file_id}..."
            st.rerun() # Update status
            try:
                result = apply_metadata_single(api_client, item)
                if not result.get("success"): 
                    application_errors[file_id] = result.get("error", "Unknown application error")
            except Exception as exc:
                logger.error(f"Metadata application for file {file_id} generated an exception: {exc}")
                application_errors[file_id] = str(exc)
            finally:
                completed_count += 1
                st.session_state.application_state["progress"] = completed_count
                # Rerun needed here too for sequential progress update
                st.rerun()

    # --- Finalize Application --- 
    st.session_state.application_state["errors"] = application_errors
    st.session_state.application_state["is_applying"] = False # Mark as finished
    # Final status message and rerun handled in display_processing_ui
    logger.info(f"Metadata application finished. Success: {total_files - len(application_errors)}, Errors: {len(application_errors)}")
    # No final rerun here, let display_processing_ui handle it


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
