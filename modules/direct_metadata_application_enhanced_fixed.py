import streamlit as st
import logging
import json
from boxsdk import Client

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_metadata_format(metadata_values):
    """
    Fix the metadata format by converting string representations of dictionaries
    to actual Python dictionaries.
    
    Args:
        metadata_values (dict): The original metadata values dictionary
        
    Returns:
        dict: A new dictionary with properly formatted metadata values
    """
    formatted_metadata = {}
    
    for key, value in metadata_values.items():
        # If the value is a string that looks like a dictionary, parse it
        if isinstance(value, str) and value.startswith('{') and value.endswith('}'):
            try:
                # Replace single quotes with double quotes for JSON compatibility
                json_compatible_str = value.replace("'", '"')
                # Parse the string representation into a proper Python dictionary
                parsed_value = json.loads(json_compatible_str)
                formatted_metadata[key] = parsed_value
            except json.JSONDecodeError:
                # If parsing fails, keep the original string value
                formatted_metadata[key] = value
        else:
            # For non-dictionary string values, keep as is
            formatted_metadata[key] = value
    
    return formatted_metadata

def flatten_metadata_for_template(metadata_values):
    """
    Flatten the metadata structure by extracting fields from the 'answer' object
    and placing them directly at the top level to match the template structure.
    
    Args:
        metadata_values (dict): The metadata values with nested objects
        
    Returns:
        dict: A flattened dictionary with fields at the top level
    """
    flattened_metadata = {}
    
    # Check if 'answer' exists and is a dictionary
    if 'answer' in metadata_values and isinstance(metadata_values['answer'], dict):
        # Extract fields from the 'answer' object and place them at the top level
        for key, value in metadata_values['answer'].items():
            flattened_metadata[key] = value
    else:
        # If there's no 'answer' object, use the original metadata
        flattened_metadata = metadata_values.copy()
    
    # Remove any non-template fields that shouldn't be sent to Box API
    # These are fields that are used internally but not part of the template
    keys_to_remove = ['ai_agent_info', 'created_at', 'completion_reason', 'answer']
    for key in keys_to_remove:
        if key in flattened_metadata:
            del flattened_metadata[key]
    
    return flattened_metadata

def filter_confidence_fields(metadata_values):
    """
    Filter out confidence score fields from metadata values.
    
    Args:
        metadata_values (dict): The original metadata values dictionary
        
    Returns:
        dict: A new dictionary with confidence fields removed
    """
    filtered_metadata = {}
    
    for key, value in metadata_values.items():
        # Skip any keys that end with "_confidence" as these are confidence score fields
        if not key.endswith("_confidence"):
            filtered_metadata[key] = value
    
    return filtered_metadata

def apply_metadata_direct():
    """
    Direct approach to apply metadata to Box files with comprehensive fixes
    for session state alignment and metadata extraction
    """
    st.title("Apply Metadata")
    
    # Debug checkbox
    debug_mode = st.sidebar.checkbox("Debug Session State", key="debug_checkbox")
    if debug_mode:
        st.sidebar.write("### Session State Debug")
        st.sidebar.write("**Session State Keys:**")
        st.sidebar.write(list(st.session_state.keys()))
        
        if "client" in st.session_state:
            st.sidebar.write("**Client:** Available")
            try:
                user = st.session_state.client.user().get()
                st.sidebar.write(f"**Authenticated as:** {user.name}")
            except Exception as e:
                st.sidebar.write(f"**Client Error:** {str(e)}")
        else:
            st.sidebar.write("**Client:** Not available")
            
        if "processing_state" in st.session_state:
            st.sidebar.write("**Processing State Keys:**")
            st.sidebar.write(list(st.session_state.processing_state.keys()))
            
            # Dump the first processing result for debugging
            if st.session_state.processing_state:
                first_key = next(iter(st.session_state.processing_state))
                st.sidebar.write(f"**First Processing Result ({first_key}):**")
                st.sidebar.json(st.session_state.processing_state[first_key])
    
    # Check if client exists directly
    if 'client' not in st.session_state:
        st.error("Box client not found. Please authenticate first.")
        if st.button("Go to Authentication", key="go_to_auth_btn"):
            st.session_state.current_page = "Home"  # Assuming Home page has authentication
            st.rerun()
        return
    
    # Get client directly
    client = st.session_state.client
    
    # Verify client is working
    try:
        user = client.user().get()
        logger.info(f"Verified client authentication as {user.name}")
        st.success(f"Authenticated as {user.name}")
    except Exception as e:
        logger.error(f"Error verifying client: {str(e)}")
        st.error(f"Authentication error: {str(e)}. Please re-authenticate.")
        if st.button("Go to Authentication", key="go_to_auth_error_btn"):
            st.session_state.current_page = "Home"
            st.rerun()
        return
    
    # Check if processing state exists
    if "processing_state" not in st.session_state or not st.session_state.processing_state:
        st.warning("No processing results available. Please process files first.")
        if st.button("Go to Process Files", key="go_to_process_files_btn"):
            st.session_state.current_page = "Process Files"
            st.rerun()
        return
    
    # Debug the structure of processing_state
    processing_state = st.session_state.processing_state
    logger.info(f"Processing state keys: {list(processing_state.keys())}")
    
    # Add debug dump to sidebar
    st.sidebar.write("🔍 RAW processing_state")
    st.sidebar.json(processing_state)
    
    # Extract file IDs and metadata from processing_state
    available_file_ids = []
    
    # Check if we have any selected files in session state
    if "selected_files" in st.session_state and st.session_state.selected_files:
        selected_files = st.session_state.selected_files
        logger.info(f"Found {len(selected_files)} selected files in session state")
        for file_info in selected_files:
            if isinstance(file_info, dict) and "id" in file_info and file_info["id"]:
                # CRITICAL FIX: Ensure file ID is a string
                file_id = str(file_info["id"])
                file_name = file_info.get("name", "Unknown")
                available_file_ids.append(file_id)
                logger.info(f"Added file ID {file_id} from selected_files")
    
    # Pull out the real per‐file results dict
    results_map = processing_state.get("results", {})
    logger.info(f"Results map keys: {list(results_map.keys())}")
    
    file_id_to_metadata = {}
    file_id_to_file_name = {}
    
    # Initialize file_id_to_file_name from selected_files
    if "selected_files" in st.session_state and st.session_state.selected_files:
        for i, file_info in enumerate(st.session_state.selected_files):
            if isinstance(file_info, dict) and "id" in file_info and file_info["id"]:
                file_id = str(file_info["id"])
                file_id_to_file_name[file_id] = file_info.get("name", f"File {file_id}")
    
    for raw_id, payload in results_map.items():
        file_id = str(raw_id)
        available_file_ids.append(file_id)
        
        # Most APIs put your AI fields under payload["results"]
        metadata = payload.get("results", payload)
        
        # If metadata is a string that looks like JSON, try to parse it
        if isinstance(metadata, str):
            try:
                parsed_metadata = json.loads(metadata)
                if isinstance(parsed_metadata, dict):
                    metadata = parsed_metadata
            except json.JSONDecodeError:
                # Not valid JSON, keep as is
                pass
        
        # If payload has an "answer" field that's a JSON string, parse it
        if isinstance(payload, dict) and "answer" in payload and isinstance(payload["answer"], str):
            try:
                parsed_answer = json.loads(payload["answer"])
                if isinstance(parsed_answer, dict):
                    metadata = parsed_answer
            except json.JSONDecodeError:
                # Not valid JSON, keep as is
                pass
        
        file_id_to_metadata[file_id] = metadata
        logger.info(f"Extracted metadata for {file_id}: {metadata!r}")
    
    # Remove duplicates while preserving order
    available_file_ids = list(dict.fromkeys(available_file_ids))
    
    # Debug logging
    logger.info(f"Available file IDs: {available_file_ids}")
    logger.info(f"File ID to file name mapping: {file_id_to_file_name}")
    logger.info(f"File ID to metadata mapping: {list(file_id_to_metadata.keys())}")
    
    st.write("Apply extracted metadata to your Box files.")
    
    # Display selected files
    st.subheader("Selected Files")
    
    if not available_file_ids:
        st.error("No file IDs available for metadata application. Please process files first.")
        if st.button("Go to Process Files", key="go_to_process_files_error_btn"):
            st.session_state.current_page = "Process Files"
            st.rerun()
        return
    
    st.write(f"You have selected {len(available_file_ids)} files for metadata application.")
    
    with st.expander("View Selected Files"):
        for file_id in available_file_ids:
            file_name = file_id_to_file_name.get(file_id, "Unknown")
            st.write(f"- {file_name} ({file_id})")
    
    # Metadata application options
    st.subheader("Application Options")
    
    # For freeform extraction
    st.write("Freeform extraction results will be applied as properties metadata.")
    
    # Option to normalize keys
    normalize_keys = st.checkbox(
        "Normalize keys",
        value=True,
        help="If checked, keys will be normalized (lowercase, spaces replaced with underscores).",
        key="normalize_keys_checkbox"
    )
    
    # Option to filter placeholder values
    filter_placeholders = st.checkbox(
        "Filter placeholder values",
        value=True,
        help="If checked, placeholder values like 'insert date' will be filtered out.",
        key="filter_placeholders_checkbox"
    )
    
    # Batch size (simplified to just 1)
    st.subheader("Batch Processing Options")
    st.write("Using single file processing for reliability.")
    
    # Operation timeout
    timeout_seconds = st.slider(
        "Operation Timeout (seconds)",
        min_value=10,
        max_value=300,
        value=60,
        help="Maximum time to wait for each operation to complete.",
        key="timeout_slider"
    )
    
    # Apply metadata button
    col1, col2 = st.columns(2)
    
    with col1:
        apply_button = st.button(
            "Apply Metadata",
            use_container_width=True,
            key="apply_metadata_btn"
        )
    
    with col2:
        cancel_button = st.button(
            "Cancel",
            use_container_width=True,
            key="cancel_btn"
        )
    
    # Progress tracking
    progress_container = st.container()
    
    # Function to check if a value is a placeholder
    def is_placeholder(value):
        """Check if a value appears to be a placeholder"""
        if not isinstance(value, str):
            return False
            
        placeholder_indicators = [
            "insert", "placeholder", "<", ">", "[", "]", 
            "enter", "fill in", "your", "example"
        ]
        
        value_lower = value.lower()
        return any(indicator in value_lower for indicator in placeholder_indicators)
    
    # Direct function to apply metadata to a single file
    def apply_metadata_to_file_direct(client, file_id, metadata_values):
        """
        Apply metadata to a single file with direct client reference
        
        Args:
            client: Box client object
            file_id: File ID to apply metadata to
            metadata_values: Dictionary of metadata values to apply
            
        Returns:
            dict: Result of metadata application
        """
        try:
            file_name = file_id_to_file_name.get(file_id, "Unknown")
            
            # CRITICAL FIX: Validate metadata values
            if not metadata_values:
                logger.error(f"No metadata found for file {file_name} ({file_id})")
                return {
                    "file_id": file_id,
                    "file_name": file_name,
                    "success": False,
                    "error": "No metadata found for this file"
                }
            
            # Log original metadata values for debugging
            logger.info(f"Original metadata values for file {file_name} ({file_id}): {json.dumps(metadata_values, default=str)}")
            
            # CRITICAL FIX: Filter out confidence score fields
            metadata_values = filter_confidence_fields(metadata_values)
            logger.info(f"Metadata values after filtering confidence fields: {json.dumps(metadata_values, default=str)}")
            
            # Filter out placeholder values if requested
            if filter_placeholders:
                filtered_metadata = {}
                for key, value in metadata_values.items():
                    if not is_placeholder(value):
                        filtered_metadata[key] = value
                
                # If all values were placeholders, keep at least one for debugging
                if not filtered_metadata and metadata_values:
                    # Get the first key-value pair
                    first_key = next(iter(metadata_values))
                    filtered_metadata[first_key] = metadata_values[first_key]
                    filtered_metadata["_note"] = "All other values were placeholders"
                
                metadata_values = filtered_metadata
            
            # If no metadata values after filtering, return error
            if not metadata_values:
                logger.warning(f"No valid metadata found for file {file_name} ({file_id}) after filtering")
                return {
                    "file_id": file_id,
                    "file_name": file_name,
                    "success": False,
                    "error": "No valid metadata found after filtering placeholders"
                }
            
            # Convert all values to strings for Box metadata
            metadata_values_final = {}
            for key, value in metadata_values.items():
                if not isinstance(value, (str, int, float, bool)):
                    metadata_values_final[key] = str(value)
                else:
                    metadata_values_final[key] = value
            metadata_values = metadata_values_final # Use the processed dict
            
            # Debug logging
            logger.info(f"Applying metadata for file: {file_name} ({file_id})")
            logger.info(f"Metadata values before template/properties check: {json.dumps(metadata_values, default=str)}")
            
            # Get file object
            file_obj = client.file(file_id=file_id)
                        # --- BEGIN TEMPLATE SELECTION LOGIC ---
            target_template_id = ""
            apply_as_template = False

            # Check if structured extraction is selected
            if st.session_state.metadata_config.get("extraction_method") == "structured":
                # 1. Get document type for the current file
                doc_type = None
                if hasattr(st.session_state, "document_categorization") and file_id in st.session_state.document_categorization.get("results", {}):
                    doc_type = st.session_state.document_categorization["results"][file_id].get("document_type")
                    logger.info(f"File {file_id} has document type: {doc_type}")
                else:
                    logger.info(f"No document type found for file {file_id} in categorization results.")

                # 2. Check for specific mapping for this document type
                if doc_type and hasattr(st.session_state, "document_type_to_template") and doc_type in st.session_state.document_type_to_template:
                    mapped_template_id = st.session_state.document_type_to_template[doc_type]
                    if mapped_template_id:
                        target_template_id = mapped_template_id
                        apply_as_template = True
                        logger.info(f"Using mapped template 	{target_template_id}	 for doc type {doc_type}")
                    else:
                        logger.info(f"Doc type {doc_type} is mapped to None, will check default.")
                else:
                    logger.info(f"No specific template mapping found for doc type {doc_type}, will check default.")

                # 3. If no specific mapping, check for default template
                if not apply_as_template:
                    default_template_id = st.session_state.metadata_config.get("template_id", "")
                    if default_template_id:
                        target_template_id = default_template_id
                        apply_as_template = True
                        logger.info(f"Using default template 	{target_template_id}	 for file {file_id}")
                    else:
                        logger.info(f"No default template selected. Will apply as properties.")
            else:
                logger.info("Extraction method is not structured. Will apply as properties.")
            # --- END TEMPLATE SELECTION LOGIC ---

            # Check if we determined a template should be used
            if apply_as_template and target_template_id:
                # Parse the target_template_id to extract the correct components
                # Format is typically: scope_id_templateKey (e.g., enterprise_336904155_financialReport)
                parts = target_template_id.split('_')
                
                # Extract the scope and enterprise ID
                scope = parts[0]  # e.g., "enterprise"
                enterprise_id = parts[1] if len(parts) > 1 else ""
                
                # Extract the actual template key (last part)
                template_key = parts[-1] if len(parts) > 2 else target_template_id # Fallback if format is unexpected
                
                # Format the scope with enterprise ID
                scope_with_id = f"{scope}_{enterprise_id}"
                
                logger.info(f"Applying as template: Scope={scope_with_id}, Key={template_key}")
                
                try:
                    # ENHANCED FIX: Step 1 - Fix metadata format by converting string representations to dictionaries
                    formatted_metadata = fix_metadata_format(metadata_values)
                    logger.info(f"Formatted metadata after fix_metadata_format: {json.dumps(formatted_metadata, default=str)}")
                    
                    # ENHANCED FIX: Step 2 - Flatten metadata structure to match template requirements
                    flattened_metadata = flatten_metadata_for_template(formatted_metadata)
                    logger.info(f"Flattened metadata after flatten_metadata_for_template: {json.dumps(flattened_metadata, default=str)}")
                    
                    # Log the flattened metadata before potential modification
                    logger.info(f"Flattened metadata before targeted conversion: {json.dumps(flattened_metadata, default=str)}")

                    # --- BEGIN TARGETED VALUE TYPE CONVERSION FOR TEMPLATES ---
                    metadata_for_api = flattened_metadata.copy() # Start with a copy
                    target_key = "totalTaxWithheld" # Key to potentially convert

                    if target_key in metadata_for_api and isinstance(metadata_for_api[target_key], str):
                        value_str = metadata_for_api[target_key]
                        try:
                            # Remove commas and attempt conversion to float
                            cleaned_value = value_str.replace(",", "")
                            numeric_value = float(cleaned_value)
                            metadata_for_api[target_key] = numeric_value # Update the dictionary
                            logger.info(f"Converted value for key 	{target_key}	 from 	{value_str}	 to float: {numeric_value}")
                        except ValueError:
                            # If conversion fails, keep original string
                            logger.info(f"Value for key 	{target_key}	 kept as string: {value_str}")
                    # --- END TARGETED VALUE TYPE CONVERSION FOR TEMPLATES ---

                    # Log the final metadata being sent to Box API
                    logger.info(f"Sending final metadata to Box API (Create): {json.dumps(metadata_for_api, default=str)}")
                    
                    # Apply metadata using the template with potentially converted value
                    metadata = file_obj.metadata(scope_with_id, template_key).create(metadata_for_api)
                    logger.info(f"Successfully applied template metadata to file {file_name} ({file_id})")
                    return {
                        "file_id": file_id,
                        "file_name": file_name,
                        "success": True,
                        "metadata": metadata
                    }
                except Exception as e:
                    if "already exists" in str(e).lower():
                        # If metadata already exists, update it
                        try:
                            # ENHANCED FIX: Step 1 - Fix metadata format by converting string representations to dictionaries
                            formatted_metadata = fix_metadata_format(metadata_values)
                            logger.info(f"Formatted metadata after fix_metadata_format (update path): {json.dumps(formatted_metadata, default=str)}")
                            
                            # ENHANCED FIX: Step 2 - Flatten metadata structure to match template requirements
                            flattened_metadata = flatten_metadata_for_template(formatted_metadata)
                            logger.info(f"Flattened metadata after flatten_metadata_for_template (update path): {json.dumps(flattened_metadata, default=str)}")
                            
                            # Log the flattened metadata being sent to Box API
                            logger.info(f"Updating with flattened metadata: {json.dumps(flattened_metadata, default=str)}")
                            
                            # Create update operations with flattened metadata
                            operations = []
                            for key, value in flattened_metadata.items():
                                operations.append({
                                    "op": "replace",
                                    "path": f"/{key}",
                                    "value": value
                                })
                            
                            # Update metadata
                            logger.info(f"Template metadata already exists, updating with operations: {json.dumps(operations, default=str)}")
                            metadata = file_obj.metadata(scope_with_id, template_key).update(operations)
                            
                            logger.info(f"Successfully updated template metadata for file {file_name} ({file_id})")
                            return {
                                "file_id": file_id,
                                "file_name": file_name,
                                "success": True,
                                "metadata": metadata
                            }
                        except Exception as update_error:
                            logger.error(f"Error updating template metadata for file {file_name} ({file_id}): {str(update_error)}")
                            return {
                                "file_id": file_id,
                                "file_name": file_name,
                                "success": False,
                                "error": f"Error updating template metadata: {str(update_error)}"
                            }
                    else:
                        logger.error(f"Error creating template metadata for file {file_name} ({file_id}): {str(e)}")
                        return {
                            "file_id": file_id,
                            "file_name": file_name,
                            "success": False,
                            "error": f"Error creating template metadata: {str(e)}"
                        }
            else:
                # For non-template metadata (freeform), apply as properties
                
                # --- BEGIN CONDITIONAL KEY NORMALIZATION ---
                # Check if normalization is requested via checkbox
                normalize_keys = st.session_state.get("normalize_keys_checkbox", False) # Default to False if not found
                
                if normalize_keys:
                    logger.info(f"Applying key normalization (removing underscores) for properties metadata.")
                    normalized_metadata = {}
                    for key, value in metadata_values.items():
                        # Specific normalization: remove underscores
                        normalized_key = key.replace("_", "")
                        normalized_metadata[normalized_key] = value
                    metadata_to_apply = normalized_metadata
                    logger.info(f"Metadata after normalization: {json.dumps(metadata_to_apply, default=str)}")
                else:
                    metadata_to_apply = metadata_values # Use original keys
                    logger.info(f"Skipping key normalization for properties metadata.")
                # --- END CONDITIONAL KEY NORMALIZATION ---
                
                try:
                    # Apply metadata as properties using metadata_to_apply
                    metadata = file_obj.metadata("global", "properties").create(metadata_to_apply)
                    logger.info(f"Successfully applied metadata to file {file_name} ({file_id})")
                    return {
                        "file_id": file_id,
                        "file_name": file_name,
                        "success": True,
                        "metadata": metadata
                    }
                except Exception as e:
                    if "already exists" in str(e).lower():
                        # If metadata already exists, update it
                        try:
                            # Create update operations
                            operations = []
                            for key, value in metadata_values.items():
                                operations.append({
                                    "op": "replace",
                                    "path": f"/{key}",
                                    "value": value
                                })
                            
                            # Update metadata
                            logger.info(f"Metadata already exists, updating with operations")
                            metadata = file_obj.metadata("global", "properties").update(operations)
                            
                            logger.info(f"Successfully updated metadata for file {file_name} ({file_id})")
                            return {
                                "file_id": file_id,
                                "file_name": file_name,
                                "success": True,
                                "metadata": metadata
                            }
                        except Exception as update_error:
                            logger.error(f"Error updating metadata for file {file_name} ({file_id}): {str(update_error)}")
                            return {
                                "file_id": file_id,
                                "file_name": file_name,
                                "success": False,
                                "error": f"Error updating metadata: {str(update_error)}"
                            }
                    else:
                        logger.error(f"Error creating metadata for file {file_name} ({file_id}): {str(e)}")
                        return {
                            "file_id": file_id,
                            "file_name": file_name,
                            "success": False,
                            "error": f"Error creating metadata: {str(e)}"
                        }
        
        except Exception as e:
            logger.exception(f"Unexpected error applying metadata to file {file_id}: {str(e)}")
            return {
                "file_id": file_id,
                "file_name": file_id_to_file_name.get(file_id, "Unknown"),
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    # Handle apply button click - DIRECT APPROACH WITHOUT THREADING
    if apply_button:
        # Check if client exists directly again
        if 'client' not in st.session_state:
            st.error("Box client not found. Please authenticate first.")
            return
        
        # Get client directly
        client = st.session_state.client
        
        # Process files one by one
        results = []
        errors = []
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each file
        for i, file_id in enumerate(available_file_ids):
            file_name = file_id_to_file_name.get(file_id, "Unknown")
            status_text.text(f"Processing {file_name}...")
            
            # Get metadata for this file
            metadata_values = file_id_to_metadata.get(file_id, {})
            
            # CRITICAL FIX: Log the metadata values before application
            logger.info(f"Metadata values for file {file_name} ({file_id}) before application: {json.dumps(metadata_values, default=str)}")
            
            # Apply metadata directly
            result = apply_metadata_to_file_direct(client, file_id, metadata_values)
            
            if result["success"]:
                results.append(result)
            else:
                errors.append(result)
            
            # Update progress
            progress = (i + 1) / len(available_file_ids)
            progress_bar.progress(progress)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Show results
        st.subheader("Metadata Application Results")
        st.write(f"Successfully applied metadata to {len(results)} of {len(available_file_ids)} files.")
        
        if errors:
            with st.expander("View Errors"):
                for error in errors:
                    st.write(f"**{error['file_name']}:** {error['error']}")
        
        if results:
            with st.expander("View Results"):
                for result in results:
                    st.write(f"**{result['file_name']}:** Metadata applied successfully")
    
    # Handle cancel button click
    if cancel_button:
        st.session_state.current_page = "Home"
        st.rerun()
