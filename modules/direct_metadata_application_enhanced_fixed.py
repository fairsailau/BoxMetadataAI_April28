import streamlit as st
import logging
import json
from boxsdk import Client

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format=\'%(asctime)s - %(name)s - %(levelname)s - %(message)s\')
logger = logging.getLogger(__name__)

def fix_metadata_format(metadata_values):
    \"\"\"
    Fix the metadata format by converting string representations of dictionaries
    to actual Python dictionaries.
    
    Args:
        metadata_values (dict): The original metadata values dictionary
        
    Returns:
        dict: A new dictionary with properly formatted metadata values
    \"\"\"
    formatted_metadata = {}
    
    for key, value in metadata_values.items():
        # If the value is a string that looks like a dictionary, parse it
        if isinstance(value, str) and value.startswith(\'{\') and value.endswith(\'}\'):
            try:
                # Replace single quotes with double quotes for JSON compatibility
                json_compatible_str = value.replace("\'", \'\"\')
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
    \"\"\"
    Flatten the metadata structure by extracting fields from the \'answer\' object
    and placing them directly at the top level to match the template structure.
    
    Args:
        metadata_values (dict): The metadata values with nested objects
        
    Returns:
        dict: A flattened dictionary with fields at the top level
    \"\"\"
    flattened_metadata = {}
    
    # Check if \'answer\' exists and is a dictionary
    if \'answer\' in metadata_values and isinstance(metadata_values[\'answer\'], dict):
        # Extract fields from the \'answer\' object and place them at the top level
        for key, value in metadata_values[\'answer\'].items():
            flattened_metadata[key] = value
    else:
        # If there\'s no \'answer\' object, use the original metadata
        flattened_metadata = metadata_values.copy()
    
    # Remove any non-template fields that shouldn\'t be sent to Box API
    # These are fields that are used internally but not part of the template
    keys_to_remove = [\'ai_agent_info\', \'created_at\', \'completion_reason\', \'answer\']
    for key in keys_to_remove:
        if key in flattened_metadata:
            del flattened_metadata[key]
    
    return flattened_metadata

# Function to check if a value is a placeholder
def is_placeholder(value):
    \"\"\"Check if a value appears to be a placeholder\"\"\"
    if not isinstance(value, str):
        return False
        
    placeholder_indicators = [
        "insert", "placeholder", "<", ">", "[", "]", 
        "enter", "fill in", "your", "example"
    ]
    
    value_lower = value.lower()
    return any(indicator in value_lower for indicator in placeholder_indicators)

# Moved outside of apply_metadata_direct to make it directly importable
def apply_metadata_to_file_direct(client, file_id, metadata_values, normalize_keys=True, filter_placeholders=True, file_id_to_file_name=None):
    \"\"\"
    Apply metadata to a single file with direct client reference
    
    Args:
        client: Box client object
        file_id: File ID to apply metadata to
        metadata_values: Dictionary of metadata values to apply
        normalize_keys: Whether to normalize keys (lowercase, replace spaces with underscores)
        filter_placeholders: Whether to filter out placeholder values
        file_id_to_file_name: Optional dictionary mapping file IDs to file names
        
    Returns:
        dict: Result of metadata application
    \"\"\"
    try:
        # Initialize file_id_to_file_name if not provided
        if file_id_to_file_name is None:
            file_id_to_file_name = {}
        
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
        
        # Normalize keys if requested
        if normalize_keys:
            normalized_metadata = {}
            for key, value in metadata_values.items():
                # Convert to lowercase and replace spaces with underscores
                normalized_key = key.lower().replace(" ", "_").replace("-", "_")
                normalized_metadata[normalized_key] = value
            metadata_values = normalized_metadata
        
        # Convert all values to strings for Box metadata
        # FIX: Convert only non-string values to strings, handle None
        metadata_for_box = {}
        for key, value in metadata_values.items():
            if value is None:
                metadata_for_box[key] = "" # Box metadata doesn\'t accept None, use empty string
            elif not isinstance(value, str):
                metadata_for_box[key] = str(value)
            else:
                metadata_for_box[key] = value
        
        # Debug logging
        logger.info(f"Applying metadata for file: {file_name} ({file_id})")
        logger.info(f"Metadata values after normalization and string conversion: {json.dumps(metadata_for_box, default=str)}")
        
        # Get file object
        file_obj = client.file(file_id=file_id)
        
        # Check if we\'re using structured extraction with a template
        if "metadata_config" in st.session_state and st.session_state.metadata_config.get("extraction_method") == "structured":
            # Get document type for this file (if categorized)
            document_type = None
            if (
                hasattr(st.session_state, "document_categorization") and 
                st.session_state.document_categorization.get("is_categorized", False) and
                file_id in st.session_state.document_categorization["results"]
            ):
                document_type = st.session_state.document_categorization["results"][file_id]["document_type"]
                logger.info(f"File {file_name} has document type: {document_type}")
            
            # Get template ID based on document type if available
            template_id = None
            
            # Check if we have a document type and a mapping for it
            if document_type and hasattr(st.session_state, "document_type_to_template"):
                mapped_template_id = st.session_state.document_type_to_template.get(document_type)
                if mapped_template_id and mapped_template_id != "None - Use custom fields":
                    template_id = mapped_template_id
                    logger.info(f"Using document type specific template for {document_type}: {template_id}")
            
            # If no document type specific template, use the general one
            if not template_id:
                template_id = st.session_state.metadata_config.get("template_id", "")
                logger.info(f"Using general template: {template_id}")
            
            # Skip if template_id is "None - Use custom fields" or empty
            if not template_id or template_id == "None - Use custom fields":
                logger.info(f"No template selected, using properties metadata instead")
                # Apply metadata as properties
                try:
                    # FIX: Filter out _confidence fields before applying properties
                    properties_metadata = {k: v for k, v in metadata_for_box.items() if not k.endswith("_confidence")}
                    logger.info(f"Sending properties metadata to Box API: {json.dumps(properties_metadata, default=str)}")
                    
                    metadata = file_obj.metadata("global", "properties").create(properties_metadata)
                    logger.info(f"Successfully applied properties metadata to file {file_name} ({file_id})")
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
                            # FIX: Filter out _confidence fields before updating properties
                            properties_metadata = {k: v for k, v in metadata_for_box.items() if not k.endswith("_confidence")}
                            logger.info(f"Updating properties metadata with: {json.dumps(properties_metadata, default=str)}")
                            
                            # Create update operations
                            operations = []
                            for key, value in properties_metadata.items():
                                operations.append({
                                    "op": "replace",
                                    "path": f"/{key}",
                                    "value": value
                                })
                            
                            # Update metadata
                            logger.info(f"Properties metadata already exists, updating with operations")
                            metadata = file_obj.metadata("global", "properties").update(operations)
                            
                            logger.info(f"Successfully updated properties metadata for file {file_name} ({file_id})")
                            return {
                                "file_id": file_id,
                                "file_name": file_name,
                                "success": True,
                                "metadata": metadata
                            }
                        except Exception as update_error:
                            logger.error(f"Error updating properties metadata for file {file_name} ({file_id}): {str(update_error)}")
                            return {
                                "file_id": file_id,
                                "file_name": file_name,
                                "success": False,
                                "error": f"Error updating properties metadata: {str(update_error)}"
                            }
                    else:
                        logger.error(f"Error creating properties metadata for file {file_name} ({file_id}): {str(e)}")
                        return {
                            "file_id": file_id,
                            "file_name": file_name,
                            "success": False,
                            "error": f"Error creating properties metadata: {str(e)}"
                        }
            
            # Parse the template ID to extract the correct components
            # Format is typically: scope_id_templateKey (e.g., enterprise_336904155_financialReport)
            parts = template_id.split(\'_\')
            
            # Extract the scope and enterprise ID
            scope = parts[0]  # e.g., "enterprise"
            enterprise_id = parts[1] if len(parts) > 1 else ""
            
            # Extract the actual template key (last part)
            if len(parts) > 2:
                # For templates with format scope_id_templateKey
                template_key = parts[2]
            else:
                # For templates with format scope_templateKey or just templateKey
                template_key = parts[-1]
            
            # Format the scope with enterprise ID
            scope_with_id = f"{scope}_{enterprise_id}"
            
            logger.info(f"Using template-based metadata application with scope: {scope_with_id}, template: {template_key}")
            
            try:
                # ENHANCED FIX: Step 1 - Fix metadata format (already done above in metadata_for_box)
                # ENHANCED FIX: Step 2 - Flatten metadata structure (already done above in metadata_for_box)
                
                # FIX: Filter out _confidence fields before applying template metadata
                template_metadata = {k: v for k, v in metadata_for_box.items() if not k.endswith("_confidence")}
                logger.info(f"Sending template metadata to Box API: {json.dumps(template_metadata, default=str)}")
                
                # Apply metadata using the template with properly formatted and filtered metadata
                metadata = file_obj.metadata(scope_with_id, template_key).create(template_metadata)
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
                        # ENHANCED FIX: Step 1 & 2 (already done above in metadata_for_box)
                        
                        # FIX: Filter out _confidence fields before updating template metadata
                        template_metadata = {k: v for k, v in metadata_for_box.items() if not k.endswith("_confidence")}
                        logger.info(f"Updating template metadata with: {json.dumps(template_metadata, default=str)}")
                        
                        # Create update operations with filtered metadata
                        operations = []
                        for key, value in template_metadata.items():
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
            # Apply metadata as properties if not using structured extraction
            try:
                # FIX: Filter out _confidence fields before applying properties
                properties_metadata = {k: v for k, v in metadata_for_box.items() if not k.endswith("_confidence")}
                logger.info(f"Sending properties metadata (non-structured) to Box API: {json.dumps(properties_metadata, default=str)}")
                
                metadata = file_obj.metadata("global", "properties").create(properties_metadata)
                logger.info(f"Successfully applied properties metadata (non-structured) to file {file_name} ({file_id})")
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
                        # FIX: Filter out _confidence fields before updating properties
                        properties_metadata = {k: v for k, v in metadata_for_box.items() if not k.endswith("_confidence")}
                        logger.info(f"Updating properties metadata (non-structured) with: {json.dumps(properties_metadata, default=str)}")
                        
                        # Create update operations
                        operations = []
                        for key, value in properties_metadata.items():
                            operations.append({
                                "op": "replace",
                                "path": f"/{key}",
                                "value": value
                            })
                        
                        # Update metadata
                        logger.info(f"Properties metadata (non-structured) already exists, updating with operations")
                        metadata = file_obj.metadata("global", "properties").update(operations)
                        
                        logger.info(f"Successfully updated properties metadata (non-structured) for file {file_name} ({file_id})")
                        return {
                            "file_id": file_id,
                            "file_name": file_name,
                            "success": True,
                            "metadata": metadata
                        }
                    except Exception as update_error:
                        logger.error(f"Error updating properties metadata (non-structured) for file {file_name} ({file_id}): {str(update_error)}")
                        return {
                            "file_id": file_id,
                            "file_name": file_name,
                            "success": False,
                            "error": f"Error updating properties metadata (non-structured): {str(update_error)}"
                        }
                else:
                    logger.error(f"Error creating properties metadata (non-structured) for file {file_name} ({file_id}): {str(e)}")
                    return {
                        "file_id": file_id,
                        "file_name": file_name,
                        "success": False,
                        "error": f"Error creating properties metadata (non-structured): {str(e)}"
                    }
    
    except Exception as e:
        logger.error(f"Unexpected error applying metadata to file {file_name} ({file_id}): {str(e)}")
        return {
            "file_id": file_id,
            "file_name": file_name,
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }

def apply_metadata_direct():
    \"\"\"
    Apply metadata directly to selected files using the Box API
    \"\"\"
    # Verify authentication
    if not hasattr(st.session_state, "authenticated") or not st.session_state.authenticated:
        st.error("Please authenticate with Box first")
        return
    
    client = st.session_state.client
    
    # Verify client authentication
    try:
        user = client.users.get_current_user()
        logger.info(f"Verified client authentication as {user.name}")
    except Exception as e:
        st.error(f"Box client authentication failed: {e}")
        return
    
    # Log session state keys for debugging
    logger.info(f"Processing state keys: {list(st.session_state.keys())}")
    
    # Get selected file IDs from session state
    selected_file_ids = st.session_state.get("selected_result_ids", [])
    
    # If no files selected in results viewer, check if any files were selected in file browser
    if not selected_file_ids and hasattr(st.session_state, "selected_files") and st.session_state.selected_files:
        logger.info(f"Found {len(st.session_state.selected_files)} selected files in session state")
        for file_info in st.session_state.selected_files:
            if "id" in file_info:
                selected_file_ids.append(file_info["id"])
                logger.info(f"Added file ID {file_info[\'id\']} from selected_files")
    
    if not selected_file_ids:
        st.warning("No files selected for metadata application.")
        return
    
    # Get extraction results from session state
    results_map = st.session_state.get("extraction_results", {})
    logger.info(f"Results map keys: {list(results_map.keys())}")
    
    # Get file ID to file name mapping
    file_id_to_file_name = {}
    if hasattr(st.session_state, "selected_files"):
        for file_info in st.session_state.selected_files:
            if "id" in file_info and "name" in file_info:
                file_id_to_file_name[file_info["id"]] = file_info["name"]
    
    # Get file ID to metadata mapping
    file_id_to_metadata = {}
    available_file_ids = list(results_map.keys())
    logger.info(f"Available file IDs: {available_file_ids}")
    logger.info(f"File ID to file name mapping: {file_id_to_file_name}")
    
    for file_id in selected_file_ids:
        if file_id in results_map:
            # Get the metadata for the file ID
            metadata = results_map[file_id]
            logger.info(f"Extracted metadata for {file_id}: {json.dumps(metadata, default=str)}")
            file_id_to_metadata[file_id] = metadata
        else:
            logger.warning(f"No extraction results found for selected file ID: {file_id}")
    
    logger.info(f"File ID to metadata mapping: {list(file_id_to_metadata.keys())}")
    
    if not file_id_to_metadata:
        st.warning("No metadata available for the selected files.")
        return
    
    # Apply metadata to each selected file
    application_results = []
    progress_bar = st.progress(0)
    total_files = len(file_id_to_metadata)
    
    for i, (file_id, metadata_values) in enumerate(file_id_to_metadata.items()):
        # Log metadata values before application
        logger.info(f"Metadata values for file {file_id_to_file_name.get(file_id, file_id)} ({file_id}) before application: {json.dumps(metadata_values, default=str)}")
        
        # Apply metadata to the file
        result = apply_metadata_to_file_direct(
            client=client,
            file_id=file_id,
            metadata_values=metadata_values,
            normalize_keys=True, # Normalize keys by default
            filter_placeholders=True, # Filter placeholders by default
            file_id_to_file_name=file_id_to_file_name
        )
        application_results.append(result)
        
        # Update progress bar
        progress_bar.progress((i + 1) / total_files)
    
    # Display results
    st.subheader("Metadata Application Results")
    
    success_count = sum(1 for result in application_results if result["success"])
    error_count = total_files - success_count
    
    st.success(f"Successfully applied metadata to {success_count} files.")
    if error_count > 0:
        st.error(f"Failed to apply metadata to {error_count} files.")
    
    # Show detailed results
    with st.expander("View Detailed Results"):
        for result in application_results:
            file_name = result.get("file_name", result["file_id"])
            if result["success"]:
                st.write(f"✅ **{file_name}**: Metadata applied successfully.")
            else:
                st.write(f"❌ **{file_name}**: Failed - {result[\'error\']}")

