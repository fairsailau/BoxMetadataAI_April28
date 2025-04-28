import streamlit as st
import logging
import json
from boxsdk import Client

# Configure logging
# Corrected logging configuration format
logging.basicConfig(level=logging.INFO, 
                   format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
        # Corrected string literal checks
        if isinstance(value, str) and value.startswith('{') and value.endswith('}'):
            try:
                # Replace single quotes with double quotes for JSON compatibility
                # Corrected string replacement
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

# Moved outside of apply_metadata_direct to make it directly importable
def apply_metadata_to_file_direct(client, file_id, metadata_values, normalize_keys=True, filter_placeholders=True, file_id_to_file_name=None):
    """
    Apply metadata to a single file with direct client reference
    
    Args:
        client: Box client object
        file_id: File ID to apply metadata to
        metadata_values: Dictionary of metadata values to apply (should contain the actual values, not confidence)
        normalize_keys: Whether to normalize keys (lowercase, replace spaces with underscores)
        filter_placeholders: Whether to filter out placeholder values
        file_id_to_file_name: Optional dictionary mapping file IDs to file names
        
    Returns:
        dict: Result of metadata application
    """
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
            # CRITICAL FIX: Ensure _confidence fields are NOT included here
            if key.endswith("_confidence"):
                continue # Skip confidence fields
                
            if value is None:
                metadata_for_box[key] = "" # Box metadata doesn't accept None, use empty string
            elif not isinstance(value, str):
                metadata_for_box[key] = str(value)
            else:
                metadata_for_box[key] = value
        
        # Debug logging
        logger.info(f"Applying metadata for file: {file_name} ({file_id})")
        logger.info(f"Metadata values after normalization, filtering, and string conversion: {json.dumps(metadata_for_box, default=str)}")
        
        # Get file object
        file_obj = client.file(file_id=file_id)
        
        # Check if we're using structured extraction with a template
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
                    # FIX: metadata_for_box already has confidence fields filtered out
                    properties_metadata = metadata_for_box
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
                            # FIX: metadata_for_box already has confidence fields filtered out
                            properties_metadata = metadata_for_box
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
            # Corrected string split
            parts = template_id.split('_')
            
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
                # FIX: metadata_for_box already has confidence fields filtered out
                template_metadata = metadata_for_box
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
                        # FIX: metadata_for_box already has confidence fields filtered out
                        template_metadata = metadata_for_box
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
                        logger.info(f"Template metadata already exists, updating with operations")
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
            # Fallback to properties metadata if not using structured extraction
            logger.info(f"Not using structured extraction, applying as properties metadata")
            try:
                # FIX: metadata_for_box already has confidence fields filtered out
                properties_metadata = metadata_for_box
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
                        # FIX: metadata_for_box already has confidence fields filtered out
                        properties_metadata = metadata_for_box
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
    except Exception as e:
        logger.error(f"Unexpected error applying metadata to file {file_id}: {str(e)}")
        return {
            "file_id": file_id,
            "file_name": file_id_to_file_name.get(file_id, "Unknown"),
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }

def apply_metadata_direct():
    """
    Apply metadata to selected files using direct client reference
    """
    if not hasattr(st.session_state, "selected_result_ids") or not st.session_state.selected_result_ids:
        st.warning("No files selected for metadata application.")
        return
    
    if not hasattr(st.session_state, "extraction_results") or not st.session_state.extraction_results:
        st.error("No extraction results available to apply.")
        return
    
    # Get client from session state
    client = st.session_state.client
    
    # Create file ID to file name mapping
    file_id_to_file_name = {}
    if hasattr(st.session_state, "selected_files"):
        for file in st.session_state.selected_files:
            file_id_to_file_name[file["id"]] = file["name"]
    
    results = []
    progress_bar = st.progress(0)
    total_files = len(st.session_state.selected_result_ids)
    
    for i, file_id in enumerate(st.session_state.selected_result_ids):
        if file_id in st.session_state.extraction_results:
            # Get the processed result data (which should contain only values, not confidence)
            # This assumes results_viewer has prepared the data correctly
            # We need to re-process here to be sure
            raw_result = st.session_state.extraction_results[file_id]
            metadata_values = {}
            
            # Re-process the raw result to extract only the values
            if isinstance(raw_result, dict):
                if "answer" in raw_result:
                    answer = raw_result["answer"]
                    if isinstance(answer, str):
                        try: parsed_answer = json.loads(answer.replace("'", '"'))
                        except: parsed_answer = {"extracted_text": answer}
                    elif isinstance(answer, dict): parsed_answer = answer
                    else: parsed_answer = {"extracted_text": str(answer)}
                    
                    if isinstance(parsed_answer, dict):
                        for key, value in parsed_answer.items():
                            if isinstance(value, dict) and "value" in value: metadata_values[key] = value["value"]
                            else: metadata_values[key] = value
                elif "items" in raw_result and isinstance(raw_result["items"], list) and len(raw_result["items"]) > 0:
                    item = raw_result["items"][0]
                    if isinstance(item, dict) and "answer" in item:
                        answer = item["answer"]
                        if isinstance(answer, str):
                            try: parsed_answer = json.loads(answer.replace("'", '"'))
                            except: parsed_answer = {"extracted_text": answer}
                        elif isinstance(answer, dict): parsed_answer = answer
                        else: parsed_answer = {"extracted_text": str(answer)}
                        
                        if isinstance(parsed_answer, dict):
                            for key, value in parsed_answer.items():
                                if isinstance(value, dict) and "value" in value: metadata_values[key] = value["value"]
                                else: metadata_values[key] = value
                elif any(key.endswith("_confidence") for key in raw_result.keys()):
                    for key, value in raw_result.items():
                        if not key.endswith("_confidence") and not key.startswith("_"): metadata_values[key] = value
                else:
                    # Fallback: use all non-internal fields
                    for key, value in raw_result.items():
                        if not key.startswith("_") and not key.endswith("_confidence"): metadata_values[key] = value
            else:
                metadata_values = {"extracted_text": str(raw_result)}
            
            # Apply metadata to the file
            result = apply_metadata_to_file_direct(
                client=client,
                file_id=file_id,
                metadata_values=metadata_values, # Pass only the values
                normalize_keys=True, # Assuming normalization is desired
                filter_placeholders=True, # Assuming placeholder filtering is desired
                file_id_to_file_name=file_id_to_file_name
            )
            results.append(result)
        else:
            results.append({
                "file_id": file_id,
                "file_name": file_id_to_file_name.get(file_id, "Unknown"),
                "success": False,
                "error": "Extraction result not found"
            })
        
        # Update progress bar
        progress_bar.progress((i + 1) / total_files)
    
    # Display results
    st.subheader("Metadata Application Results")
    successful_applications = [r for r in results if r["success"]]
    failed_applications = [r for r in results if not r["success"]] 
    
    if successful_applications:
        st.success(f"Successfully applied metadata to {len(successful_applications)} files:")
        # Optionally display details of successful applications
        # with st.expander("Show Success Details"):
        #     st.json(successful_applications)
            
    if failed_applications:
        st.error(f"Failed to apply metadata to {len(failed_applications)} files:")
        for failure in failed_applications:
            st.write(f"- File: {failure['file_name']} ({failure['file_id']}) - Error: {failure['error']}")

