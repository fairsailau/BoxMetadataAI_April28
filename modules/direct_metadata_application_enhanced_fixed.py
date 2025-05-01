import streamlit as st
import logging
import json
from boxsdk import Client

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Functions copied from the working repository (MetadataAI-April29) --- 

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
        if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
            try:
                # Replace single quotes with double quotes for JSON compatibility
                json_compatible_str = value.replace("\"", "\"") # Use double quotes for replacement
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
    Flatten the metadata structure by extracting fields from the \'answer\' object
    and placing them directly at the top level to match the template structure.
    
    Args:
        metadata_values (dict): The metadata values with nested objects
        
    Returns:
        dict: A flattened dictionary with fields at the top level
    """
    flattened_metadata = {}
    
    # Check if \'answer\' exists and is a dictionary
    if "answer" in metadata_values and isinstance(metadata_values["answer"], dict):
        # Extract fields from the \'answer\' object and place them at the top level
        for key, value in metadata_values["answer"].items():
            flattened_metadata[key] = value
    else:
        # If there\'s no \'answer\' object, use the original metadata
        flattened_metadata = metadata_values.copy()
    
    # Remove any non-template fields that shouldn\'t be sent to Box API
    # These are fields that are used internally but not part of the template
    keys_to_remove = ["ai_agent_info", "created_at", "completion_reason", "answer"]
    for key in keys_to_remove:
        if key in flattened_metadata:
            del flattened_metadata[key]
            
    # CRITICAL: Also remove confidence fields here before sending to Box
    confidence_keys = [key for key in flattened_metadata if key.endswith("_confidence")]
    for key in confidence_keys:
        del flattened_metadata[key]
        logger.info(f"Removed confidence field {key} during flattening.")

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

# --- End of functions copied from the working repository --- 

# Updated apply_metadata_to_file_direct incorporating logic from the working repository
def apply_metadata_to_file_direct(client, file_id, metadata_values, normalize_keys=True, filter_placeholders=True, file_id_to_file_name=None):
    """
    Apply metadata to a single file with direct client reference
    (Incorporates logic from MetadataAI-April29 for metadata preparation)
    
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
        
        # --- Metadata Preparation Steps from Working Repo --- 
        
        # 1. Fix format (string dicts to actual dicts) - Applied to original values
        prepared_metadata = fix_metadata_format(metadata_values)
        logger.info(f"Metadata after fix_metadata_format: {json.dumps(prepared_metadata, default=str)}")

        # 2. Filter placeholders if requested - Applied after format fix
        if filter_placeholders:
            filtered_metadata_placeholders = {}
            for key, value in prepared_metadata.items():
                if not is_placeholder(value):
                    filtered_metadata_placeholders[key] = value
            
            if not filtered_metadata_placeholders and prepared_metadata:
                first_key = next(iter(prepared_metadata))
                filtered_metadata_placeholders[first_key] = prepared_metadata[first_key]
                filtered_metadata_placeholders["_note"] = "All other values were placeholders"
            
            prepared_metadata = filtered_metadata_placeholders
            logger.info(f"Metadata after placeholder filtering: {json.dumps(prepared_metadata, default=str)}")

        # 3. Normalize keys if requested - Applied after placeholder filtering
        if normalize_keys:
            normalized_metadata_keys = {}
            for key, value in prepared_metadata.items():
                normalized_key = key.lower().replace(" ", "_").replace("-", "_")
                normalized_metadata_keys[normalized_key] = value
            prepared_metadata = normalized_metadata_keys
            logger.info(f"Metadata after key normalization: {json.dumps(prepared_metadata, default=str)}")

        # 4. Flatten structure for template (removes answer, confidence fields etc.) - Applied after normalization
        # This step is crucial for template application and also removes confidence fields
        metadata_for_box = flatten_metadata_for_template(prepared_metadata)
        logger.info(f"Metadata after flattening (and confidence removal): {json.dumps(metadata_for_box, default=str)}")

        # 5. Convert remaining values to strings for Box API - Applied last
        final_metadata_for_box = {}
        for key, value in metadata_for_box.items():
            if value is None:
                final_metadata_for_box[key] = "" # Box metadata doesn\'t accept None
            elif not isinstance(value, str):
                final_metadata_for_box[key] = str(value)
            else:
                final_metadata_for_box[key] = value
        logger.info(f"Final metadata prepared for Box API: {json.dumps(final_metadata_for_box, default=str)}")
        
        # --- End of Metadata Preparation Steps --- 

        # If no metadata values remain after preparation, return error
        if not final_metadata_for_box:
            logger.warning(f"No valid metadata found for file {file_name} ({file_id}) after preparation")
            return {
                "file_id": file_id,
                "file_name": file_name,
                "success": False,
                "error": "No valid metadata found after preparation steps"
            }
        
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
                    properties_metadata = final_metadata_for_box # Use the fully prepared metadata
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
                        try:
                            properties_metadata = final_metadata_for_box # Use the fully prepared metadata
                            logger.info(f"Updating properties metadata with: {json.dumps(properties_metadata, default=str)}")
                            
                            operations = []
                            for key, value in properties_metadata.items():
                                operations.append({
                                    "op": "replace",
                                    "path": f"/{key}",
                                    "value": value
                                })
                            
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
            parts = template_id.split("_")
            scope = parts[0]
            enterprise_id = parts[1] if len(parts) > 1 else ""
            template_key = parts[2] if len(parts) > 2 else parts[-1]
            scope_with_id = f"{scope}_{enterprise_id}"
            
            logger.info(f"Using template-based metadata application with scope: {scope_with_id}, template: {template_key}")
            
            try:
                template_metadata = final_metadata_for_box # Use the fully prepared metadata
                logger.info(f"Sending template metadata to Box API: {json.dumps(template_metadata, default=str)}")
                
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
                    try:
                        template_metadata = final_metadata_for_box # Use the fully prepared metadata
                        logger.info(f"Updating template metadata with: {json.dumps(template_metadata, default=str)}")
                        
                        operations = []
                        for key, value in template_metadata.items():
                            operations.append({
                                "op": "replace",
                                "path": f"/{key}",
                                "value": value
                            })
                        
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
            # Apply metadata as properties (non-structured extraction)
            try:
                properties_metadata = final_metadata_for_box # Use the fully prepared metadata
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
                    try:
                        properties_metadata = final_metadata_for_box # Use the fully prepared metadata
                        logger.info(f"Updating properties metadata (non-structured) with: {json.dumps(properties_metadata, default=str)}")
                        
                        operations = []
                        for key, value in properties_metadata.items():
                            operations.append({
                                "op": "replace",
                                "path": f"/{key}",
                                "value": value
                            })
                        
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
        logger.exception(f"Unexpected error applying metadata to file {file_id}: {str(e)}")
        return {
            "file_id": file_id,
            "file_name": file_id_to_file_name.get(file_id, "Unknown"),
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }

def apply_metadata_direct():
    """
    Apply metadata to selected files using the direct client reference
    """
    if not hasattr(st.session_state, "selected_result_ids") or not st.session_state.selected_result_ids:
        st.warning("No files selected for metadata application.")
        return
    
    if not hasattr(st.session_state, "extraction_results") or not st.session_state.extraction_results:
        st.error("No extraction results available to apply.")
        return
    
    if not hasattr(st.session_state, "client") or not st.session_state.client:
        st.error("Box client not initialized. Please authenticate.")
        return
    
    client = st.session_state.client
    results = []
    
    # Create file ID to name mapping for better logging
    file_id_to_name = {}
    if hasattr(st.session_state, "selected_files"):
        for file in st.session_state.selected_files:
            file_id_to_name[file["id"]] = file["name"]
    
    # Determine if normalization and placeholder filtering are needed
    # These could be made configurable later
    normalize_keys = True
    filter_placeholders = True
    
    with st.spinner("Applying metadata..."):
        for file_id in st.session_state.selected_result_ids:
            if file_id in st.session_state.extraction_results:
                # Get the processed result data (which should contain only values, not confidence)
                # We need to ensure the data passed here is the actual extracted values
                # Let\'s re-process from the original result if available
                original_result = st.session_state.extraction_results[file_id]
                
                # Re-extract the actual data values, excluding confidence
                metadata_to_apply = {}
                if isinstance(original_result, dict):
                    if "answer" in original_result:
                        answer = original_result["answer"]
                        if isinstance(answer, str):
                            try:
                                parsed_answer = json.loads(answer.replace("\"", "\""))
                                if isinstance(parsed_answer, dict):
                                    for key, value in parsed_answer.items():
                                        if isinstance(value, dict) and "value" in value:
                                            metadata_to_apply[key] = value["value"]
                                        else:
                                            metadata_to_apply[key] = value
                            except json.JSONDecodeError:
                                metadata_to_apply["extracted_text"] = answer
                        elif isinstance(answer, dict):
                             for key, value in answer.items():
                                if isinstance(value, dict) and "value" in value:
                                    metadata_to_apply[key] = value["value"]
                                else:
                                    metadata_to_apply[key] = value
                    elif "items" in original_result and isinstance(original_result["items"], list) and len(original_result["items"]) > 0:
                         # Similar logic as above for items[0]["answer"]
                         item_answer = original_result["items"][0].get("answer")
                         if isinstance(item_answer, str):
                            try:
                                parsed_answer = json.loads(item_answer.replace("\"", "\""))
                                if isinstance(parsed_answer, dict):
                                    for key, value in parsed_answer.items():
                                        if isinstance(value, dict) and "value" in value:
                                            metadata_to_apply[key] = value["value"]
                                        else:
                                            metadata_to_apply[key] = value
                            except json.JSONDecodeError:
                                metadata_to_apply["extracted_text"] = item_answer
                         elif isinstance(item_answer, dict):
                             for key, value in item_answer.items():
                                if isinstance(value, dict) and "value" in value:
                                    metadata_to_apply[key] = value["value"]
                                else:
                                    metadata_to_apply[key] = value
                    else:
                        # Assume the dictionary itself contains the metadata, filter confidence
                        for key, value in original_result.items():
                            if not key.endswith("_confidence"):
                                metadata_to_apply[key] = value
                else:
                    metadata_to_apply["extracted_text"] = str(original_result)
                
                logger.info(f"Metadata extracted for application to file {file_id}: {json.dumps(metadata_to_apply, default=str)}")
                
                result = apply_metadata_to_file_direct(
                    client,
                    file_id,
                    metadata_to_apply, # Pass the re-extracted values
                    normalize_keys=normalize_keys,
                    filter_placeholders=filter_placeholders,
                    file_id_to_file_name=file_id_to_name
                )
                results.append(result)
            else:
                results.append({
                    "file_id": file_id,
                    "file_name": file_id_to_name.get(file_id, "Unknown"),
                    "success": False,
                    "error": "Extraction result not found"
                })
    
    # Display results
    st.subheader("Metadata Application Results")
    successful_count = sum(1 for r in results if r["success"])
    failed_count = len(results) - successful_count
    
    if successful_count > 0:
        st.success(f"Successfully applied metadata to {successful_count} file(s).")
    if failed_count > 0:
        st.error(f"Failed to apply metadata to {failed_count} file(s):")
        for result in results:
            if not result["success"]:
                st.write(f"- File: {result["file_name"]} ({result["file_id"]}) - Error: {result["error"]}")

