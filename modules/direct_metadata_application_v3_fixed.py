import streamlit as st
import logging
import json
from boxsdk import Client, exception
# Removed incompatible import for SDK v3.x
# from boxsdk.schemas import GetMetadataTemplateScope 
from dateutil import parser
from datetime import timezone

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Cache for template schemas to avoid repeated API calls
if 'template_schema_cache' not in st.session_state:
    st.session_state.template_schema_cache = {}

def get_template_schema(client, scope_str, template_key):
    """
    Fetches the metadata template schema from Box API (compatible with SDK v3.x).
    Uses a cache to avoid redundant API calls.
    Handles scope parameter as string ('enterprise' or 'global').
    
    Args:
        client: Box client object
        scope_str (str): The scope of the template (e.g., 'enterprise_12345' or 'global')
        template_key (str): The key of the template (e.g., 'invoiceData')
        
    Returns:
        dict: A dictionary mapping field keys to their types, or None if error.
    """
    cache_key = f'{scope_str}_{template_key}'
    if cache_key in st.session_state.template_schema_cache:
        logger.info(f"Using cached schema for {scope_str}/{template_key}")
        return st.session_state.template_schema_cache[cache_key]

    try:
        # Determine the correct scope string ('enterprise' or 'global') for SDK v3
        if scope_str.startswith('enterprise'):
            # SDK v3 expects just 'enterprise' for enterprise-scoped templates
            scope_param = 'enterprise'
        elif scope_str == 'global':
            scope_param = 'global'
        else:
            logger.error(f"Unknown scope format: {scope_str}. Cannot determine scope parameter for SDK v3.")
            st.session_state.template_schema_cache[cache_key] = None
            return None

        logger.info(f"Fetching template schema for {scope_str}/{template_key} using scope parameter '{scope_param}'")
        # Use the correct pattern for SDK v3: call metadata_template() then .get()
        template = client.metadata_template(scope_param, template_key).get()
        
        if template and hasattr(template, 'fields') and template.fields:
            # Extract key and type from the field dictionaries (SDK v3 structure)
            schema_map = {field['key']: field['type'] for field in template.fields}
            st.session_state.template_schema_cache[cache_key] = schema_map
            logger.info(f"Successfully fetched and cached schema for {scope_str}/{template_key}")
            return schema_map
        else:
            logger.warning(f"Template {scope_str}/{template_key} found but has no fields or is invalid.")
            st.session_state.template_schema_cache[cache_key] = {}
            return {}
            
    except exception.BoxAPIException as e:
        logger.error(f"Box API Error fetching template schema for {scope_str}/{template_key}: {e}")
        st.session_state.template_schema_cache[cache_key] = None 
        return None
    except Exception as e:
        logger.exception(f"Unexpected error fetching template schema for {scope_str}/{template_key}: {e}")
        st.session_state.template_schema_cache[cache_key] = None
        return None

def convert_value_for_template(key, value, field_type):
    """
    Converts a metadata value to the type specified by the template field.

    Args:
        key (str): The metadata field key.
        value: The original value.
        field_type (str): The target field type ('string', 'float', 'date', 'enum', 'multiSelect').

    Returns:
        Converted value or original value if conversion fails or type is unknown.
    """
    if value is None:
        return None # Keep None as None
        
    original_value_repr = repr(value) # For logging

    try:
        if field_type == 'float':
            # Try converting to float. Handle potential strings like '5,000.00' or '$5000'
            if isinstance(value, str):
                # Remove common currency symbols and commas
                cleaned_value = value.replace('$', '').replace(',', '')
                try:
                    return float(cleaned_value)
                except ValueError:
                    logger.warning(f"Could not convert string '{value}' to float for key '{key}'. Keeping original.")
                    return value # Keep original if conversion fails
            elif isinstance(value, (int, float)):
                return float(value) # Already a number
            else:
                 logger.warning(f"Value {original_value_repr} for key '{key}' is not a string or number, cannot convert to float. Keeping original.")
                 return value
                 
        elif field_type == 'date':
            # Box expects RFC 3339 format, typically YYYY-MM-DDTHH:MM:SSZ or with offset
            if isinstance(value, str):
                try:
                    # Parse the date string using dateutil parser (handles various formats)
                    dt = parser.parse(value)
                    # Format as YYYY-MM-DDTHH:MM:SSZ (UTC)
                    # If timezone naive, assume UTC. If timezone aware, convert to UTC.
                    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    else:
                        dt = dt.astimezone(timezone.utc)
                    # Format to RFC3339 with Z for UTC
                    return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
                except (parser.ParserError, ValueError) as e:
                    logger.warning(f"Could not parse date string '{value}' for key '{key}': {e}. Keeping original.")
                    return value # Keep original if parsing fails
            else:
                logger.warning(f"Value {original_value_repr} for key '{key}' is not a string, cannot convert to date. Keeping original.")
                return value
                
        elif field_type == 'string' or field_type == 'enum':
            # Ensure the value is a string
            if not isinstance(value, str):
                logger.info(f"Converting value {original_value_repr} to string for key '{key}' (type {field_type}).")
                return str(value)
            return value # Already a string
            
        elif field_type == 'multiSelect':
            # Box expects a list of strings for multiSelect
            if isinstance(value, list):
                # Ensure all items in the list are strings
                converted_list = [str(item) for item in value]
                if converted_list != value:
                     logger.info(f"Converting items in list {original_value_repr} to string for key '{key}' (type multiSelect).")
                return converted_list
            elif isinstance(value, str):
                # Treat it as a single selection in a list
                logger.info(f"Converting string value {original_value_repr} to list of strings for key '{key}' (type multiSelect).")
                return [value]
            else:
                # Convert other types to a list containing the string representation
                logger.info(f"Converting value {original_value_repr} to list of strings for key '{key}' (type multiSelect).")
                return [str(value)]
                
        else:
            # Unknown field type, return original value
            logger.warning(f"Unknown field type '{field_type}' for key '{key}'. Keeping original value {original_value_repr}.")
            return value
            
    except Exception as e:
        logger.error(f"Unexpected error converting value {original_value_repr} for key '{key}' (type {field_type}): {e}. Keeping original.")
        return value

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
        # Skip any keys that end with "_confidence"
        if not key.endswith("_confidence"):
            filtered_metadata[key] = value
    
    return filtered_metadata

def apply_metadata_direct():
    """
    Direct approach to apply metadata to Box files with type conversion based on template schema.
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
        # Ensure file_id is added even if only from results_map
        if file_id not in available_file_ids:
             available_file_ids.append(file_id)
             logger.info(f"Added file ID {file_id} from results_map")

        # Most APIs put your AI fields under payload["results"]
        metadata = payload.get("results", payload)
        
        # If metadata is a string that looks like JSON, try to parse it
        if isinstance(metadata, str):
            try:
                parsed_metadata = json.loads(metadata)
                if isinstance(parsed_metadata, dict):
                    metadata = parsed_metadata
            except json.JSONDecodeError:
                pass # Not valid JSON, keep as is
        
        # If payload has an 'answer' field that's a JSON string, parse it
        if isinstance(payload, dict) and 'answer' in payload and isinstance(payload['answer'], str):
            try:
                parsed_answer = json.loads(payload['answer'])
                if isinstance(parsed_answer, dict):
                    # Prefer 'answer' if it parses correctly as a dict
                    metadata = parsed_answer 
            except json.JSONDecodeError:
                pass # Not valid JSON, keep original metadata
        
        # Ensure metadata is a dictionary before proceeding
        if not isinstance(metadata, dict):
            logger.warning(f"Metadata for file {file_id} is not a dictionary: {repr(metadata)}. Skipping.")
            continue # Skip this file if metadata is not a dict
            
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
    st.write("Extracted metadata will be applied based on matching template fields.")
    
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

    # Mapping from document type to template scope/key (Example - needs to be configured)
    # This should ideally be loaded from a config file or environment variables
    doc_type_to_template_map = {
        "Loan document": ("enterprise_336904155", "homeLoan"),
        "Driver License": ("enterprise_336904155", "driverLicense"), # Assuming this template exists
        # Add other mappings as needed
    }

    # Direct function to apply metadata to a single file
    def apply_metadata_to_file_direct(client, file_id, metadata_values):
        """
        Apply metadata to a single file with type conversion based on template schema.
        
        Args:
            client: Box client object
            file_id: File ID to apply metadata to
            metadata_values: Dictionary of metadata values to apply
            
        Returns:
            dict: Result of metadata application
        """
        try:
            file_name = file_id_to_file_name.get(file_id, "Unknown")
            
            # Validate metadata values
            if not metadata_values or not isinstance(metadata_values, dict):
                logger.error(f"Invalid or empty metadata for file {file_name} ({file_id}): {metadata_values!r}")
                return {
                    "file_id": file_id,
                    "file_name": file_name,
                    "success": False,
                    "error": "Invalid or empty metadata received"
                }
            
            logger.info(f"Original metadata for {file_name} ({file_id}): {json.dumps(metadata_values, default=str)}")
            
            # Flatten metadata if needed (assuming 'answer' structure)
            # metadata_values = flatten_metadata_for_template(metadata_values) # Apply flattening if structure requires it
            # logger.info(f"Flattened metadata: {json.dumps(metadata_values, default=str)}")

            # Filter out confidence score fields
            metadata_values = filter_confidence_fields(metadata_values)
            logger.info(f"Metadata after filtering confidence: {json.dumps(metadata_values, default=str)}")
            
            # Filter out placeholder values if requested
            if filter_placeholders:
                original_count = len(metadata_values)
                metadata_values = {k: v for k, v in metadata_values.items() if not is_placeholder(v)}
                if len(metadata_values) < original_count:
                    logger.info(f"Filtered out {original_count - len(metadata_values)} placeholder values.")
            
            if not metadata_values:
                logger.warning(f"No valid metadata left for {file_name} ({file_id}) after filtering.")
                return {
                    "file_id": file_id,
                    "file_name": file_name,
                    "success": False,
                    "error": "No valid metadata after filtering placeholders/confidence"
                }
            
            # --- Determine Template and Fetch Schema ---
            template_scope_str = None
            template_key = None
            doc_type = None

            # --- NEW FIX: Get doc_type from categorization results --- 
            categorization_results = st.session_state.get("document_categorization", {}).get("results", {})
            # Use str(file_id) for lookup consistency
            doc_type = categorization_results.get(str(file_id), {}).get("document_type") # FIX: Use 'document_type' key
            if doc_type:
                logger.info(f"Retrieved document type 	'{doc_type}'	 from categorization results for file {file_id}")
            else:
                logger.warning(f"Could not retrieve document type from categorization results for file {file_id}. Template mapping might fail.")
            # --- END NEW FIX ---

            # --- Use the retrieved doc_type for template mapping ---
            # Get the mapping from session state (assuming it's loaded elsewhere)
            doc_type_to_template_map = st.session_state.get("document_type_to_template", {})
            # --- ADDED LOGGING: Log retrieved doc_type and available mapping keys ---
            logger.info(f"Retrieved doc_type for mapping lookup: 	'{doc_type}'")
            logger.info(f"Available keys in doc_type_to_template_map: {list(doc_type_to_template_map.keys())}")
            # --- END ADDED LOGGING ---
            
            if doc_type and doc_type in doc_type_to_template_map:
                template_id = doc_type_to_template_map[doc_type]
                if template_id:
                    # Parse the template_id (e.g., "enterprise_12345_myTemplate" or "enterprise_myTemplate")
                    parts = template_id.split('_')
                    if len(parts) >= 2:
                        scope = parts[0]
                        key = '_'.join(parts[1:])
                        # Handle potential enterprise ID in scope like "enterprise_12345"
                        if scope.startswith('enterprise') and len(parts) > 2 and parts[1].isdigit():
                            template_scope_str = f"{parts[0]}_{parts[1]}" # e.g., enterprise_12345
                            template_key = '_'.join(parts[2:])
                        else:
                            template_scope_str = scope # e.g., enterprise or global
                            template_key = key
                        logger.info(f"File {file_name} identified as '{doc_type}', using template {template_scope_str}/{template_key}")
                    else:
                        logger.error(f"Invalid template ID format '{template_id}' found in mapping for doc type '{doc_type}'.")
                else:
                    logger.warning(f"Template mapping found for doc type '{doc_type}', but the template ID is empty or None.")
            else:
                logger.warning(f"No template mapping found in session state for document type: {doc_type or 'Unknown'}")

            # --- Fallback or error if template not determined ---
            if not template_scope_str or not template_key:
                # Fallback to default template from config if available
                default_template_id = st.session_state.get("metadata_config", {}).get("template_id")
                if default_template_id:
                    logger.warning(f"Falling back to default template ID: {default_template_id}")
                    parts = default_template_id.split('_')
                    if len(parts) >= 2:
                        scope = parts[0]
                        key = '_'.join(parts[1:])
                        if scope.startswith('enterprise') and len(parts) > 2 and parts[1].isdigit():
                            template_scope_str = f"{parts[0]}_{parts[1]}"
                            template_key = '_'.join(parts[2:])
                        else:
                            template_scope_str = scope
                            template_key = key
                        logger.info(f"Using default template {template_scope_str}/{template_key}")
                    else:
                         logger.error(f"Invalid default template ID format: {default_template_id}")
                else:
                    logger.error(f"Could not determine template (specific or default) for file {file_name}. Cannot apply template metadata.")
                    return {
                        "file_id": file_id,
                        "file_name": file_name,
                        "success": False,
                        "error": f"Could not determine metadata template for document type: {doc_type or 'Unknown'}"
                    }

            # Fetch the template schema using the scope string
            template_schema = get_template_schema(client, template_scope_str, template_key)
            
            if template_schema is None:
                logger.error(f"Failed to fetch or invalid schema for template {template_scope_str}/{template_key}. Cannot apply metadata.")
                return {
                    "file_id": file_id,
                    "file_name": file_name,
                    "success": False,
                    "error": f"Failed to fetch schema for template {template_scope_str}/{template_key}"
                }
            elif not template_schema:
                 logger.warning(f"Template schema for {template_scope_str}/{template_key} is empty. Applying metadata without type conversion.")
                 converted_metadata = metadata_values.copy()
            else:
                # --- Convert values based on schema ---
                converted_metadata = {}
                for key, value in metadata_values.items():
                    if key in template_schema:
                        field_type = template_schema[key]
                        converted_value = convert_value_for_template(key, value, field_type)
                        # Only include if the converted value is not None
                        if converted_value is not None:
                            converted_metadata[key] = converted_value
                            logger.info(f"Added field \t\'{key}\'\t with converted value: {repr(converted_value)}") # Added logging for clarity
                        else:
                            logger.warning(f"Skipping field \t\'{key}\'\t because its value is None after conversion (original: {repr(value)}).")
                    else:
                        logger.warning(f"Key '{key}' from extracted metadata not found in template {template_scope_str}/{template_key}. Skipping this field.")
            
            if not converted_metadata:
                logger.warning(f"No metadata fields remaining after type conversion and validation for {file_name} ({file_id}).")
                return {
                    "file_id": file_id,
                    "file_name": file_name,
                    "success": False,
                    "error": "No metadata fields applicable to the template after conversion"
                }

            # Determine scope parameter for the update call ("enterprise" or "global")
            if template_scope_str.startswith("enterprise"):
                scope_param_for_update = "enterprise"
            elif template_scope_str == "global":
                scope_param_for_update = "global"
            else:
                logger.error(f"Invalid scope string 	{template_scope_str}	 for metadata update.")
                return {
                    "file_id": file_id,
                    "file_name": file_name,
                    "success": False,
                    "error": f"Invalid scope string for update: {template_scope_str}"
                }

            # --- Apply Metadata to Box --- 
            logger.info(f"Applying CONVERTED metadata to {file_name} ({file_id}) using template {template_scope_str}/{template_key}: {json.dumps(converted_metadata, default=str)}")
            
            # --- ADDED LOGGING: Log the scope parameter being used for the API call ---
            logger.info(f"Using scope parameter 	'{scope_param_for_update}'	 for metadata API call (template: {template_scope_str}/{template_key})")
            # --- END ADDED LOGGING ---

            try:
                # Get the metadata resource object
                metadata_resource = client.file(file_id=file_id).metadata(
                    scope=scope_param_for_update, 
                    template=template_key
                )

                # Check if metadata instance already exists
                try:
                    existing_metadata = metadata_resource.get()
                    logger.info(f"Metadata instance exists for {file_name} ({file_id}), performing update.")
                    
                    # Start update and add replace operations using JSON Pointer paths
                    updates = metadata_resource.start_update()
                    for key, value in converted_metadata.items():
                        updates.replace(f"/{key}", value) # Use JSON Pointer format /key
                    
                    # Apply the updates
                    metadata_instance = metadata_resource.update(updates)
                    
                except exception.BoxAPIException as e:
                    if e.status == 404:
                        # Metadata instance doesn't exist, create it
                        logger.info(f"Metadata instance does not exist for {file_name} ({file_id}), creating new instance.")
                        metadata_instance = metadata_resource.create(converted_metadata)
                    else:
                        # Re-raise other API errors
                        raise

                logger.info(f"Successfully applied/updated template metadata for {file_name} ({file_id})")
                return {
                    "file_id": file_id,
                    "file_name": file_name,
                    "success": True,
                    "metadata_applied": converted_metadata
                }
            except exception.BoxAPIException as e:
                logger.error(f"Box API Error applying template metadata for {file_name} ({file_id}): {e}")
                # Provide more detailed error info if available
                error_details = e.context_info or {}
                error_message = e.message or "Unknown Box API Error"
                return {
                    "file_id": file_id,
                    "file_name": file_name,
                    "success": False,
                    "error": f"Box API Error: {error_message} (Status: {e.status}, Code: {e.code})",
                    "details": error_details
                }

        except Exception as e:
            logger.exception(f"Unexpected error in apply_metadata_to_file_direct for {file_id}: {e}")
            return {
                "file_id": file_id,
                "file_name": file_id_to_file_name.get(file_id, "Unknown"),
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }

    # Main execution logic when 'Apply Metadata' is clicked
    if apply_button:
        st.session_state.apply_results = []
        st.session_state.apply_errors = []
        total_files = len(available_file_ids)
        
        progress_bar = progress_container.progress(0)
        status_text = progress_container.empty()
        
        for i, file_id in enumerate(available_file_ids):
            file_name = file_id_to_file_name.get(file_id, "Unknown")
            status_text.text(f"Processing file {i+1}/{total_files}: {file_name} ({file_id})...")
            
            metadata_to_apply = file_id_to_metadata.get(file_id)
            
            if not metadata_to_apply:
                logger.warning(f"No metadata found in mapping for file {file_name} ({file_id}). Skipping.")
                st.session_state.apply_errors.append({
                    "file_id": file_id,
                    "file_name": file_name,
                    "error": "No metadata found in processing results"
                })
                continue
                
            # Apply metadata to the single file
            result = apply_metadata_to_file_direct(client, file_id, metadata_to_apply)
            
            if result['success']:
                st.session_state.apply_results.append(result)
            else:
                st.session_state.apply_errors.append(result)
            
            # Update progress
            progress_bar.progress((i + 1) / total_files)
        
        status_text.text(f"Metadata application complete for {total_files} files.")
        
        # Display results
        st.subheader("Application Results")
        
        if st.session_state.apply_results:
            st.success(f"Successfully applied metadata to {len(st.session_state.apply_results)} files.")
            with st.expander("View Success Details"):
                for res in st.session_state.apply_results:
                    st.write(f"- **{res['file_name']} ({res['file_id']})**: Applied {len(res['metadata_applied'])} fields.")
                    # st.json(res['metadata_applied']) # Optional: show applied data
        
        if st.session_state.apply_errors:
            st.error(f"Failed to apply metadata to {len(st.session_state.apply_errors)} files.")
            with st.expander("View Error Details"):
                for err in st.session_state.apply_errors:
                    st.write(f"- **{err['file_name']} ({err['file_id']})**: {err['error']}")
                    if 'details' in err and err['details']:
                         st.json(err['details'])

    if cancel_button:
        st.warning("Metadata application cancelled.")
        # Optionally clear state or redirect
        if 'apply_results' in st.session_state: del st.session_state.apply_results
        if 'apply_errors' in st.session_state: del st.session_state.apply_errors
        st.rerun()

# Example usage (if run directly, though it's meant to be called by Streamlit)
if __name__ == '__main__':
    # This part is mostly for structure; Streamlit handles the execution flow
    # You would typically run this via `streamlit run your_app.py`
    pass

