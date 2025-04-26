import streamlit as st
import pandas as pd
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def render_per_file_metadata_config(selected_files: List[Dict[str, Any]], available_templates: List[Dict[str, Any]]):
    """
    Render the per-file metadata configuration UI component.
    
    Args:
        selected_files: List of selected file objects with id and name
        available_templates: List of available metadata templates
    """
    st.title("Metadata Configuration")
    
    if not selected_files:
        st.warning("No files selected. Please select files first.")
        return
    
    # Display document categorization results
    st.header("Document Categorization Results")
    
    # Create a dataframe for better display
    file_data = []
    for i, file_info in enumerate(selected_files):
        file_id = file_info.get("id", "")
        file_name = file_info.get("name", "Unknown")
        doc_type = file_info.get("document_type", "Unknown")
        file_data.append({"index": i, "file_id": file_id, "file_name": file_name, "document_type": doc_type})
    
    # Display as a table
    df = pd.DataFrame(file_data)
    st.dataframe(
        df[["file_name", "document_type"]],
        column_config={
            "file_name": "File Name",
            "document_type": "Document Type"
        },
        hide_index=True
    )
    
    # Initialize per-file configuration in session state if not exists
    if "file_metadata_config" not in st.session_state:
        st.session_state.file_metadata_config = {}
    
    # Ensure all selected files have a configuration
    for file_info in selected_files:
        file_id = file_info.get("id", "")
        if file_id and file_id not in st.session_state.file_metadata_config:
            st.session_state.file_metadata_config[file_id] = {
                "extraction_method": "structured",  # Default to structured
                "template_id": "",
                "custom_prompt": ""
            }
    
    # Per-file configuration section
    st.header("Per-File Extraction Configuration")
    st.info("Configure extraction method and template for each file individually.")
    
    # Create tabs for each file
    file_tabs = st.tabs([f"{file_info.get('name', 'File')} ({i+1}/{len(selected_files)})" 
                         for i, file_info in enumerate(selected_files)])
    
    for i, (tab, file_info) in enumerate(zip(file_tabs, selected_files)):
        file_id = file_info.get("id", "")
        file_name = file_info.get("name", "Unknown")
        doc_type = file_info.get("document_type", "Unknown")
        
        with tab:
            st.subheader(f"Configuration for: {file_name}")
            st.write(f"Document Type: {doc_type}")
            
            # Get current config for this file
            file_config = st.session_state.file_metadata_config.get(file_id, {})
            
            # Extraction method selection
            extraction_method = st.radio(
                "Select extraction method",
                options=["Structured", "Freeform"],
                index=0 if file_config.get("extraction_method", "structured") == "structured" else 1,
                key=f"extraction_method_{file_id}",
                horizontal=True
            )
            
            # Update the extraction method in session state
            file_config["extraction_method"] = extraction_method.lower()
            
            # Different options based on extraction method
            if extraction_method.lower() == "structured":
                st.subheader("Structured Extraction Configuration")
                
                # Template selection
                template_options = [""] + [t.get("id", "") for t in available_templates]
                template_labels = ["Select a template..."] + [t.get("displayName", t.get("id", "")) for t in available_templates]
                
                template_index = 0
                current_template = file_config.get("template_id", "")
                if current_template in template_options:
                    template_index = template_options.index(current_template)
                
                selected_template = st.selectbox(
                    "Select Metadata Template",
                    options=template_options,
                    format_func=lambda x: template_labels[template_options.index(x)] if x in template_options else x,
                    index=template_index,
                    key=f"template_select_{file_id}"
                )
                
                # Update template in session state
                file_config["template_id"] = selected_template
                file_config["custom_prompt"] = ""  # Clear custom prompt when using structured
                
                # Show template details if selected
                if selected_template:
                    template_info = next((t for t in available_templates if t.get("id") == selected_template), None)
                    if template_info:
                        st.info(f"Template: {template_info.get('displayName', template_info.get('id', ''))}")
                        
                        # Show fields if available
                        fields = template_info.get("fields", [])
                        if fields:
                            st.write("Template Fields:")
                            field_data = []
                            for field in fields:
                                field_data.append({
                                    "key": field.get("key", ""),
                                    "displayName": field.get("displayName", field.get("key", "")),
                                    "type": field.get("type", "string"),
                                    "required": "Yes" if field.get("hidden", False) else "No"
                                })
                            
                            st.dataframe(
                                pd.DataFrame(field_data),
                                column_config={
                                    "key": "Field Key",
                                    "displayName": "Display Name",
                                    "type": "Type",
                                    "required": "Required"
                                },
                                hide_index=True
                            )
            else:
                st.subheader("Freeform Extraction Configuration")
                
                # Custom prompt for freeform extraction
                custom_prompt = st.text_area(
                    "Custom Extraction Prompt",
                    value=file_config.get("custom_prompt", ""),
                    height=150,
                    key=f"custom_prompt_{file_id}",
                    help="Enter a custom prompt for extracting metadata from this file."
                )
                
                # Update custom prompt in session state
                file_config["custom_prompt"] = custom_prompt
                file_config["template_id"] = ""  # Clear template when using freeform
            
            # Save the updated config for this file
            st.session_state.file_metadata_config[file_id] = file_config
    
    # Summary section
    st.header("Configuration Summary")
    
    # Create a summary table
    summary_data = []
    for file_info in selected_files:
        file_id = file_info.get("id", "")
        file_name = file_info.get("name", "Unknown")
        file_config = st.session_state.file_metadata_config.get(file_id, {})
        
        extraction_method = file_config.get("extraction_method", "structured")
        template_id = file_config.get("template_id", "")
        custom_prompt = file_config.get("custom_prompt", "")
        
        # Get template display name if available
        template_name = template_id
        if template_id:
            template_info = next((t for t in available_templates if t.get("id") == template_id), None)
            if template_info:
                template_name = template_info.get("displayName", template_id)
        
        # Format the configuration details
        config_details = template_name if extraction_method == "structured" else "Custom prompt"
        
        summary_data.append({
            "file_name": file_name,
            "extraction_method": extraction_method.capitalize(),
            "config_details": config_details
        })
    
    # Display summary table
    st.dataframe(
        pd.DataFrame(summary_data),
        column_config={
            "file_name": "File Name",
            "extraction_method": "Extraction Method",
            "config_details": "Template/Prompt"
        },
        hide_index=True
    )
    
    # Save configuration button
    if st.button("Save Configuration", use_container_width=True):
        st.success("Configuration saved successfully!")
        logger.info(f"Saved per-file metadata configuration for {len(selected_files)} files")
        
        # Log the configuration for debugging
        for file_id, config in st.session_state.file_metadata_config.items():
            logger.info(f"File ID: {file_id}, Config: {config}")
        
        # Store the overall extraction method as per-file for backward compatibility
        st.session_state.metadata_config = {
            "extraction_method": "per_file",
            "use_template": True
        }

def get_file_specific_config(file_id: str) -> Dict[str, Any]:
    """
    Get the specific metadata configuration for a file.
    
    Args:
        file_id: The ID of the file to get configuration for
        
    Returns:
        Dict containing the file's metadata configuration
    """
    if "file_metadata_config" not in st.session_state:
        return {
            "extraction_method": "structured",
            "template_id": "",
            "custom_prompt": ""
        }
    
    return st.session_state.file_metadata_config.get(file_id, {
        "extraction_method": "structured",
        "template_id": "",
        "custom_prompt": ""
    })

def process_file_with_specific_config(file_id: str, file_name: str, client: Any) -> Dict[str, Any]:
    """
    Process a file using its specific metadata configuration.
    
    Args:
        file_id: The ID of the file to process
        file_name: The name of the file
        client: Box client object
        
    Returns:
        Dict containing the processing results
    """
    # Get file-specific configuration
    file_config = get_file_specific_config(file_id)
    extraction_method = file_config.get("extraction_method", "structured")
    
    logger.info(f"Processing file {file_name} ({file_id}) with {extraction_method} extraction method")
    
    if extraction_method == "structured":
        template_id = file_config.get("template_id", "")
        if not template_id:
            logger.warning(f"No template selected for structured extraction of file {file_name} ({file_id})")
            return {
                "file_id": file_id,
                "file_name": file_name,
                "success": False,
                "error": "No template selected for structured extraction"
            }
        
        # Call structured extraction function
        # This would be implemented elsewhere and called here
        logger.info(f"Using template {template_id} for structured extraction of file {file_name} ({file_id})")
        # result = extract_structured_metadata(client, file_id, template_id)
        
    else:  # freeform
        custom_prompt = file_config.get("custom_prompt", "")
        if not custom_prompt:
            logger.warning(f"No custom prompt provided for freeform extraction of file {file_name} ({file_id})")
            return {
                "file_id": file_id,
                "file_name": file_name,
                "success": False,
                "error": "No custom prompt provided for freeform extraction"
            }
        
        # Call freeform extraction function
        # This would be implemented elsewhere and called here
        logger.info(f"Using custom prompt for freeform extraction of file {file_name} ({file_id})")
        # result = extract_freeform_metadata(client, file_id, custom_prompt)
    
    # This is a placeholder - the actual implementation would call the appropriate extraction function
    # and return the real results
    return {
        "file_id": file_id,
        "file_name": file_name,
        "success": True,
        "extraction_method": extraction_method,
        "config": file_config
    }
