import streamlit as st
import pandas as pd
from typing import Dict, List, Any
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_confidence_color(confidence_level):
    """Get color based on confidence level."""
    if confidence_level == "High":
        return "green"
    elif confidence_level == "Medium":
        return "orange"
    elif confidence_level == "Low":
        return "red"
    else:
        return "gray"

def view_results():
    """
    View and manage extraction results - ENHANCED WITH CONFIDENCE SCORES
    """
    st.title("View Results")
    
    # Validate session state
    if not hasattr(st.session_state, "authenticated") or not hasattr(st.session_state, "client") or not st.session_state.authenticated or not st.session_state.client:
        st.error("Please authenticate with Box first")
        return
    
    # Ensure extraction_results is initialized
    if not hasattr(st.session_state, "extraction_results"):
        st.session_state.extraction_results = {}
        logger.info("Initialized extraction_results in view_results")
    
    # Ensure selected_result_ids is initialized
    if not hasattr(st.session_state, "selected_result_ids"):
        st.session_state.selected_result_ids = []
        logger.info("Initialized selected_result_ids in view_results")
    
    # Ensure metadata_config is initialized
    if not hasattr(st.session_state, "metadata_config"):
        st.session_state.metadata_config = {
            "extraction_method": "freeform",
            "freeform_prompt": "Extract key metadata from this document.",
            "use_template": False,
            "template_id": "",
            "custom_fields": [],
            "ai_model": "azure__openai__gpt_4o_mini",
            "batch_size": 5
        }
        logger.info("Initialized metadata_config in view_results")
    
    if not hasattr(st.session_state, "extraction_results") or not st.session_state.extraction_results:
        st.warning("No extraction results available. Please process files first.")
        if st.button("Go to Process Files", key="go_to_process_files_btn"):
            st.session_state.current_page = "Process Files"
            st.rerun()
        return
    
    st.write("Review and manage the metadata extraction results.")
    
    # Initialize session state for results viewer
    if not hasattr(st.session_state, "results_filter"):
        st.session_state.results_filter = ""
    if not hasattr(st.session_state, "confidence_filter"):
        st.session_state.confidence_filter = ["High", "Medium", "Low"] # Default to show all
    
    # Filter options
    st.subheader("Filter Results")
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.results_filter = st.text_input(
            "Filter by file name",
            value=st.session_state.results_filter,
            key="filter_input"
        )
    with col2:
        st.session_state.confidence_filter = st.multiselect(
            "Filter by Confidence Level",
            options=["High", "Medium", "Low"],
            default=st.session_state.confidence_filter,
            key="confidence_filter_select"
        )
    
    # Get filtered results
    filtered_results = {}
    
    # Process extraction_results to prepare for display
    for file_id, result in st.session_state.extraction_results.items():
        # Create a standardized result structure
        processed_result = {
            "file_id": file_id,
            "file_name": "Unknown",
            "result_data": {},
            "confidence_levels": {}
        }
        
        # Try to find file name
        if hasattr(st.session_state, "selected_files"):
            for file in st.session_state.selected_files:
                if file["id"] == file_id:
                    processed_result["file_name"] = file["name"]
                    break
        
        # Log the raw result for debugging
        logger.info(f"Processing result for file_id {file_id}: {json.dumps(result) if isinstance(result, dict) else str(result)}")
        
        # Process the result data based on its structure
        if isinstance(result, dict):
            # Store the original result
            processed_result["original_data"] = result
            
            # Check if this is a direct API response with an answer field
            if "answer" in result:
                answer = result["answer"]
                logger.info(f"Found \'answer\' field in result: {answer}")
                
                # Check if answer is a JSON string that needs parsing
                if isinstance(answer, str):
                    try:
                        parsed_answer = json.loads(answer)
                        if isinstance(parsed_answer, dict):
                            logger.info(f"Successfully parsed answer as JSON dictionary: {parsed_answer}")
                            # Process parsed answer for value and confidence
                            for key, value in parsed_answer.items():
                                if isinstance(value, dict) and "value" in value and "confidence" in value:
                                    processed_result["result_data"][key] = value["value"]
                                    processed_result["confidence_levels"][key] = value["confidence"]
                                    logger.info(f"Extracted field {key} with value \'{value['value']}\' and confidence \'{value['confidence']}\'")
                                else:
                                    processed_result["result_data"][key] = value
                                    processed_result["confidence_levels"][key] = "Medium" # Default
                                    logger.info(f"Field {key} doesn\'t have expected structure, using value \'{value}\' with default Medium confidence")
                        else:
                            logger.warning(f"Parsed answer is not a dictionary: {parsed_answer}")
                            processed_result["result_data"] = {"extracted_text": answer}
                    except json.JSONDecodeError as e:
                        # Not valid JSON, treat as text
                        logger.warning(f"Failed to parse answer as JSON: {e}. Using raw text.")
                        processed_result["result_data"] = {"extracted_text": answer}
                elif isinstance(answer, dict):
                    # Already a dictionary, process for value and confidence
                    logger.info(f"Answer is already a dictionary: {answer}")
                    for key, value in answer.items():
                        if isinstance(value, dict) and "value" in value and "confidence" in value:
                            processed_result["result_data"][key] = value["value"]
                            processed_result["confidence_levels"][key] = value["confidence"]
                            logger.info(f"Extracted field {key} with value \'{value['value']}\' and confidence \'{value['confidence']}\'")
                        else:
                            processed_result["result_data"][key] = value
                            processed_result["confidence_levels"][key] = "Medium" # Default
                            logger.info(f"Field {key} doesn\'t have expected structure, using value \'{value}\' with default Medium confidence")
                else:
                    # Some other format, store as is
                    logger.warning(f"Answer is neither string nor dictionary: {type(answer)}. Using as is.")
                    processed_result["result_data"] = {"extracted_text": str(answer)}
            
            # Check for items array with answer field (common in Box AI responses)
            elif "items" in result and isinstance(result["items"], list) and len(result["items"]) > 0:
                item = result["items"][0]
                logger.info(f"Found \'items\' array in result, processing first item: {item}")
                if isinstance(item, dict) and "answer" in item:
                    answer = item["answer"]
                    logger.info(f"Found \'answer\' field in item: {answer}")
                    
                    # Check if answer is a JSON string that needs parsing
                    if isinstance(answer, str):
                        try:
                            parsed_answer = json.loads(answer)
                            if isinstance(parsed_answer, dict):
                                logger.info(f"Successfully parsed item answer as JSON dictionary: {parsed_answer}")
                                # Process parsed answer for value and confidence
                                for key, value in parsed_answer.items():
                                    if isinstance(value, dict) and "value" in value and "confidence" in value:
                                        processed_result["result_data"][key] = value["value"]
                                        processed_result["confidence_levels"][key] = value["confidence"]
                                        logger.info(f"Extracted field {key} with value \'{value['value']}\' and confidence \'{value['confidence']}\'")
                                    else:
                                        processed_result["result_data"][key] = value
                                        processed_result["confidence_levels"][key] = "Medium" # Default
                                        logger.info(f"Field {key} doesn\'t have expected structure, using value \'{value}\' with default Medium confidence")
                            else:
                                logger.warning(f"Parsed item answer is not a dictionary: {parsed_answer}")
                                processed_result["result_data"] = {"extracted_text": answer}
                        except json.JSONDecodeError as e:
                            # Not valid JSON, treat as text
                            logger.warning(f"Failed to parse item answer as JSON: {e}. Using raw text.")
                            processed_result["result_data"] = {"extracted_text": answer}
                    elif isinstance(answer, dict):
                        # Already a dictionary, process for value and confidence
                        logger.info(f"Item answer is already a dictionary: {answer}")
                        for key, value in answer.items():
                            if isinstance(value, dict) and "value" in value and "confidence" in value:
                                processed_result["result_data"][key] = value["value"]
                                processed_result["confidence_levels"][key] = value["confidence"]
                                logger.info(f"Extracted field {key} with value \'{value['value']}\' and confidence \'{value['confidence']}\'")
                            else:
                                processed_result["result_data"][key] = value
                                processed_result["confidence_levels"][key] = "Medium" # Default
                                logger.info(f"Field {key} doesn\'t have expected structure, using value \'{value}\' with default Medium confidence")
                    else:
                        # Some other format, store as is
                        logger.warning(f"Item answer is neither string nor dictionary: {type(answer)}. Using as is.")
                        processed_result["result_data"] = {"extracted_text": str(answer)}
            
            # Check for fields with _confidence suffix (from our metadata_extraction update)
            elif any(key.endswith("_confidence") for key in result.keys()):
                logger.info(f"Found fields with _confidence suffix in result")
                confidence_fields = [key for key in result.keys() if key.endswith("_confidence")]
                logger.info(f"Confidence fields: {confidence_fields}")
                
                for key, value in result.items():
                    if key.endswith("_confidence"):
                        base_key = key[:-len("_confidence")]
                        if base_key in result:
                            processed_result["result_data"][base_key] = result[base_key]
                            processed_result["confidence_levels"][base_key] = value
                            logger.info(f"Extracted field {base_key} with value \'{result[base_key]}\' and confidence \'{value}\'")
                    elif not key.startswith("_") and not any(key == field[:-len("_confidence")] for field in confidence_fields):
                        # Field without confidence, add it with default
                        processed_result["result_data"][key] = value
                        processed_result["confidence_levels"][key] = "Medium" # Default
                        logger.info(f"Field {key} has no confidence field, using value \'{value}\' with default Medium confidence")
            
            # If no structured data found, check for other fields that might contain data
            if not processed_result["result_data"]:
                logger.warning(f"No structured data found in result, looking for alternative fields")
                # Look for any fields that might contain extracted data
                for key in ["extracted_data", "data", "result", "metadata"]:
                    if key in result and result[key]:
                        logger.info(f"Found potential data in field \'{key}\': {result[key]}")
                        if isinstance(result[key], dict):
                            processed_result["result_data"] = result[key]
                            # Add default confidence for all fields
                            for field_key in result[key].keys():
                                if field_key not in processed_result["confidence_levels"]:
                                    processed_result["confidence_levels"][field_key] = "Medium"
                            break
                        elif isinstance(result[key], str):
                            try:
                                parsed_data = json.loads(result[key])
                                if isinstance(parsed_data, dict):
                                    processed_result["result_data"] = parsed_data
                                    # Add default confidence for all fields
                                    for field_key in parsed_data.keys():
                                        if field_key not in processed_result["confidence_levels"]:
                                            processed_result["confidence_levels"][field_key] = "Medium"
                                    break
                            except json.JSONDecodeError:
                                processed_result["result_data"] = {"extracted_text": result[key]}
                                processed_result["confidence_levels"]["extracted_text"] = "Medium"
                                break
                
                # If still no result_data, use the entire result as is
                if not processed_result["result_data"]:
                    logger.warning(f"No structured data found in any expected fields, using entire result")
                    processed_result["result_data"] = result
                    # Add default confidence for all non-internal fields
                    for field_key in result.keys():
                        if not field_key.startswith("_") and not field_key.endswith("_confidence"):
                            processed_result["confidence_levels"][field_key] = "Medium"
        else:
            # Not a dictionary, store as text
            logger.warning(f"Result is not a dictionary: {type(result)}. Using as text.")
            processed_result["result_data"] = {"extracted_text": str(result)}
            processed_result["confidence_levels"]["extracted_text"] = "Medium"
        
        # Filter based on file name
        if st.session_state.results_filter.lower() in processed_result["file_name"].lower():
            # Filter based on confidence level
            if not st.session_state.confidence_filter: # If no confidence filter selected, show all
                filtered_results[file_id] = processed_result
            else:
                # Check if any field has a confidence level matching the filter
                has_matching_confidence = False
                for confidence in processed_result["confidence_levels"].values():
                    if confidence in st.session_state.confidence_filter:
                        has_matching_confidence = True
                        break
                if has_matching_confidence:
                    filtered_results[file_id] = processed_result
    
    # Prepare data for table view
    table_data = []
    for file_id, data in filtered_results.items():
        row = {"File Name": data["file_name"], "File ID": file_id}
        # Add extracted data and confidence to the row
        for key, value in data["result_data"].items():
            if not key.startswith("_") and key != "extracted_text": # Skip internal/text fields
                row[key] = value
                confidence_key = f"{key} Confidence"
                row[confidence_key] = data["confidence_levels"].get(key, "N/A")
        table_data.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(table_data)
    
    # Reorder columns: File Name, File ID, then pairs of (field, field Confidence)
    if not df.empty:
        base_cols = ["File Name", "File ID"]
        field_cols = sorted([col for col in df.columns if col not in base_cols and not col.endswith(" Confidence")])
        ordered_cols = base_cols + [item for field in field_cols for item in (field, f"{field} Confidence") if item in df.columns]
        df = df[ordered_cols]
    
    # Display results in tabs
    st.subheader("Extraction Results")
    tab1, tab2 = st.tabs(["Table View", "Detailed View"])
    
    # Store filtered results for detailed view
    final_filtered_results = filtered_results
    
    with tab1:
        st.write(f"Showing {len(df)} of {len(st.session_state.extraction_results)} results")
        
        if not df.empty:
            # Display table with confidence highlighting
            st.dataframe(
                df.style.applymap(
                    lambda x: f"color: {get_confidence_color(x)}", 
                    subset=[col for col in df.columns if col.endswith(" Confidence")]
                ),
                use_container_width=True,
                hide_index=True,
                # Enable selection
                # on_select="rerun", # Trigger rerun on selection change
                # selection_mode="multi-row"
            )
            
            # Export options
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Export as CSV", use_container_width=True, key="export_csv_btn"):
                    pass # Placeholder to fix indentation after commenting out download button
                    # In a real app, we would save to a file
                    # Temporarily commented out due to persistent deployment error
                    # st.download_button(
                    #     label="Download CSV",
                    #     data=df.to_csv(index=False).encode("utf-8"),
                    #     file_name="extraction_results.csv",
                    #     mime="text/csv",
                    #     key="download_csv_btn"
                    # )
            
            with col2:
                if st.button("Export as Excel", use_container_width=True, key="export_excel_btn"):
                    # In a real app, we would save to a file
                    st.info("Excel export would be implemented in the full app")
        else:
            st.info("No results match the current filter")
    
    with tab2:
        # COMPLETELY REDESIGNED DETAILED VIEW TO AVOID NESTED EXPANDERS
        # Instead of using expanders for each file, use a selectbox to choose which file to view
        file_options = [(file_id, result_data.get("file_name", "Unknown")) 
                        for file_id, result_data in final_filtered_results.items()]
        
        if not file_options:
            st.info("No results match the current filter")
        else:
            # Add a "Select a file" option at the beginning
            file_options = [("", "Select a file...")] + file_options
            
            # Create a selectbox for file selection
            selected_file_id_name = st.selectbox(
                "Select a file to view details",
                options=file_options,
                format_func=lambda x: x[1],  # Display the file name
                key="file_selector"
            )
            
            # Get the selected file ID
            selected_file_id = selected_file_id_name[0] if selected_file_id_name[0] else None
            
            # Display file details if a file is selected
            if selected_file_id and selected_file_id in final_filtered_results:
                result_data = final_filtered_results[selected_file_id]
                
                # Display file info
                st.write("### File Information")
                st.write(f"**File:** {result_data.get('file_name', 'Unknown')}")
                st.write(f"**File ID:** {selected_file_id}")
                
                # Display extraction results
                st.write("### Extracted Metadata")
                
                # Extract and display metadata
                extracted_data = {}
                confidence_levels = result_data.get("confidence_levels", {})
                
                # Get the result data
                if "result_data" in result_data and result_data["result_data"]:
                    if isinstance(result_data["result_data"], dict):
                        # Extract key-value pairs from the result
                        for key, value in result_data["result_data"].items():
                            if not key.startswith("_") and key != "extracted_text":  # Skip internal fields
                                extracted_data[key] = value
                        
                        # Check for extracted_text field
                        if "extracted_text" in result_data["result_data"]:
                            st.write("#### Extracted Text")
                            st.write(result_data["result_data"]["extracted_text"])
                    elif isinstance(result_data["result_data"], str):
                        # If result_data is a string, display it as extracted text
                        st.write("#### Extracted Text")
                        st.write(result_data["result_data"])
                
                # Display extracted data as editable fields with confidence
                if extracted_data:
                    st.write("#### Key-Value Pairs")
                    for key, value in extracted_data.items():
                        confidence = confidence_levels.get(key, "N/A")
                        confidence_color = get_confidence_color(confidence)
                        
                        # FIX: Display confidence using st.markdown below the input field
                        # Use columns for better layout
                        col_input, col_confidence = st.columns([3, 1])
                        
                        with col_input:
                            # Create editable fields without HTML in label
                            if isinstance(value, list):
                                # For multiSelect fields
                                new_value = st.multiselect(
                                    key, # Use plain key as label
                                    options=value + ["Option 1", "Option 2", "Option 3"],
                                    default=value,
                                    key=f"edit_{selected_file_id}_{key}",
                                    help=f"Edit the value for {key}"
                                )
                            else:
                                # For other field types
                                new_value = st.text_input(
                                    key, # Use plain key as label
                                    value=str(value) if value is not None else "", # Handle None
                                    key=f"edit_{selected_file_id}_{key}",
                                    help=f"Edit the value for {key}"
                                )
                        
                        with col_confidence:
                            # Display confidence level using markdown with color
                            st.markdown(f"<span style='color:{confidence_color}; font-weight:bold;'>({confidence})</span>", unsafe_allow_html=True)
                        
                        # Update value if changed
                        if new_value != value:
                            # Find the original result in extraction_results
                            if selected_file_id in st.session_state.extraction_results:
                                # Get the original result
                                original_result = st.session_state.extraction_results[selected_file_id]
                                
                                # Update the result_data within the processed result
                                if "result_data" in result_data and isinstance(result_data["result_data"], dict):
                                    result_data["result_data"][key] = new_value
                                    # Optionally update confidence if user edits?
                                    # result_data["confidence_levels"][key] = "Edited"
                                
                                # Update the original result (more complex, depends on original structure)
                                # This part needs careful implementation based on how original results are stored
                                # For now, we just update the displayed data
                                logger.info(f"Value for {key} in file {selected_file_id} changed to {new_value}")
                                # Need to decide how to persist this change back to original_result
                else:
                    st.write("No structured data extracted")
                
                # Display raw result data for debugging
                if "original_data" in result_data:
                    st.write("### Raw Result Data (Debug View)")
                    with st.expander("Show Raw Data"):
                        st.json(result_data["original_data"])
    
    # Batch operations
    st.subheader("Batch Operations")
    
    # Get list of file IDs currently displayed
    displayed_file_ids = list(final_filtered_results.keys())
    
    # Select/Deselect All buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Select All", use_container_width=True, key="select_all_btn"):
            st.session_state.selected_result_ids = displayed_file_ids
            st.rerun()
    with col2:
        if st.button("Deselect All", use_container_width=True, key="deselect_all_btn"):
            st.session_state.selected_result_ids = []
            st.rerun()
    
    # Show count of selected files
    st.write(f"Selected {len(st.session_state.selected_result_ids)} of {len(displayed_file_ids)} results")
    
    # Apply Metadata button
    if st.button("Apply Metadata", use_container_width=True, key="apply_metadata_btn"):
        if not st.session_state.selected_result_ids:
            st.warning("Please select at least one file to apply metadata.")
        else:
            # Call the direct metadata application function
            from modules.direct_metadata_application_enhanced_fixed import apply_metadata_direct
            apply_metadata_direct()


