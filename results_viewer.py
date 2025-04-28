import streamlit as st
import pandas as pd
from typing import Dict, List, Any
import json
import logging
import base64

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
                logger.info(f"Found 'answer' field in result: {answer}")
                
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
                                    logger.info(f"Extracted field {key} with value '{value['value']}' and confidence '{value['confidence']}'")
                                else:
                                    processed_result["result_data"][key] = value
                                    processed_result["confidence_levels"][key] = "Medium" # Default
                                    logger.info(f"Field {key} doesn't have expected structure, using value '{value}' with default Medium confidence")
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
                            logger.info(f"Extracted field {key} with value '{value['value']}' and confidence '{value['confidence']}'")
                        else:
                            processed_result["result_data"][key] = value
                            processed_result["confidence_levels"][key] = "Medium" # Default
                            logger.info(f"Field {key} doesn't have expected structure, using value '{value}' with default Medium confidence")
                else:
                    # Some other format, store as is
                    logger.warning(f"Answer is neither string nor dictionary: {type(answer)}. Using as is.")
                    processed_result["result_data"] = {"extracted_text": str(answer)}
            
            # Check for items array with answer field (common in Box AI responses)
            elif "items" in result and isinstance(result["items"], list) and len(result["items"]) > 0:
                item = result["items"][0]
                logger.info(f"Found 'items' array in result, processing first item: {item}")
                if isinstance(item, dict) and "answer" in item:
                    answer = item["answer"]
                    logger.info(f"Found 'answer' field in item: {answer}")
                    
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
                                        logger.info(f"Extracted field {key} with value '{value['value']}' and confidence '{value['confidence']}'")
                                    else:
                                        processed_result["result_data"][key] = value
                                        processed_result["confidence_levels"][key] = "Medium" # Default
                                        logger.info(f"Field {key} doesn't have expected structure, using value '{value}' with default Medium confidence")
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
                                logger.info(f"Extracted field {key} with value '{value['value']}' and confidence '{value['confidence']}'")
                            else:
                                processed_result["result_data"][key] = value
                                processed_result["confidence_levels"][key] = "Medium" # Default
                                logger.info(f"Field {key} doesn't have expected structure, using value '{value}' with default Medium confidence")
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
                            logger.info(f"Extracted field {base_key} with value '{result[base_key]}' and confidence '{value}'")
                    elif not key.startswith("_") and not any(key == field[:-len("_confidence")] for field in confidence_fields):
                        # Field without confidence, add it with default
                        processed_result["result_data"][key] = value
                        processed_result["confidence_levels"][key] = "Medium" # Default
                        logger.info(f"Field {key} has no confidence field, using value '{value}' with default Medium confidence")
            
            # If no structured data found, check for other fields that might contain data
            if not processed_result["result_data"]:
                logger.warning(f"No structured data found in result, looking for alternative fields")
                # Look for any fields that might contain extracted data
                for key in ["extracted_data", "data", "result", "metadata"]:
                    if key in result and result[key]:
                        logger.info(f"Found potential data in field '{key}': {result[key]}")
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
        
        # Apply file name filter if set
        if st.session_state.results_filter and st.session_state.results_filter.lower() not in processed_result["file_name"].lower():
            continue
        
        # Apply confidence filter if set
        if st.session_state.confidence_filter:
            # Check if any field has a confidence level in the filter
            has_matching_confidence = False
            for confidence in processed_result["confidence_levels"].values():
                if confidence in st.session_state.confidence_filter:
                    has_matching_confidence = True
                    break
            
            if not has_matching_confidence:
                continue
        
        # Add to filtered results
        filtered_results[file_id] = processed_result
    
    # Show number of results
    st.write(f"Showing {len(filtered_results)} of {len(st.session_state.extraction_results)} results")
    
    # Display results
    st.subheader("Extraction Results")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Table View", "Detailed View"])
    
    with tab1:
        # Create a DataFrame for table view
        if filtered_results:
            # Get all unique fields across all results
            all_fields = set()
            for result in filtered_results.values():
                all_fields.update(result["result_data"].keys())
            
            # Create rows for the DataFrame
            rows = []
            for file_id, result in filtered_results.items():
                row = {"File Name": result["file_name"], "File ID": file_id}
                
                # Add fields and their confidence levels
                for field in all_fields:
                    if field in result["result_data"]:
                        row[field] = result["result_data"][field]
                        confidence = result["confidence_levels"].get(field, "Medium")
                        row[f"{field} Confidence"] = confidence
                    else:
                        row[field] = ""
                        row[f"{field} Confidence"] = ""
                
                rows.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(rows)
            
            # Reorder columns to group fields with their confidence
            columns = ["File Name", "File ID"]
            for field in all_fields:
                columns.extend([field, f"{field} Confidence"])
            
            # Ensure all columns exist in the DataFrame
            for col in columns:
                if col not in df.columns:
                    df[col] = ""
            
            # Reorder columns
            df = df[columns]
            
            # Apply styling to confidence columns
            styled_df = df.copy()
            
            # Display the table with a download button
            st.dataframe(
                styled_df,
                column_config={
                    **{field: st.column_config.TextColumn(field) for field in all_fields},
                    **{
                        f"{field} Confidence": st.column_config.TextColumn(
                            f"{field} Confidence",
                            help=f"Confidence level for {field}",
                            width="small"
                        )
                        for field in all_fields
                    }
                },
                hide_index=True
            )
            
            # Add export buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Export as CSV"):
                    # Create a download link for CSV
                    csv = df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="extraction_results.csv">Download CSV File</a>'
                    st.markdown(href, unsafe_allow_html=True)
            
            with col2:
                if st.button("Export as Excel"):
                    # Create a download link for Excel
                    # Note: This requires additional libraries like openpyxl
                    try:
                        excel_file = df.to_excel(index=False)
                        b64 = base64.b64encode(excel_file).decode()
                        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="extraction_results.xlsx">Download Excel File</a>'
                        st.markdown(href, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error exporting to Excel: {str(e)}")
                        st.info("Try exporting as CSV instead.")
        else:
            st.info("No results match the current filters.")
    
    with tab2:
        # Detailed view of a single result
        if filtered_results:
            # Create a dropdown to select a file
            file_options = {result["file_name"]: file_id for file_id, result in filtered_results.items()}
            selected_file_name = st.selectbox("Select a file to view details", options=list(file_options.keys()))
            selected_file_id = file_options[selected_file_name]
            
            # Get the selected result
            selected_result = filtered_results[selected_file_id]
            
            # Display file information
            st.subheader("File Information")
            st.write(f"File: {selected_result['file_name']}")
            st.write(f"File ID: {selected_file_id}")
            
            # Display extracted metadata with confidence indicators
            st.subheader("Extracted Metadata")
            
            # Display key-value pairs with confidence indicators
            st.write("Key-Value Pairs")
            
            for key, value in selected_result["result_data"].items():
                confidence = selected_result["confidence_levels"].get(key, "Medium")
                confidence_color = get_confidence_color(confidence)
                
                # Create a label with confidence indicator
                # FIX: Use markdown with HTML that will render properly
                st.markdown(f"{key} <span style='color:{confidence_color}; font-weight:bold;'>({confidence})</span>", unsafe_allow_html=True)
                
                # Display the value in a text area
                st.text_area(
                    label=f"Value for {key}",
                    value=str(value),
                    key=f"value_{key}_{selected_file_id}",
                    label_visibility="collapsed"
                )
            
            # Display raw result data for debugging
            with st.expander("Raw Result Data (Debug View)"):
                st.json(selected_result["original_data"] if "original_data" in selected_result else selected_result)
        else:
            st.info("No results match the current filters.")
    
    # Batch operations
    st.subheader("Batch Operations")
    
    # Select all / Deselect all buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Select All"):
            st.session_state.selected_result_ids = list(filtered_results.keys())
    with col2:
        if st.button("Deselect All"):
            st.session_state.selected_result_ids = []
    
    # Show selected count
    st.write(f"Selected {len(st.session_state.selected_result_ids)} of {len(filtered_results)} results")
    
    # Apply metadata button
    if st.button("Apply Metadata"):
        if not st.session_state.selected_result_ids:
            st.warning("No files selected for metadata application.")
        else:
            # Import here to avoid circular imports
            from modules.direct_metadata_application_enhanced_fixed import apply_metadata_direct
            apply_metadata_direct()
