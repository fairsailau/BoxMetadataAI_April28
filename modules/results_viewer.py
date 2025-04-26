import streamlit as st
import pandas as pd
from typing import Dict, List, Any
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def view_results():
    """
    View and manage extraction results - COMPLETELY REDESIGNED TO AVOID NESTED EXPANDERS
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
    
    # Filter options
    st.subheader("Filter Results")
    st.session_state.results_filter = st.text_input(
        "Filter by file name",
        value=st.session_state.results_filter,
        key="filter_input"
    )
    
    # Get filtered results
    filtered_results = {}
    
    # Process extraction_results to prepare for display
    for file_id, result in st.session_state.extraction_results.items():
        # Create a standardized result structure
        processed_result = {
            "file_id": file_id,
            "file_name": "Unknown"
        }
        
        # Try to find file name
        for file in st.session_state.selected_files:
            if file["id"] == file_id:
                processed_result["file_name"] = file["name"]
                break
        
        # Process the result data based on its structure
        if isinstance(result, dict):
            # Store the original result
            processed_result["original_data"] = result
            
            # Check if this is a direct API response with an answer field
            if "answer" in result:
                answer = result["answer"]
                
                # Check if answer is a JSON string that needs parsing
                if isinstance(answer, str):
                    try:
                        parsed_answer = json.loads(answer)
                        if isinstance(parsed_answer, dict):
                            processed_result["result_data"] = parsed_answer
                        else:
                            processed_result["result_data"] = {"extracted_text": answer}
                    except json.JSONDecodeError:
                        # Not valid JSON, treat as text
                        processed_result["result_data"] = {"extracted_text": answer}
                elif isinstance(answer, dict):
                    # Already a dictionary
                    processed_result["result_data"] = answer
                else:
                    # Some other format, store as is
                    processed_result["result_data"] = {"extracted_text": str(answer)}
            
            # Check for items array with answer field (common in Box AI responses)
            elif "items" in result and isinstance(result["items"], list) and len(result["items"]) > 0:
                item = result["items"][0]
                if isinstance(item, dict) and "answer" in item:
                    answer = item["answer"]
                    
                    # Check if answer is a JSON string that needs parsing
                    if isinstance(answer, str):
                        try:
                            parsed_answer = json.loads(answer)
                            if isinstance(parsed_answer, dict):
                                processed_result["result_data"] = parsed_answer
                            else:
                                processed_result["result_data"] = {"extracted_text": answer}
                        except json.JSONDecodeError:
                            # Not valid JSON, treat as text
                            processed_result["result_data"] = {"extracted_text": answer}
                    elif isinstance(answer, dict):
                        # Already a dictionary
                        processed_result["result_data"] = answer
                    else:
                        # Some other format, store as is
                        processed_result["result_data"] = {"extracted_text": str(answer)}
            
            # If no structured data found, check for other fields that might contain data
            if "result_data" not in processed_result:
                # Look for any fields that might contain extracted data
                for key in ["extracted_data", "data", "result", "metadata"]:
                    if key in result and result[key]:
                        if isinstance(result[key], dict):
                            processed_result["result_data"] = result[key]
                            break
                        elif isinstance(result[key], str):
                            try:
                                parsed_data = json.loads(result[key])
                                if isinstance(parsed_data, dict):
                                    processed_result["result_data"] = parsed_data
                                    break
                            except json.JSONDecodeError:
                                processed_result["result_data"] = {"extracted_text": result[key]}
                                break
                
                # If still no result_data, use the entire result as is
                if "result_data" not in processed_result:
                    processed_result["result_data"] = result
        else:
            # Not a dictionary, store as text
            processed_result["result_data"] = {"extracted_text": str(result)}
        
        # Add to filtered results
        filtered_results[file_id] = processed_result
    
    # Apply filter if specified
    if st.session_state.results_filter:
        filtered_results = {
            file_id: result for file_id, result in filtered_results.items()
            if st.session_state.results_filter.lower() in result.get("file_name", "").lower()
        }
    
    # Display results count
    st.write(f"Showing {len(filtered_results)} of {len(st.session_state.extraction_results)} results")
    
    # Display results
    st.subheader("Extraction Results")
    
    # Determine if we're using structured or freeform extraction
    is_structured = st.session_state.metadata_config.get("extraction_method") == "structured"
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Table View", "Detailed View"])
    
    with tab1:
        # Table view
        table_data = []
        
        for file_id, result_data in filtered_results.items():
            # Basic file info
            row = {"File Name": result_data.get("file_name", "Unknown"), "File ID": file_id}
            
            # Extract and add metadata to the table
            extracted_text = ""
            
            # Get the result data
            if "result_data" in result_data and result_data["result_data"]:
                if isinstance(result_data["result_data"], dict):
                    # For structured data, add key fields to the table
                    for key, value in result_data["result_data"].items():
                        if not key.startswith("_") and key != "extracted_text":  # Skip internal fields
                            row[key] = str(value) if not isinstance(value, list) else ", ".join(str(v) for v in value)
                            # Limit to first 5 fields to keep table manageable
                            if len(row) > 7:  # File Name, File ID + 5 fields
                                break
                    
                    # Create a summary for the Extracted Text column
                    extracted_text = ", ".join([f"{k}: {v}" for k, v in list(result_data["result_data"].items())[:3]])
                elif isinstance(result_data["result_data"], str):
                    # If result_data is a string, use it directly
                    extracted_text = result_data["result_data"]
            
            # Add extracted text to row if not already added
            if "Extracted Text" not in row and extracted_text:
                row["Extracted Text"] = (extracted_text[:100] + "...") if len(extracted_text) > 100 else extracted_text
            elif "Extracted Text" not in row:
                row["Extracted Text"] = "No text extracted"
            
            table_data.append(row)
        
        if table_data:
            # Create dataframe
            df = pd.DataFrame(table_data)
            
            # Display dataframe
            st.dataframe(df, use_container_width=True)
            
            # Export options
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Export as CSV", use_container_width=True, key="export_csv_btn"):
                    # In a real app, we would save to a file
                    st.download_button(
                        label="Download CSV",
                        data=df.to_csv(index=False).encode('utf-8'),
                        file_name="extraction_results.csv",
                        mime="text/csv",
                        key="download_csv_btn"
                    )
            
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
                        for file_id, result_data in filtered_results.items()]
        
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
            if selected_file_id and selected_file_id in filtered_results:
                result_data = filtered_results[selected_file_id]
                
                # Display file info
                st.write("### File Information")
                st.write(f"**File:** {result_data.get('file_name', 'Unknown')}")
                st.write(f"**File ID:** {selected_file_id}")
                
                # Display extraction results
                st.write("### Extracted Metadata")
                
                # Extract and display metadata
                extracted_data = {}
                
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
                
                # Display extracted data as editable fields
                if extracted_data:
                    st.write("#### Key-Value Pairs")
                    for key, value in extracted_data.items():
                        # Create editable fields
                        if isinstance(value, list):
                            # For multiSelect fields
                            new_value = st.multiselect(
                                key,
                                options=value + ["Option 1", "Option 2", "Option 3"],
                                default=value,
                                key=f"edit_{selected_file_id}_{key}"
                            )
                        else:
                            # For other field types
                            new_value = st.text_input(key, value=str(value), key=f"edit_{selected_file_id}_{key}")
                        
                        # Update value if changed
                        if new_value != value:
                            # Find the original result in extraction_results
                            if selected_file_id in st.session_state.extraction_results:
                                # Get the original result
                                original_result = st.session_state.extraction_results[selected_file_id]
                                
                                # Check if original result has answer field
                                if isinstance(original_result, dict) and "answer" in original_result:
                                    # Check if answer is a JSON string
                                    if isinstance(original_result["answer"], str):
                                        try:
                                            # Parse the JSON string
                                            parsed_answer = json.loads(original_result["answer"])
                                            if isinstance(parsed_answer, dict):
                                                # Update the value
                                                parsed_answer[key] = new_value
                                                # Convert back to JSON string
                                                original_result["answer"] = json.dumps(parsed_answer)
                                        except json.JSONDecodeError:
                                            # Not valid JSON, can't update
                                            pass
                                    elif isinstance(original_result["answer"], dict):
                                        # Update the value directly
                                        original_result["answer"][key] = new_value
                                
                                # Also update the processed result for display
                                if "result_data" in result_data and isinstance(result_data["result_data"], dict):
                                    result_data["result_data"][key] = new_value
                else:
                    st.write("No structured data extracted")
                
                # CRITICAL FIX: Instead of using an expander for raw data, use a checkbox to toggle display
                st.write("### Raw Result Data (Debug View)")
                show_raw_data = st.checkbox("Show raw data", key=f"show_raw_{selected_file_id}")
                
                if show_raw_data:
                    if "original_data" in result_data:
                        st.json(result_data["original_data"])
                    else:
                        st.json(result_data)
                
                # Selection checkbox for batch operations
                st.write("### Batch Operations")
                selected = st.checkbox(
                    "Select for batch operations", 
                    value=selected_file_id in st.session_state.selected_result_ids,
                    key=f"select_{selected_file_id}"
                )
                
                if selected and selected_file_id not in st.session_state.selected_result_ids:
                    st.session_state.selected_result_ids.append(selected_file_id)
                elif not selected and selected_file_id in st.session_state.selected_result_ids:
                    st.session_state.selected_result_ids.remove(selected_file_id)
    
    # Batch operations
    st.subheader("Batch Operations")
    
    # Select/deselect all buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Select All", use_container_width=True):
            st.session_state.selected_result_ids = list(filtered_results.keys())
            st.rerun()
    
    with col2:
        if st.button("Deselect All", use_container_width=True):
            st.session_state.selected_result_ids = []
            st.rerun()
    
    # Display selected count
    st.write(f"Selected {len(st.session_state.selected_result_ids)} of {len(filtered_results)} results")
    
    # Apply metadata button
    if st.button("Apply Metadata", use_container_width=True):
        if st.session_state.selected_result_ids:
            st.session_state.current_page = "Apply Metadata"
            st.rerun()
        else:
            st.warning("Please select at least one result to apply metadata")
