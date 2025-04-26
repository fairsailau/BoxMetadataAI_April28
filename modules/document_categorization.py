import streamlit as st
import logging
import json
import requests
import re
import os
import datetime
import pandas as pd
import altair as alt
from typing import Dict, Any, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def document_categorization():
    """
    Enhanced document categorization with improved confidence metrics
    """
    st.title("Document Categorization")
    
    if not st.session_state.authenticated or not st.session_state.client:
        st.error("Please authenticate with Box first")
        return
    
    if not st.session_state.selected_files:
        st.warning("No files selected. Please select files in the File Browser first.")
        if st.button("Go to File Browser", key="go_to_file_browser_button_cat"):
            st.session_state.current_page = "File Browser"
            st.rerun()
        return
    
    # Initialize document categorization state if not exists
    if "document_categorization" not in st.session_state:
        st.session_state.document_categorization = {
            "is_categorized": False,
            "results": {},
            "errors": {}
        }
    
    # Initialize confidence thresholds if not exists
    if "confidence_thresholds" not in st.session_state:
        st.session_state.confidence_thresholds = {
            "auto_accept": 0.85,
            "verification": 0.6,
            "rejection": 0.4
        }
    
    # Display selected files
    num_files = len(st.session_state.selected_files)
    st.write(f"Ready to categorize {num_files} files using Box AI.")
    
    # Create tabs for main interface and settings
    tab1, tab2 = st.tabs(["Categorization", "Confidence Settings"])
    
    with tab1:
        # AI Model selection
        ai_models = [
            "azure__openai__gpt_4o_mini",
            "azure__openai__gpt_4o_2024_05_13",
            "google__gemini_2_0_flash_001",
            "google__gemini_2_0_flash_lite_preview",
            "google__gemini_1_5_flash_001",
            "google__gemini_1_5_pro_001",
            "aws__claude_3_haiku",
            "aws__claude_3_sonnet",
            "aws__claude_3_5_sonnet",
            "aws__claude_3_7_sonnet",
            "aws__titan_text_lite"
        ]
        
        selected_model = st.selectbox(
            "Select AI Model for Categorization",
            options=ai_models,
            index=0,
            key="ai_model_select_cat",
            help="Choose the AI model to use for document categorization"
        )
        
        # Enhanced categorization options
        st.write("### Categorization Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Two-stage categorization option
            use_two_stage = st.checkbox(
                "Use two-stage categorization",
                value=True,
                help="When enabled, documents with low confidence will undergo a second analysis"
            )
            
            # Multi-model consensus option
            use_consensus = st.checkbox(
                "Use multi-model consensus",
                value=False,
                help="When enabled, multiple AI models will be used and their results combined for more accurate categorization"
            )
        
        with col2:
            # Confidence threshold for second-stage
            confidence_threshold = st.slider(
                "Confidence threshold for second-stage",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                step=0.05,
                help="Documents with confidence below this threshold will undergo second-stage analysis",
                disabled=not use_two_stage
            )
            
            # Select models for consensus
            consensus_models = []
            if use_consensus:
                consensus_models = st.multiselect(
                    "Select models for consensus",
                    options=ai_models,
                    default=[ai_models[0], ai_models[2]] if len(ai_models) > 2 else ai_models[:1],
                    help="Select 2-3 models for best results (more models will increase processing time)"
                )
                
                if len(consensus_models) < 1:
                    st.warning("Please select at least one model for consensus categorization")
        
        # Categorization controls
        col1, col2 = st.columns(2)
        
        with col1:
            start_button = st.button("Start Categorization", key="start_categorization_button_cat", use_container_width=True)
        
        with col2:
            cancel_button = st.button("Cancel Categorization", key="cancel_categorization_button_cat", use_container_width=True)
        
        # Process categorization
        if start_button:
            with st.spinner("Categorizing documents..."):
                # Reset categorization results
                st.session_state.document_categorization = {
                    "is_categorized": False,
                    "results": {},
                    "errors": {}
                }
                
                # Process each file
                for file in st.session_state.selected_files:
                    file_id = file["id"]
                    file_name = file["name"]
                    
                    try:
                        if use_consensus and consensus_models:
                            # Multi-model consensus categorization
                            consensus_results = []
                            
                            # Create progress bar for models
                            model_progress = st.progress(0)
                            model_status = st.empty()
                            
                            # Process with each model
                            for i, model in enumerate(consensus_models):
                                model_status.text(f"Processing with {model}...")
                                result = categorize_document(file_id, model)
                                consensus_results.append(result)
                                model_progress.progress((i + 1) / len(consensus_models))
                            
                            # Clear progress indicators
                            model_progress.empty()
                            model_status.empty()
                            
                            # Combine results using weighted voting
                            result = combine_categorization_results(consensus_results)
                            
                            # Add model details to reasoning
                            models_text = ", ".join(consensus_models)
                            result["reasoning"] = f"Consensus from models: {models_text}\n\n" + result["reasoning"]
                        else:
                            # First-stage categorization
                            result = categorize_document(file_id, selected_model)
                            
                            # Check if second-stage is needed
                            if use_two_stage and result["confidence"] < confidence_threshold:
                                st.info(f"Low confidence ({result['confidence']:.2f}) for {file_name}, performing detailed analysis...")
                                
                                # Second-stage categorization with more detailed prompt
                                detailed_result = categorize_document_detailed(file_id, selected_model, result["document_type"])
                                
                                # Merge results, preferring the detailed analysis
                                result = {
                                    "document_type": detailed_result["document_type"],
                                    "confidence": detailed_result["confidence"],
                                    "reasoning": detailed_result["reasoning"],
                                    "first_stage_type": result["document_type"],
                                    "first_stage_confidence": result["confidence"]
                                }
                        
                        # Extract document features for multi-factor confidence
                        document_features = extract_document_features(file_id)
                        
                        # Calculate multi-factor confidence
                        document_types = [
                            "Sales Contract",
                            "Invoices",
                            "Tax",
                            "Financial Report",
                            "Employment Contract",
                            "PII",
                            "Other"
                        ]
                        
                        multi_factor_confidence = calculate_multi_factor_confidence(
                            result["confidence"],
                            document_features,
                            result["document_type"],
                            result.get("reasoning", ""),
                            document_types
                        )
                        
                        # Apply confidence calibration if available
                        calibrated_confidence = apply_confidence_calibration(
                            result["document_type"],
                            multi_factor_confidence["overall"]
                        )
                        
                        # Store result with enhanced confidence data
                        st.session_state.document_categorization["results"][file_id] = {
                            "file_id": file_id,
                            "file_name": file_name,
                            "document_type": result["document_type"],
                            "confidence": result["confidence"],  # Original AI confidence
                            "multi_factor_confidence": multi_factor_confidence,  # Detailed confidence factors
                            "calibrated_confidence": calibrated_confidence,  # Calibrated overall confidence
                            "reasoning": result["reasoning"],
                            "first_stage_type": result.get("first_stage_type"),
                            "first_stage_confidence": result.get("first_stage_confidence"),
                            "document_features": document_features
                        }
                    except Exception as e:
                        logger.error(f"Error categorizing document {file_name}: {str(e)}")
                        st.session_state.document_categorization["errors"][file_id] = {
                            "file_id": file_id,
                            "file_name": file_name,
                            "error": str(e)
                        }
                
                # Apply confidence thresholds
                st.session_state.document_categorization["results"] = apply_confidence_thresholds(
                    st.session_state.document_categorization["results"]
                )
                
                # Mark as categorized
                st.session_state.document_categorization["is_categorized"] = True
                
                # Show success message
                num_processed = len(st.session_state.document_categorization["results"])
                num_errors = len(st.session_state.document_categorization["errors"])
                
                if num_errors == 0:
                    st.success(f"Categorization complete! Processed {num_processed} files.")
                else:
                    st.warning(f"Categorization complete! Processed {num_processed} files with {num_errors} errors.")
        
        # Display categorization results
        if st.session_state.document_categorization["is_categorized"]:
            display_categorization_results()
    
    with tab2:
        # Confidence settings
        st.write("### Confidence Configuration")
        
        # Confidence threshold configuration
        configure_confidence_thresholds()
        
        # Confidence validation
        with st.expander("Confidence Validation", expanded=False):
            validate_confidence_with_examples()

def display_categorization_results():
    """
    Display categorization results with enhanced confidence visualization
    """
    st.write("### Categorization Results")
    
    # Get results from session state
    results = st.session_state.document_categorization["results"]
    
    if not results:
        st.info("No categorization results available.")
        return
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["Table View", "Detailed View"])
    
    with tab1:
        # Create a table of results with enhanced confidence display
        results_data = []
        for file_id, result in results.items():
            # Determine status based on thresholds
            status = result.get("status", "Review")
            
            # Determine confidence level and color
            confidence = result.get("calibrated_confidence", result.get("confidence", 0.0))
            if confidence >= 0.8:
                confidence_level = "High"
                confidence_color = "green"
            elif confidence >= 0.6:
                confidence_level = "Medium"
                confidence_color = "orange"
            else:
                confidence_level = "Low"
                confidence_color = "red"
            
            results_data.append({
                "File Name": result["file_name"],
                "Document Type": result["document_type"],
                "Confidence": f"<span style='color: {confidence_color};'>{confidence_level} ({confidence:.2f})</span>",
                "Status": status
            })
        
        if results_data:
            # Convert to DataFrame for display
            df = pd.DataFrame(results_data)
            
            # Display as HTML to preserve formatting
            st.markdown(
                df.to_html(escape=False, index=False),
                unsafe_allow_html=True
            )
    
    with tab2:
        # Create detailed view with confidence visualization
        for file_id, result in results.items():
            with st.container():
                st.write(f"### {result['file_name']}")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Display document type and confidence
                    st.write(f"**Category:** {result['document_type']}")
                    
                    # Display confidence visualization
                    if "multi_factor_confidence" in result:
                        display_confidence_visualization(result["multi_factor_confidence"])
                    else:
                        # Fallback for results without multi-factor confidence
                        confidence = result.get("confidence", 0.0)
                        if confidence >= 0.8:
                            confidence_color = "#28a745"  # Green
                        elif confidence >= 0.6:
                            confidence_color = "#ffc107"  # Yellow
                        else:
                            confidence_color = "#dc3545"  # Red
                        
                        st.markdown(
                            f"""
                            <div style="margin-bottom: 10px;">
                                <div style="display: flex; align-items: center; margin-bottom: 5px;">
                                    <div style="font-weight: bold; margin-right: 10px;">Confidence:</div>
                                    <div style="font-weight: bold; color: {confidence_color};">{confidence:.2f}</div>
                                </div>
                                <div style="width: 100%; background-color: #f0f0f0; height: 10px; border-radius: 5px; overflow: hidden;">
                                    <div style="width: {confidence*100}%; background-color: {confidence_color}; height: 100%;"></div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # Display confidence explanation
                    if "multi_factor_confidence" in result:
                        explanations = get_confidence_explanation(
                            result["multi_factor_confidence"],
                            result["document_type"]
                        )
                        st.info(explanations["overall"])
                    
                    # Display reasoning
                    with st.expander("Reasoning", expanded=False):
                        st.write(result.get("reasoning", "No reasoning provided"))
                    
                    # Display first-stage results if available
                    if result.get("first_stage_type"):
                        with st.expander("First-Stage Results", expanded=False):
                            st.write(f"**First-stage category:** {result['first_stage_type']}")
                            st.write(f"**First-stage confidence:** {result['first_stage_confidence']:.2f}")
                
                with col2:
                    # Category override
                    document_types = [
                        "Sales Contract",
                        "Invoices",
                        "Tax",
                        "Financial Report",
                        "Employment Contract",
                        "PII",
                        "Other"
                    ]
                    
                    st.write("**Override Category:**")
                    new_category = st.selectbox(
                        "Select category",
                        options=document_types,
                        index=document_types.index(result["document_type"]) if result["document_type"] in document_types else 0,
                        key=f"override_{file_id}"
                    )
                    
                    if st.button("Apply Override", key=f"apply_override_{file_id}"):
                        # Save feedback for calibration
                        save_categorization_feedback(file_id, result["document_type"], new_category)
                        
                        # Update the result
                        st.session_state.document_categorization["results"][file_id]["document_type"] = new_category
                        st.session_state.document_categorization["results"][file_id]["confidence"] = 1.0
                        st.session_state.document_categorization["results"][file_id]["calibrated_confidence"] = 1.0
                        st.session_state.document_categorization["results"][file_id]["reasoning"] += "\n\nManually overridden by user."
                        st.session_state.document_categorization["results"][file_id]["status"] = "Accepted"
                        
                        st.success(f"Category updated to {new_category}")
                        st.rerun()
                    
                    # Document preview
                    st.write("**Document Preview:**")
                    
                    # Get document preview using Box API
                    preview_url = get_document_preview_url(file_id)
                    
                    if preview_url:
                        st.image(preview_url, caption="Document Preview", use_column_width=True)
                    else:
                        # Fallback to document properties
                        client = st.session_state.client
                        try:
                            file_info = client.file(file_id).get()
                            
                            st.write(f"**Size:** {file_info.size / 1024:.1f} KB")
                            st.write(f"**Created:** {file_info.created_at}")
                            st.write(f"**Modified:** {file_info.modified_at}")
                            st.write(f"**Type:** {file_info.type}")
                        except Exception as e:
                            st.write("Could not retrieve file information")
                
                # User feedback section
                with st.expander("Provide Feedback", expanded=False):
                    collect_user_feedback(file_id, result)
                
                st.markdown("---")
        
        # Continue button
        st.write("---")
        if st.button("Continue to Metadata Configuration", key="continue_to_metadata_button_cat", use_container_width=True):
            st.session_state.current_page = "Metadata Configuration"
            st.rerun()

def categorize_document(file_id: str, model: str = "azure__openai__gpt_4o_mini") -> Dict[str, Any]:
    """
    Categorize a document using Box AI
    
    Args:
        file_id: Box file ID
        model: AI model to use for categorization
        
    Returns:
        dict: Document categorization result
    """
    # Get access token from client
    access_token = None
    if hasattr(st.session_state.client, '_oauth'):
        access_token = st.session_state.client._oauth.access_token
    elif hasattr(st.session_state.client, 'auth') and hasattr(st.session_state.client.auth, 'access_token'):
        access_token = st.session_state.client.auth.access_token
    
    if not access_token:
        raise ValueError("Could not retrieve access token from client")
    
    # Set headers
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    # Define document types to categorize
    document_types = [
        "Sales Contract",
        "Invoices",
        "Tax",
        "Financial Report",
        "Employment Contract",
        "PII",
        "Other"
    ]
    
    # Create prompt for document categorization with confidence score request
    prompt = (
        f"Analyze this document and determine which category it belongs to from the following options: "
        f"{', '.join(document_types)}. "
        f"Provide your answer in the following format:\n"
        f"Category: [selected category]\n"
        f"Confidence: [confidence score between 0 and 1, where 1 is highest confidence]\n"
        f"Reasoning: [detailed explanation of your categorization, including key features of the document that support this categorization]"
    )
    
    # Construct API URL for Box AI Ask
    api_url = "https://api.box.com/2.0/ai/ask"
    
    # Construct request body according to the API documentation
    request_body = {
        "mode": "single_item_qa",  # Required parameter - single_item_qa or multiple_item_qa
        "prompt": prompt,
        "items": [
            {
                "type": "file",
                "id": file_id
            }
        ],
        "ai_agent": {
            "type": "ai_agent_ask",
            "basic_text": {
                "model": model,
                "mode": "default"  # Required parameter for basic_text
            }
        }
    }
    
    try:
        # Make API call
        logger.info(f"Making Box AI API call with request: {json.dumps(request_body)}")
        response = requests.post(api_url, headers=headers, json=request_body)
        
        # Log response for debugging
        logger.info(f"Box AI API response status: {response.status_code}")
        if response.status_code != 200:
            logger.error(f"Box AI API error response: {response.text}")
            raise Exception(f"Error in Box AI API call: {response.status_code} Client Error: Bad Request for url: {api_url}")
        
        # Parse response
        response_data = response.json()
        logger.info(f"Box AI API response data: {json.dumps(response_data)}")
        
        # Extract answer from response
        if "answer" in response_data:
            answer_text = response_data["answer"]
            
            # Parse the structured response to extract category, confidence, and reasoning
            document_type, confidence, reasoning = parse_categorization_response(answer_text, document_types)
            
            return {
                "document_type": document_type,
                "confidence": confidence,
                "reasoning": reasoning
            }
        
        # If no answer in response, return default
        return {
            "document_type": "Other",
            "confidence": 0.0,
            "reasoning": "Could not determine document type"
        }
    
    except Exception as e:
        logger.error(f"Error in Box AI API call: {str(e)}")
        raise Exception(f"Error categorizing document: {str(e)}")

def categorize_document_detailed(file_id: str, model: str, initial_category: str) -> Dict[str, Any]:
    """
    Perform a more detailed categorization for documents with low confidence
    
    Args:
        file_id: Box file ID
        model: AI model to use for categorization
        initial_category: Initial category from first-stage categorization
        
    Returns:
        dict: Document categorization result
    """
    # Get access token from client
    access_token = None
    if hasattr(st.session_state.client, '_oauth'):
        access_token = st.session_state.client._oauth.access_token
    elif hasattr(st.session_state.client, 'auth') and hasattr(st.session_state.client.auth, 'access_token'):
        access_token = st.session_state.client.auth.access_token
    
    if not access_token:
        raise ValueError("Could not retrieve access token from client")
    
    # Set headers
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    
    # Define document types to categorize
    document_types = [
        "Sales Contract",
        "Invoices",
        "Tax",
        "Financial Report",
        "Employment Contract",
        "PII",
        "Other"
    ]
    
    # Create a more detailed prompt for second-stage analysis
    prompt = (
        f"Analyze this document in detail to determine its category. "
        f"The initial categorization suggested it might be '{initial_category}', but we need a more thorough analysis.\n\n"
        f"For each of the following categories, provide a score from 0-10 indicating how well the document matches that category, "
        f"along with specific evidence from the document:\n\n"
        f"{', '.join(document_types)}\n\n"
        f"Then provide your final categorization in the following format:\n"
        f"Category: [selected category]\n"
        f"Confidence: [confidence score between 0 and 1, where 1 is highest confidence]\n"
        f"Reasoning: [detailed explanation with specific evidence from the document]"
    )
    
    # Construct API URL for Box AI Ask
    api_url = "https://api.box.com/2.0/ai/ask"
    
    # Construct request body according to the API documentation
    request_body = {
        "mode": "single_item_qa",
        "prompt": prompt,
        "items": [
            {
                "type": "file",
                "id": file_id
            }
        ],
        "ai_agent": {
            "type": "ai_agent_ask",
            "basic_text": {
                "model": model,
                "mode": "default"
            }
        }
    }
    
    try:
        # Make API call
        logger.info(f"Making detailed Box AI API call with request: {json.dumps(request_body)}")
        response = requests.post(api_url, headers=headers, json=request_body)
        
        # Check response
        if response.status_code != 200:
            logger.error(f"Box AI API error response: {response.text}")
            raise Exception(f"Error in Box AI API call: {response.status_code} Client Error: Bad Request for url: {api_url}")
        
        # Parse response
        response_data = response.json()
        
        # Extract answer from response
        if "answer" in response_data:
            answer_text = response_data["answer"]
            
            # Parse the structured response to extract category, confidence, and reasoning
            document_type, confidence, reasoning = parse_categorization_response(answer_text, document_types)
            
            # Boost confidence slightly for detailed analysis
            # This reflects the more thorough analysis performed
            confidence = min(confidence * 1.1, 1.0)
            
            return {
                "document_type": document_type,
                "confidence": confidence,
                "reasoning": reasoning
            }
        
        # If no answer in response, return default
        return {
            "document_type": initial_category,
            "confidence": 0.3,
            "reasoning": "Could not determine document type in detailed analysis"
        }
    
    except Exception as e:
        logger.error(f"Error in detailed Box AI API call: {str(e)}")
        raise Exception(f"Error in detailed categorization: {str(e)}")

def parse_categorization_response(response_text: str, document_types: List[str]) -> Tuple[str, float, str]:
    """
    Parse the AI response to extract document type, confidence score, and reasoning
    
    Args:
        response_text: The AI response text
        document_types: List of valid document types
        
    Returns:
        tuple: (document_type, confidence, reasoning)
    """
    # Default values
    document_type = "Other"
    confidence = 0.5
    reasoning = response_text
    
    try:
        # Try to extract category using regex
        category_match = re.search(r"Category:\s*([^\n]+)", response_text, re.IGNORECASE)
        if category_match:
            category_text = category_match.group(1).strip()
            # Find the closest matching document type
            for dt in document_types:
                if dt.lower() in category_text.lower():
                    document_type = dt
                    break
        
        # Try to extract confidence using regex
        confidence_match = re.search(r"Confidence:\s*(0\.\d+|1\.0|1)", response_text, re.IGNORECASE)
        if confidence_match:
            confidence = float(confidence_match.group(1))
        else:
            # If no explicit confidence, try to find confidence-related words
            confidence_words = {
                "very high": 0.9,
                "high": 0.8,
                "good": 0.7,
                "moderate": 0.6,
                "medium": 0.5,
                "low": 0.4,
                "very low": 0.3,
                "uncertain": 0.2
            }
            
            for word, value in confidence_words.items():
                if word in response_text.lower():
                    confidence = value
                    break
        
        # Try to extract reasoning
        reasoning_match = re.search(r"Reasoning:\s*([^\n]+(?:\n[^\n]+)*)", response_text, re.IGNORECASE)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        
        # If no document type was found in the structured response, try to find it in the full text
        if document_type == "Other":
            for dt in document_types:
                if dt.lower() in response_text.lower():
                    document_type = dt
                    break
        
        return document_type, confidence, reasoning
    
    except Exception as e:
        logger.error(f"Error parsing categorization response: {str(e)}")
        return document_type, confidence, reasoning

def extract_document_features(file_id: str) -> Dict[str, Any]:
    """
    Extract features from a document to aid in categorization
    
    Args:
        file_id: Box file ID
        
    Returns:
        dict: Document features
    """
    try:
        client = st.session_state.client
        file_info = client.file(file_id).get()
        
        features = {
            "extension": file_info.name.split(".")[-1].lower() if "." in file_info.name else "",
            "size_kb": file_info.size / 1024,
            "file_type": file_info.type
        }
        
        # Get text content preview if possible
        try:
            # Use Box API to get text representation
            text_content = ""
            
            # This is a simplified approach - in a real implementation,
            # you would use Box's content API to get text content
            # For now, we'll extract text from the file name and type
            text_content = f"{file_info.name} {file_info.type}"
            
            features["text_content"] = text_content
        except Exception as e:
            logger.warning(f"Could not extract text content: {str(e)}")
            features["text_content"] = ""
        
        return features
    except Exception as e:
        logger.error(f"Error extracting document features: {str(e)}")
        return {}

def calculate_multi_factor_confidence(
    ai_confidence: float,
    document_features: dict,
    category: str,
    response_text: str,
    document_types: List[str]
) -> dict:
    """
    Calculate a multi-factor confidence score based on various aspects
    
    Args:
        ai_confidence: The confidence score reported by the AI
        document_features: Features extracted from the document
        category: The assigned category
        response_text: The full AI response text
        document_types: List of valid document types
        
    Returns:
        dict: Multi-factor confidence scores and overall confidence
    """
    # Initialize confidence factors
    confidence_factors = {
        "ai_reported": ai_confidence,
        "response_quality": 0.0,
        "category_specificity": 0.0,
        "reasoning_quality": 0.0,
        "document_features": 0.0
    }
    
    # 1. Response Quality - How well-structured was the AI response?
    expected_sections = ["Category:", "Confidence:", "Reasoning:"]
    sections_found = sum(1 for section in expected_sections if section in response_text)
    confidence_factors["response_quality"] = sections_found / len(expected_sections)
    
    # 2. Category Specificity - How specific is the category assignment?
    if category == "Other":
        confidence_factors["category_specificity"] = 0.3  # Low confidence for "Other" category
    else:
        # Check how many times the category appears in the reasoning
        category_mentions = len(re.findall(r'\b' + re.escape(category) + r'\b', response_text, re.IGNORECASE))
        confidence_factors["category_specificity"] = min(0.5 + (category_mentions * 0.1), 1.0)
    
    # 3. Reasoning Quality - How detailed and specific is the reasoning?
    reasoning_match = re.search(r"Reasoning:\s*([^\n]+(?:\n[^\n]+)*)", response_text, re.IGNORECASE)
    if reasoning_match:
        reasoning_text = reasoning_match.group(1).strip()
        word_count = len(reasoning_text.split())
        
        # More detailed reasoning increases confidence
        if word_count < 10:
            confidence_factors["reasoning_quality"] = 0.3
        elif word_count < 30:
            confidence_factors["reasoning_quality"] = 0.6
        else:
            confidence_factors["reasoning_quality"] = 0.9
            
        # Check for specific keywords that indicate uncertainty
        uncertainty_words = ["maybe", "perhaps", "possibly", "might", "could be", "uncertain", "not clear"]
        uncertainty_count = sum(1 for word in uncertainty_words if word in reasoning_text.lower())
        
        # Reduce confidence based on uncertainty words
        confidence_factors["reasoning_quality"] *= max(0.5, 1.0 - (uncertainty_count * 0.1))
    
    # 4. Document Features - Do document features align with the category?
    if document_features:
        # Define feature patterns for different document types
        category_feature_patterns = {
            "Sales Contract": {
                "keywords": ["agreement", "contract", "sale", "purchase", "terms", "conditions", "party"],
                "extension_preference": ["pdf", "docx"]
            },
            "Invoices": {
                "keywords": ["invoice", "bill", "payment", "amount", "total", "due", "tax"],
                "extension_preference": ["pdf", "xlsx"]
            },
            "Tax": {
                "keywords": ["tax", "return", "irs", "income", "deduction", "filing"],
                "extension_preference": ["pdf"]
            },
            "Financial Report": {
                "keywords": ["financial", "report", "statement", "balance", "income", "cash flow", "quarter", "annual"],
                "extension_preference": ["pdf", "xlsx"]
            },
            "Employment Contract": {
                "keywords": ["employment", "employee", "employer", "salary", "compensation", "termination", "confidentiality"],
                "extension_preference": ["pdf", "docx"]
            },
            "PII": {
                "keywords": ["personal", "information", "ssn", "social security", "address", "phone", "email", "confidential"],
                "extension_preference": ["pdf", "docx", "xlsx"]
            }
        }
        
        # Calculate feature match score
        feature_match_score = 0.5  # Default middle score
        
        if category in category_feature_patterns:
            pattern = category_feature_patterns[category]
            matches = 0
            total_checks = 0
            
            # Check keywords in text content
            if "text_content" in document_features and "keywords" in pattern:
                total_checks += 1
                keyword_matches = sum(1 for keyword in pattern["keywords"] if keyword in document_features["text_content"].lower())
                if keyword_matches >= 2:
                    matches += 1
            
            # Check file extension
            if "extension" in document_features and "extension_preference" in pattern:
                total_checks += 1
                if document_features["extension"] in pattern["extension_preference"]:
                    matches += 1
            
            # Calculate feature match score if we have checks
            if total_checks > 0:
                feature_match_score = matches / total_checks
        
        confidence_factors["document_features"] = feature_match_score
    
    # Calculate weighted overall confidence
    weights = {
        "ai_reported": 0.4,
        "response_quality": 0.1,
        "category_specificity": 0.2,
        "reasoning_quality": 0.2,
        "document_features": 0.1
    }
    
    overall_confidence = sum(
        confidence_factors[factor] * weights[factor]
        for factor in confidence_factors
    )
    
    # Add overall confidence to the result
    confidence_factors["overall"] = overall_confidence
    
    return confidence_factors

def display_confidence_visualization(confidence_data: dict, container=None):
    """
    Display a comprehensive confidence visualization
    
    Args:
        confidence_data: Dictionary containing confidence factors and overall confidence
        container: Optional Streamlit container to render in
    """
    # Use provided container or create a new one
    if container is None:
        container = st
    
    overall_confidence = confidence_data.get("overall", 0.0)
    
    # Determine confidence level and color
    if overall_confidence >= 0.8:
        confidence_level = "High"
        confidence_color = "#28a745"  # Green
    elif overall_confidence >= 0.6:
        confidence_level = "Medium"
        confidence_color = "#ffc107"  # Yellow
    else:
        confidence_level = "Low"
        confidence_color = "#dc3545"  # Red
    
    # Create confidence meter
    container.markdown(
        f"""
        <div style="margin-bottom: 10px;">
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="font-weight: bold; margin-right: 10px;">Confidence:</div>
                <div style="font-weight: bold; color: {confidence_color};">{confidence_level} ({overall_confidence:.2f})</div>
            </div>
            <div style="width: 100%; background-color: #f0f0f0; height: 10px; border-radius: 5px; overflow: hidden;">
                <div style="width: {overall_confidence*100}%; background-color: {confidence_color}; height: 100%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Display confidence factors if they exist
    factors_to_display = {
        "ai_reported": "AI Model",
        "response_quality": "Response Quality",
        "category_specificity": "Category Specificity",
        "reasoning_quality": "Reasoning Quality",
        "document_features": "Document Features"
    }
    
    # Check if we have detailed factors
    has_factors = any(factor in confidence_data for factor in factors_to_display)
    
    if has_factors:
        with container.expander("Confidence Breakdown", expanded=False):
            for factor_key, factor_name in factors_to_display.items():
                if factor_key in confidence_data:
                    factor_value = confidence_data[factor_key]
                    
                    # Determine factor color
                    if factor_value >= 0.8:
                        factor_color = "#28a745"  # Green
                    elif factor_value >= 0.6:
                        factor_color = "#ffc107"  # Yellow
                    else:
                        factor_color = "#dc3545"  # Red
                    
                    # Display factor meter
                    container.markdown(
                        f"""
                        <div style="display: flex; align-items: center; margin-bottom: 5px;">
                            <div style="width: 150px;">{factor_name}:</div>
                            <div style="flex-grow: 1; background-color: #f0f0f0; height: 8px; border-radius: 4px; overflow: hidden; margin: 0 10px;">
                                <div style="width: {factor_value*100}%; background-color: {factor_color}; height: 100%;"></div>
                            </div>
                            <div style="width: 50px; text-align: right; color: {factor_color};">{factor_value:.2f}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            # Add explanation of factors
            container.markdown("""
            **Confidence Factors Explained:**
            - **AI Model**: Confidence reported directly by the AI model
            - **Response Quality**: How well-structured the AI response was
            - **Category Specificity**: How specific and definitive the category assignment is
            - **Reasoning Quality**: How detailed and specific the reasoning is
            - **Document Features**: How well document features match the assigned category
            """)

def get_confidence_explanation(confidence_data: dict, category: str) -> dict:
    """
    Generate human-readable explanations of confidence scores
    
    Args:
        confidence_data: Dictionary containing confidence factors and overall confidence
        category: The assigned category
        
    Returns:
        dict: Explanations for overall confidence and individual factors
    """
    overall_confidence = confidence_data.get("overall", 0.0)
    
    # Generate overall confidence explanation
    if overall_confidence >= 0.8:
        overall_explanation = (
            f"High confidence ({overall_confidence:.2f}) in the '{category}' categorization. "
            f"This result is highly reliable and can be trusted."
        )
    elif overall_confidence >= 0.6:
        overall_explanation = (
            f"Medium confidence ({overall_confidence:.2f}) in the '{category}' categorization. "
            f"This result is reasonably reliable but may benefit from verification."
        )
    else:
        overall_explanation = (
            f"Low confidence ({overall_confidence:.2f}) in the '{category}' categorization. "
            f"This result should be verified manually or recategorized."
        )
    
    # Generate factor-specific explanations
    factor_explanations = {}
    
    # AI Reported confidence
    ai_confidence = confidence_data.get("ai_reported", 0.0)
    if ai_confidence >= 0.8:
        factor_explanations["ai_reported"] = f"The AI model is highly confident in its categorization."
    elif ai_confidence >= 0.6:
        factor_explanations["ai_reported"] = f"The AI model has moderate confidence in its categorization."
    else:
        factor_explanations["ai_reported"] = f"The AI model has low confidence in its categorization."
    
    # Response Quality
    response_quality = confidence_data.get("response_quality", 0.0)
    if response_quality >= 0.8:
        factor_explanations["response_quality"] = f"The AI response was well-structured and complete."
    elif response_quality >= 0.6:
        factor_explanations["response_quality"] = f"The AI response was adequately structured."
    else:
        factor_explanations["response_quality"] = f"The AI response was poorly structured or incomplete."
    
    # Category Specificity
    category_specificity = confidence_data.get("category_specificity", 0.0)
    if category_specificity >= 0.8:
        factor_explanations["category_specificity"] = f"The category assignment is very specific and definitive."
    elif category_specificity >= 0.6:
        factor_explanations["category_specificity"] = f"The category assignment is reasonably specific."
    else:
        factor_explanations["category_specificity"] = f"The category assignment is vague or uncertain."
    
    # Reasoning Quality
    reasoning_quality = confidence_data.get("reasoning_quality", 0.0)
    if reasoning_quality >= 0.8:
        factor_explanations["reasoning_quality"] = f"The reasoning is detailed and specific."
    elif reasoning_quality >= 0.6:
        factor_explanations["reasoning_quality"] = f"The reasoning is adequate but could be more detailed."
    else:
        factor_explanations["reasoning_quality"] = f"The reasoning is vague, brief, or contains uncertainty."
    
    # Document Features
    document_features = confidence_data.get("document_features", 0.0)
    if document_features >= 0.8:
        factor_explanations["document_features"] = f"The document features strongly align with the '{category}' category."
    elif document_features >= 0.6:
        factor_explanations["document_features"] = f"The document features somewhat align with the '{category}' category."
    else:
        factor_explanations["document_features"] = f"The document features don't align well with the '{category}' category."
    
    return {
        "overall": overall_explanation,
        "factors": factor_explanations
    }

def configure_confidence_thresholds():
    """
    Configure confidence thresholds for different actions
    """
    # Initialize default thresholds if not in session state
    if "confidence_thresholds" not in st.session_state:
        st.session_state.confidence_thresholds = {
            "auto_accept": 0.85,
            "verification": 0.6,
            "rejection": 0.4
        }
    
    # Create columns for thresholds
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.session_state.confidence_thresholds["auto_accept"] = st.slider(
            "Auto-Accept Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.confidence_thresholds["auto_accept"],
            step=0.05,
            help="Results with confidence above this threshold will be automatically accepted"
        )
    
    with col2:
        st.session_state.confidence_thresholds["verification"] = st.slider(
            "Verification Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.confidence_thresholds["verification"],
            step=0.05,
            help="Results with confidence below this threshold will be flagged for verification"
        )
    
    with col3:
        st.session_state.confidence_thresholds["rejection"] = st.slider(
            "Rejection Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.confidence_thresholds["rejection"],
            step=0.05,
            help="Results with confidence below this threshold will be rejected"
        )
    
    # Add threshold validation
    if st.session_state.confidence_thresholds["rejection"] >= st.session_state.confidence_thresholds["verification"]:
        st.warning("Rejection threshold should be lower than verification threshold")
    
    if st.session_state.confidence_thresholds["verification"] >= st.session_state.confidence_thresholds["auto_accept"]:
        st.warning("Verification threshold should be lower than auto-accept threshold")
    
    # Add threshold explanation
    st.markdown("""
    **Threshold Explanation:**
    - **Auto-Accept**: Results above this threshold are considered highly reliable and can be automatically accepted
    - **Verification**: Results below this threshold require manual verification
    - **Rejection**: Results below this threshold are considered unreliable and should be rejected or recategorized
    """)

def apply_confidence_thresholds(results):
    """
    Apply confidence thresholds to categorization results
    
    Args:
        results: Dictionary of categorization results
        
    Returns:
        dict: Results with threshold flags
    """
    # Get thresholds from session state
    thresholds = st.session_state.get("confidence_thresholds", {
        "auto_accept": 0.85,
        "verification": 0.6,
        "rejection": 0.4
    })
    
    # Apply thresholds to each result
    for file_id, result in results.items():
        # Use calibrated confidence if available, otherwise use original confidence
        confidence = result.get("calibrated_confidence", result.get("confidence", 0.0))
        
        # Set threshold flags
        result["auto_accept"] = confidence >= thresholds["auto_accept"]
        result["needs_verification"] = confidence < thresholds["verification"]
        result["rejected"] = confidence < thresholds["rejection"]
        
        # Set status based on thresholds
        if result["auto_accept"]:
            result["status"] = "Accepted"
        elif result["rejected"]:
            result["status"] = "Rejected"
        elif result["needs_verification"]:
            result["status"] = "Needs Verification"
        else:
            result["status"] = "Review"
    
    return results

def collect_user_feedback(file_id, result):
    """
    Collect user feedback on categorization results
    
    Args:
        file_id: The file ID
        result: The categorization result
    """
    # Initialize feedback in session state if not exists
    if "categorization_feedback" not in st.session_state:
        st.session_state.categorization_feedback = {}
    
    # Get document types
    document_types = [
        "Sales Contract",
        "Invoices",
        "Tax",
        "Financial Report",
        "Employment Contract",
        "PII",
        "Other"
    ]
    
    # Create feedback form
    col1, col2 = st.columns(2)
    
    with col1:
        # Correct category selection
        correct_category = st.selectbox(
            "Correct Category",
            options=document_types,
            index=document_types.index(result["document_type"]) if result["document_type"] in document_types else 0,
            key=f"feedback_category_{file_id}"
        )
    
    with col2:
        # Confidence rating
        confidence_rating = st.select_slider(
            "How confident are you in this categorization?",
            options=["Not at all", "Slightly", "Moderately", "Very", "Extremely"],
            value="Moderately",
            key=f"feedback_confidence_{file_id}"
        )
    
    # Additional feedback
    feedback_text = st.text_area(
        "Additional Feedback",
        key=f"feedback_text_{file_id}"
    )
    
    # Submit button
    if st.button("Submit Feedback", key=f"submit_feedback_{file_id}"):
        # Convert confidence rating to numeric value
        confidence_values = {
            "Not at all": 0.1,
            "Slightly": 0.3,
            "Moderately": 0.5,
            "Very": 0.8,
            "Extremely": 1.0
        }
        
        numeric_confidence = confidence_values.get(confidence_rating, 0.5)
        
        # Store feedback in session state
        st.session_state.categorization_feedback[file_id] = {
            "file_id": file_id,
            "file_name": result["file_name"],
            "original_category": result["document_type"],
            "corrected_category": correct_category,
            "original_confidence": result.get("confidence", 0.0),
            "user_confidence": numeric_confidence,
            "feedback_text": feedback_text,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Update the result with user feedback
        st.session_state.document_categorization["results"][file_id]["document_type"] = correct_category
        st.session_state.document_categorization["results"][file_id]["confidence"] = numeric_confidence
        st.session_state.document_categorization["results"][file_id]["calibrated_confidence"] = numeric_confidence
        st.session_state.document_categorization["results"][file_id]["user_feedback"] = True
        
        st.success("Feedback submitted successfully!")
        
        # Trigger recalibration of confidence model
        calibrate_confidence_model()
        
        return True
    
    return False

def save_categorization_feedback(file_id, original_category, corrected_category):
    """
    Save user feedback on categorization
    """
    if "categorization_feedback" not in st.session_state:
        st.session_state.categorization_feedback = {}
    
    # Get file info
    file_info = st.session_state.document_categorization["results"].get(file_id, {})
    
    # Save feedback
    st.session_state.categorization_feedback[file_id] = {
        "file_id": file_id,
        "file_name": file_info.get("file_name", "Unknown"),
        "original_category": original_category,
        "corrected_category": corrected_category,
        "original_confidence": file_info.get("confidence", 0.0),
        "user_confidence": 1.0,  # Manual override has maximum confidence
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # Log feedback
    logger.info(f"Categorization feedback saved for {file_id}: {original_category} -> {corrected_category}")
    
    # Trigger confidence calibration
    calibrate_confidence_model()

def calibrate_confidence_model():
    """
    Calibrate confidence model based on user feedback
    """
    # Check if we have enough feedback for calibration
    if "categorization_feedback" not in st.session_state:
        return
    
    feedback = list(st.session_state.categorization_feedback.values())
    
    if len(feedback) < 3:  # Need at least 3 feedback items for meaningful calibration
        return
    
    # Calculate calibration factors
    category_confidence_adjustments = {}
    
    # Group feedback by original category
    by_category = {}
    for item in feedback:
        category = item["original_category"]
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(item)
    
    # Calculate adjustment factors for each category
    for category, items in by_category.items():
        if len(items) < 2:  # Need at least 2 items per category for adjustment
            continue
        
        # Calculate average confidence adjustment
        avg_original = sum(item["original_confidence"] for item in items) / len(items)
        avg_user = sum(item["user_confidence"] for item in items) / len(items)
        
        # Calculate adjustment factor (multiplicative)
        if avg_original > 0:
            adjustment = avg_user / avg_original
        else:
            adjustment = 1.0
        
        # Store adjustment factor
        category_confidence_adjustments[category] = adjustment
    
    # Store calibration factors in session state
    st.session_state.confidence_calibration = {
        "category_adjustments": category_confidence_adjustments,
        "last_updated": datetime.datetime.now().isoformat(),
        "feedback_count": len(feedback)
    }
    
    logger.info(f"Confidence calibration updated: {st.session_state.confidence_calibration}")

def apply_confidence_calibration(category, confidence):
    """
    Apply confidence calibration to a confidence score
    
    Args:
        category: Document category
        confidence: Original confidence score
        
    Returns:
        float: Calibrated confidence score
    """
    # Check if we have calibration data
    if "confidence_calibration" not in st.session_state:
        return confidence
    
    calibration = st.session_state.confidence_calibration
    
    # Get category adjustment factor
    adjustment = calibration.get("category_adjustments", {}).get(category, 1.0)
    
    # Apply adjustment (with limits to prevent extreme values)
    calibrated = confidence * adjustment
    
    # Ensure confidence is between 0 and 1
    calibrated = max(0.0, min(1.0, calibrated))
    
    return calibrated

def validate_confidence_with_examples():
    """
    Validate confidence scores against known examples
    """
    # Initialize validation examples if not in session state
    if "validation_examples" not in st.session_state:
        st.session_state.validation_examples = {}
    
    # Add example button
    if st.button("Add Validation Example"):
        # Create a unique key for the new example
        example_key = f"example_{len(st.session_state.validation_examples) + 1}"
        
        # Add the example to session state
        st.session_state.validation_examples[example_key] = {
            "file_id": "",
            "file_name": "",
            "actual_category": "",
            "validated": False
        }
    
    # Display validation examples
    for example_key, example in list(st.session_state.validation_examples.items()):
        with st.container():
            st.write(f"**Validation Example: {example_key}**")
            
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                # File selection
                selected_files = st.session_state.get("selected_files", [])
                file_options = [{"label": f["name"], "value": f["id"]} for f in selected_files]
                
                selected_file = st.selectbox(
                    "Select File",
                    options=[{"label": "Select a file...", "value": ""}] + file_options,
                    format_func=lambda x: x["label"],
                    key=f"file_select_{example_key}"
                )
                
                if selected_file and selected_file["value"]:
                    example["file_id"] = selected_file["value"]
                    example["file_name"] = selected_file["label"]
            
            with col2:
                # Category selection
                document_types = [
                    "Sales Contract",
                    "Invoices",
                    "Tax",
                    "Financial Report",
                    "Employment Contract",
                    "PII",
                    "Other"
                ]
                
                example["actual_category"] = st.selectbox(
                    "Actual Category",
                    options=[""] + document_types,
                    key=f"category_select_{example_key}"
                )
            
            with col3:
                # Validation button
                validate_button = st.button("Validate", key=f"validate_button_{example_key}")
                delete_button = st.button("Delete", key=f"delete_button_{example_key}")
                
                if validate_button and example["file_id"] and example["actual_category"]:
                    # Run categorization on the file
                    result = categorize_document(example["file_id"])
                    
                    # Store validation result
                    example["predicted_category"] = result["document_type"]
                    example["confidence"] = result["confidence"]
                    example["reasoning"] = result["reasoning"]
                    example["validated"] = True
                    
                    # Calculate multi-factor confidence
                    document_features = extract_document_features(example["file_id"])
                    document_types = [
                        "Sales Contract",
                        "Invoices",
                        "Tax",
                        "Financial Report",
                        "Employment Contract",
                        "PII",
                        "Other"
                    ]
                    example["multi_factor_confidence"] = calculate_multi_factor_confidence(
                        result["confidence"],
                        document_features,
                        result["document_type"],
                        result["reasoning"],
                        document_types
                    )
                
                if delete_button:
                    # Remove the example from session state
                    del st.session_state.validation_examples[example_key]
                    st.rerun()
            
            # Display validation results if validated
            if example.get("validated"):
                correct = example["actual_category"] == example["predicted_category"]
                
                if correct:
                    st.success(f"Correctly categorized as '{example['predicted_category']}' with confidence {example['confidence']:.2f}")
                else:
                    st.error(f"Incorrectly categorized as '{example['predicted_category']}' (should be '{example['actual_category']}') with confidence {example['confidence']:.2f}")
                
                # Display confidence visualization
                if "multi_factor_confidence" in example:
                    display_confidence_visualization(example["multi_factor_confidence"])
                
                # Display reasoning
                with st.expander("Reasoning", expanded=False):
                    st.write(example.get("reasoning", ""))
    
    # Display validation summary if we have validated examples
    validated_examples = [e for e in st.session_state.validation_examples.values() if e.get("validated")]
    
    if validated_examples:
        st.subheader("Validation Summary")
        
        # Calculate statistics
        total = len(validated_examples)
        correct = sum(1 for e in validated_examples if e["actual_category"] == e["predicted_category"])
        accuracy = correct / total if total > 0 else 0
        
        # Group by confidence ranges
        high_conf = [e for e in validated_examples if e.get("confidence", 0) >= 0.8]
        med_conf = [e for e in validated_examples if 0.6 <= e.get("confidence", 0) < 0.8]
        low_conf = [e for e in validated_examples if e.get("confidence", 0) < 0.6]
        
        high_correct = sum(1 for e in high_conf if e["actual_category"] == e["predicted_category"])
        med_correct = sum(1 for e in med_conf if e["actual_category"] == e["predicted_category"])
        low_correct = sum(1 for e in low_conf if e["actual_category"] == e["predicted_category"])
        
        high_accuracy = high_correct / len(high_conf) if high_conf else 0
        med_accuracy = med_correct / len(med_conf) if med_conf else 0
        low_accuracy = low_correct / len(low_conf) if low_conf else 0
        
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Overall Accuracy", f"{accuracy:.0%}")
        
        with col2:
            st.metric("High Confidence Accuracy", f"{high_accuracy:.0%}")
        
        with col3:
            st.metric("Medium Confidence Accuracy", f"{med_accuracy:.0%}")
        
        with col4:
            st.metric("Low Confidence Accuracy", f"{low_accuracy:.0%}")

def combine_categorization_results(results):
    """
    Combine results from multiple models using weighted voting
    
    Args:
        results: List of categorization results from different models
        
    Returns:
        dict: Combined categorization result
    """
    if not results:
        return {
            "document_type": "Other",
            "confidence": 0.0,
            "reasoning": "No results to combine"
        }
    
    if len(results) == 1:
        return results[0]
    
    # Count votes for each category, weighted by confidence
    category_votes = {}
    all_reasonings = []
    
    for result in results:
        category = result["document_type"]
        confidence = result["confidence"]
        reasoning = result.get("reasoning", "")
        
        # Add weighted vote
        if category not in category_votes:
            category_votes[category] = 0
        
        category_votes[category] += confidence
        all_reasonings.append(f"Model vote: {category} (confidence: {confidence:.2f})\nReasoning: {reasoning}")
    
    # Find category with highest votes
    if not category_votes:
        winning_category = "Other"
        winning_votes = 0
    else:
        winning_category = max(category_votes.items(), key=lambda x: x[1])[0]
        winning_votes = category_votes[winning_category]
    
    # Calculate consensus confidence
    total_possible_votes = len(results)  # If all models voted with 100% confidence
    consensus_confidence = winning_votes / total_possible_votes
    
    # Create combined reasoning
    combined_reasoning = f"Consensus category: {winning_category} with {consensus_confidence:.2f} confidence.\n\n"
    combined_reasoning += "\n\n".join(all_reasonings)
    
    return {
        "document_type": winning_category,
        "confidence": consensus_confidence,
        "reasoning": combined_reasoning
    }

def get_document_preview_url(file_id, page=1):
    """
    Get document preview URL using Box API
    
    Args:
        file_id: Box file ID
        page: Page number to preview
        
    Returns:
        str: Preview URL or None if not available
    """
    try:
        client = st.session_state.client
        
        # Get thumbnail URL for the file
        preview_url = client.file(file_id).get_thumbnail(
            extension='png',
            min_width=400,
            min_height=400
        )
        
        return preview_url
    except Exception as e:
        logger.warning(f"Error getting document preview: {str(e)}")
        return None
