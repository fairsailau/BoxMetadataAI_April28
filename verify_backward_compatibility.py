import streamlit as st
import logging
import json
import os
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_backward_compatibility():
    """
    Verify backward compatibility of the enhanced processing functionality
    """
    print("Verifying backward compatibility of enhanced processing...")
    
    # Initialize session state for testing
    if not hasattr(st.session_state, "metadata_config"):
        st.session_state.metadata_config = {
            "extraction_method": "freeform",
            "freeform_prompt": "Extract key metadata from this document including dates, names, amounts, and other important information.",
            "use_template": False,
            "template_id": "",
            "custom_fields": [],
            "ai_model": "google__gemini_2_0_flash_001",
            "batch_size": 5
        }
    
    # Import the enhanced processing module
    from modules.enhanced_processing import process_file
    
    # Create a mock extraction function for testing
    def mock_extract_freeform_metadata(file_id, prompt, ai_model):
        print(f"Mock extraction for file {file_id} with prompt: {prompt[:50]}...")
        return {"answer": {"extracted": "data", "prompt_used": prompt}}
    
    # Create a mock extraction function for structured metadata
    def mock_extract_structured_metadata(file_id, metadata_template=None, fields=None, ai_model=None):
        if metadata_template:
            print(f"Mock structured extraction for file {file_id} with template: {metadata_template}")
            return {"answer": {"extracted": "data", "template_used": metadata_template}}
        elif fields:
            print(f"Mock structured extraction for file {file_id} with fields: {fields}")
            return {"answer": {"extracted": "data", "fields_used": fields}}
        return {"error": "No template or fields provided"}
    
    # Create mock extraction functions
    extraction_functions = {
        "extract_freeform_metadata": mock_extract_freeform_metadata,
        "extract_structured_metadata": mock_extract_structured_metadata
    }
    
    # Test 1: Basic freeform extraction without document categorization
    print("\nTest 1: Basic freeform extraction without document categorization")
    
    # Remove document categorization from session state if it exists
    if hasattr(st.session_state, "document_categorization"):
        del st.session_state.document_categorization
    
    # Set freeform extraction method
    st.session_state.metadata_config["extraction_method"] = "freeform"
    
    # Process a file
    file = {"id": "file1", "name": "test_document.pdf"}
    result = process_file(file, extraction_functions)
    
    # Check if the general prompt was used
    if result["success"] and "prompt_used" in result["data"]:
        prompt_used = result["data"]["prompt_used"]
        expected_prompt = st.session_state.metadata_config["freeform_prompt"]
        
        if prompt_used == expected_prompt:
            print("✅ Successfully used general prompt when no document categorization exists")
        else:
            print("❌ Failed to use general prompt")
            print(f"Expected: {expected_prompt}")
            print(f"Got: {prompt_used}")
    else:
        print("❌ Failed to process file or retrieve prompt used")
    
    # Test 2: Structured extraction with template without document categorization
    print("\nTest 2: Structured extraction with template without document categorization")
    
    # Set structured extraction method with template
    st.session_state.metadata_config["extraction_method"] = "structured"
    st.session_state.metadata_config["use_template"] = True
    st.session_state.metadata_config["template_id"] = "enterprise_123456_general"
    
    # Process a file
    file = {"id": "file1", "name": "test_document.pdf"}
    result = process_file(file, extraction_functions)
    
    # Check if the general template was used
    if result["success"] and "template_used" in result["data"]:
        template_used = result["data"]["template_used"]
        expected_template_key = "general"
        
        if expected_template_key in str(template_used) or "enterprise_123456_general" in str(template_used):
            print("✅ Successfully used general template when no document categorization exists")
        else:
            print("❌ Failed to use general template")
            print(f"Expected template key: {expected_template_key}")
            print(f"Got: {template_used}")
    else:
        print("❌ Failed to process file or retrieve template used")
    
    # Test 3: Structured extraction with custom fields without document categorization
    print("\nTest 3: Structured extraction with custom fields without document categorization")
    
    # Set structured extraction method with custom fields
    st.session_state.metadata_config["extraction_method"] = "structured"
    st.session_state.metadata_config["use_template"] = False
    st.session_state.metadata_config["custom_fields"] = [
        {"name": "Field1", "type": "string"},
        {"name": "Field2", "type": "number"}
    ]
    
    # Process a file
    file = {"id": "file1", "name": "test_document.pdf"}
    result = process_file(file, extraction_functions)
    
    # Check if the custom fields were used
    if result["success"] and "fields_used" in result["data"]:
        fields_used = result["data"]["fields_used"]
        
        if fields_used and len(fields_used) == 2:
            print("✅ Successfully used custom fields when no document categorization exists")
        else:
            print("❌ Failed to use custom fields")
            print(f"Expected 2 fields")
            print(f"Got: {fields_used}")
    else:
        print("❌ Failed to process file or retrieve fields used")
    
    # Test 4: Processing with feedback data
    print("\nTest 4: Processing with feedback data")
    
    # Add feedback data to session state
    if not hasattr(st.session_state, "feedback_data"):
        st.session_state.feedback_data = {}
    
    # Add feedback for the test file
    feedback_key = f"file1_freeform"
    st.session_state.feedback_data[feedback_key] = {
        "feedback_field1": "feedback_value1",
        "feedback_field2": "feedback_value2"
    }
    
    # Set freeform extraction method
    st.session_state.metadata_config["extraction_method"] = "freeform"
    
    # Process a file
    file = {"id": "file1", "name": "test_document.pdf"}
    result = process_file(file, extraction_functions)
    
    # Check if the feedback data was applied
    if result["success"]:
        data = result["data"]
        
        if "feedback_field1" in data and data["feedback_field1"] == "feedback_value1":
            print("✅ Successfully applied feedback data")
        else:
            print("❌ Failed to apply feedback data")
            print(f"Expected feedback_field1: feedback_value1")
            print(f"Got: {data}")
    else:
        print("❌ Failed to process file")
    
    print("\nBackward compatibility verification completed.")
    return {
        "success": True,
        "results": {
            "test1": "Basic freeform extraction without document categorization",
            "test2": "Structured extraction with template without document categorization",
            "test3": "Structured extraction with custom fields without document categorization",
            "test4": "Processing with feedback data"
        }
    }

if __name__ == "__main__":
    test_results = verify_backward_compatibility()
    print(f"\nTest results: {test_results}")
