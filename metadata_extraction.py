import streamlit as st
import logging
import json
import requests
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def metadata_extraction():
    """
    Implement metadata extraction using Box AI API
    
    Returns:
        dict: Dictionary of extraction functions
    """
    # Structured metadata extraction
    def extract_structured_metadata(file_id, fields=None, metadata_template=None, ai_model="azure__openai__gpt_4o_mini"):
        """
        Extract structured metadata from a file using Box AI API
        
        Args:
            file_id (str): Box file ID
            fields (list): List of field definitions for extraction
            metadata_template (dict): Metadata template definition
            ai_model (str): AI model to use for extraction
            
        Returns:
            dict: Extracted metadata
        """
        try:
            # Get client from session state
            client = st.session_state.client
            
            # Get access token from client
            access_token = None
            if hasattr(client, '_oauth'):
                access_token = client._oauth.access_token
            elif hasattr(client, 'auth') and hasattr(client.auth, 'access_token'):
                access_token = client.auth.access_token
            
            if not access_token:
                raise ValueError("Could not retrieve access token from client")
            
            # Set headers
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            # Create AI agent configuration with proper format for structured extraction
            # Enhanced with confidence scoring instructions
            ai_agent = {
                "type": "ai_agent_extract_structured",
                "long_text": {
                    "model": ai_model,
                    "mode": "default",
                    "system_message": "You are an AI assistant specialized in extracting metadata from documents based on provided field definitions. For each field, analyze the document content and extract the corresponding value. CRITICALLY IMPORTANT: Respond for EACH field with a JSON object containing two keys: 1. \"value\": The extracted metadata value as a string. 2. \"confidence\": Your confidence level for this specific extraction, chosen from ONLY these three options: \"High\", \"Medium\", or \"Low\". Base your confidence on how certain you are about the extracted value given the document content and field definition. Example Response for a field: {\"value\": \"INV-12345\", \"confidence\": \"High\"}"
                },
                "basic_text": {
                    "model": ai_model,
                    "mode": "default",
                    "system_message": "You are an AI assistant specialized in extracting metadata from documents based on provided field definitions. For each field, analyze the document content and extract the corresponding value. CRITICALLY IMPORTANT: Respond for EACH field with a JSON object containing two keys: 1. \"value\": The extracted metadata value as a string. 2. \"confidence\": Your confidence level for this specific extraction, chosen from ONLY these three options: \"High\", \"Medium\", or \"Low\". Base your confidence on how certain you are about the extracted value given the document content and field definition. Example Response for a field: {\"value\": \"INV-12345\", \"confidence\": \"High\"}"
                }
            }
            
            # Create items array with file ID
            items = [{"id": file_id, "type": "file"}]
            
            # Construct API URL for Box AI Extract Structured
            api_url = "https://api.box.com/2.0/ai/extract_structured"
            
            # Construct request body
            request_body = {
                "items": items,
                "ai_agent": ai_agent
            }
            
            # Add template or fields
            if metadata_template:
                request_body["metadata_template"] = metadata_template
            elif fields:
            # Convert fields to Box API format if needed
                api_fields = []
                for field in fields:
                    if "key" in field:
                        # Field is already in Box API format
                        api_fields.append(field)
                    else:
                        # Convert field to Box API format
                        api_field = {
                            "key": field.get("name", ""),
                            "displayName": field.get("display_name", field.get("name", "")),
                            "type": field.get("type", "string")
                        }
                        
                        # Add description and prompt if available
                        if "description" in field:
                            api_field["description"] = field["description"]
                        if "prompt" in field:
                            api_field["prompt"] = field["prompt"]
                        
                        # Add options for enum fields
                        if field.get("type") == "enum" and "options" in field:
                            api_field["options"] = field["options"]
                        
                        api_fields.append(api_field)
                
                request_body["fields"] = api_fields
            else:
                raise ValueError("Either fields or metadata_template must be provided")
            
            # Make API call
            logger.info(f"Making Box AI API call for structured extraction with request: {json.dumps(request_body)}")
            response = requests.post(api_url, headers=headers, json=request_body)
            
            # Check response
            if response.status_code != 200:
                logger.error(f"Box AI API error response: {response.text}")
                return {"error": f"Error in Box AI API call: {response.status_code} {response.reason}"}
            
            # Parse response
            response_data = response.json()
            
            # Process the response to extract confidence levels
            processed_response = {}
            
            # Check if we have entries in the response
            if "entries" in response_data and len(response_data["entries"]) > 0:
                # Get the first entry (should be our file)
                entry = response_data["entries"][0]
                
                # Check if we have metadata fields
                if "metadata" in entry:
                    metadata = entry["metadata"]
                    
                    # Process each field to extract value and confidence
                    for field_key, field_value in metadata.items():
                        try:
                            # Try to parse the field value as JSON to extract value and confidence
                            if isinstance(field_value, str):
                                try:
                                    # Attempt to parse as JSON
                                    parsed_value = json.loads(field_value)
                                    if isinstance(parsed_value, dict) and "value" in parsed_value and "confidence" in parsed_value:
                                        # Store the parsed value and confidence separately
                                        processed_response[field_key] = parsed_value["value"]
                                        processed_response[f"{field_key}_confidence"] = parsed_value["confidence"]
                                    else:
                                        # If not in expected format, store as is
                                        processed_response[field_key] = field_value
                                        processed_response[f"{field_key}_confidence"] = "Medium"  # Default confidence
                                except json.JSONDecodeError:
                                    # If not valid JSON, store as is
                                    processed_response[field_key] = field_value
                                    processed_response[f"{field_key}_confidence"] = "Medium"  # Default confidence
                            else:
                                # For non-string values, store as is
                                processed_response[field_key] = field_value
                                processed_response[f"{field_key}_confidence"] = "Medium"  # Default confidence
                        except Exception as e:
                            logger.warning(f"Error processing field {field_key}: {str(e)}")
                            processed_response[field_key] = field_value
                            processed_response[f"{field_key}_confidence"] = "Low"  # Low confidence due to error
            
            # If we couldn't process the response, return the original
            if not processed_response and "entries" in response_data:
                # Store the original response
                processed_response = response_data
                processed_response["_confidence_processing_failed"] = True
            
            # Return the processed response
            return processed_response
        
        except Exception as e:
            logger.error(f"Error in Box AI API call: {str(e)}")
            return {"error": str(e)}
    
    # Freeform metadata extraction
    def extract_freeform_metadata(file_id, prompt, ai_model="azure__openai__gpt_4o_mini"):
        """
        Extract freeform metadata from a file using Box AI API
        
        Args:
            file_id (str): Box file ID
            prompt (str): Extraction prompt
            ai_model (str): AI model to use for extraction
            
        Returns:
            dict: Extracted metadata
        """
        try:
            # Get client from session state
            client = st.session_state.client
            
            # Get access token from client
            access_token = None
            if hasattr(client, '_oauth'):
                access_token = client._oauth.access_token
            elif hasattr(client, 'auth') and hasattr(client.auth, 'access_token'):
                access_token = client.auth.access_token
            
            if not access_token:
                raise ValueError("Could not retrieve access token from client")
            
            # Set headers
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            # Enhance the prompt to request confidence levels
            enhanced_prompt = prompt
            if not "confidence" in prompt.lower():
                enhanced_prompt = prompt + " For each extracted field, provide your confidence level (High, Medium, or Low) in the accuracy of the extraction. Format your response as a JSON object with each field having a nested object containing 'value' and 'confidence' keys."
            
            # Create AI agent configuration with confidence scoring instructions
            ai_agent = {
                "type": "ai_agent_extract",
                "long_text": {
                    "model": ai_model,
                    "system_message": "You are an AI assistant specialized in extracting metadata from documents. Extract the requested information and for EACH extracted field, provide both the value and your confidence level (High, Medium, or Low). Format your response as a JSON object where each field has a nested object containing 'value' and 'confidence' keys. Example: {\"InvoiceNumber\": {\"value\": \"INV-12345\", \"confidence\": \"High\"}, \"Date\": {\"value\": \"2023-04-15\", \"confidence\": \"Medium\"}}"
                },
                "basic_text": {
                    "model": ai_model,
                    "system_message": "You are an AI assistant specialized in extracting metadata from documents. Extract the requested information and for EACH extracted field, provide both the value and your confidence level (High, Medium, or Low). Format your response as a JSON object where each field has a nested object containing 'value' and 'confidence' keys. Example: {\"InvoiceNumber\": {\"value\": \"INV-12345\", \"confidence\": \"High\"}, \"Date\": {\"value\": \"2023-04-15\", \"confidence\": \"Medium\"}}"
                }
            }
            
            # Create items array with file ID
            items = [{"id": file_id, "type": "file"}]
            
            # Construct API URL for Box AI Extract
            api_url = "https://api.box.com/2.0/ai/extract"
            
            # Construct request body
            request_body = {
                "items": items,
                "prompt": enhanced_prompt,
                "ai_agent": ai_agent
            }
            
            # Make API call
            logger.info(f"Making Box AI API call for freeform extraction with request: {json.dumps(request_body)}")
            response = requests.post(api_url, headers=headers, json=request_body)
            
            # Check response
            if response.status_code != 200:
                logger.error(f"Box AI API error response: {response.text}")
                return {"error": f"Error in Box AI API call: {response.status_code} {response.reason}"}
            
            # Parse response
            response_data = response.json()
            
            # Process the response to extract confidence levels
            processed_response = {}
            
            # Check if we have entries in the response
            if "entries" in response_data and len(response_data["entries"]) > 0:
                # Get the first entry (should be our file)
                entry = response_data["entries"][0]
                
                # Check if we have a response
                if "response" in entry:
                    response_text = entry["response"]
                    
                    # Try to parse the response as JSON
                    try:
                        # Look for JSON in the response
                        json_start = response_text.find('{')
                        json_end = response_text.rfind('}') + 1
                        
                        if json_start >= 0 and json_end > json_start:
                            json_str = response_text[json_start:json_end]
                            parsed_response = json.loads(json_str)
                            
                            # Process the parsed JSON to extract values and confidence levels
                            for field_key, field_value in parsed_response.items():
                                if isinstance(field_value, dict) and "value" in field_value and "confidence" in field_value:
                                    # Store the value and confidence separately
                                    processed_response[field_key] = field_value["value"]
                                    processed_response[f"{field_key}_confidence"] = field_value["confidence"]
                                else:
                                    # If not in expected format, store as is
                                    processed_response[field_key] = field_value
                                    processed_response[f"{field_key}_confidence"] = "Medium"  # Default confidence
                        else:
                            # If no JSON found, store the raw response
                            processed_response["_raw_response"] = response_text
                    except json.JSONDecodeError:
                        # If not valid JSON, store the raw response
                        processed_response["_raw_response"] = response_text
            
            # If we couldn't process the response, return the original
            if not processed_response and "entries" in response_data:
                # Store the original response
                processed_response = response_data
                processed_response["_confidence_processing_failed"] = True
            
            # Return the processed response
            return processed_response
        
        except Exception as e:
            logger.error(f"Error in Box AI API call: {str(e)}")
            return {"error": str(e)}
    
    # Return the extraction functions for use in other modules
    return {
        "extract_structured_metadata": extract_structured_metadata,
        "extract_freeform_metadata": extract_freeform_metadata
    }

# For backward compatibility
def extract_metadata_freeform(client, file_id, prompt=None, ai_model="azure__openai__gpt_4o_mini"):
    """
    Backward compatibility wrapper for extract_freeform_metadata
    """
    # Get extraction functions
    extraction_functions = metadata_extraction()
    
    # Call the actual function
    return extraction_functions["extract_freeform_metadata"](
        file_id=file_id,
        prompt=prompt or "Extract key metadata from this document including dates, names, amounts, and other important information.",
        ai_model=ai_model
    )

def extract_metadata_structured(client, file_id, template_id=None, custom_fields=None, ai_model="azure__openai__gpt_4o_mini"):
    """
    Backward compatibility wrapper for extract_structured_metadata
    """
    # Get extraction functions
    extraction_functions = metadata_extraction()
    
    # Prepare parameters
    if template_id:
        # Get template details
        template = get_template_by_id(template_id)
        if template:
            # Parse the template ID to extract the correct components
            # Format is typically: scope_id_templateKey (e.g., enterprise_336904155_financialReport)
            parts = template_id.split('_')
            
            # Extract the scope and enterprise ID
            scope = parts[0]  # e.g., "enterprise"
            enterprise_id = parts[1] if len(parts) > 1 else ""
            
            # Extract the actual template key (last part)
            template_key = parts[-1] if len(parts) > 2 else template["key"]
            
            # Create metadata template reference with correct format according to Box API documentation
            metadata_template = {
                "template_key": template_key,
                "type": "metadata_template",
                "scope": f"{scope}_{enterprise_id}"
            }
        else:
            raise ValueError(f"Template with ID {template_id} not found")
        
        # Call the actual function with template
        return extraction_functions["extract_structured_metadata"](
            file_id=file_id,
            metadata_template=metadata_template,
            ai_model=ai_model
        )
    elif custom_fields:
        # Call the actual function with fields
        return extraction_functions["extract_structured_metadata"](
            file_id=file_id,
            fields=custom_fields,
            ai_model=ai_model
        )
    else:
        raise ValueError("Either template_id or custom_fields must be provided")

def get_template_by_id(template_id):
    """
    Get template by ID from session state
    
    Args:
        template_id: Template ID
        
    Returns:
        dict: Template or None if not found
    """
    if not template_id:
        return None
    
    if not hasattr(st.session_state, "metadata_templates") or not st.session_state.metadata_templates:
        return None
    
    return st.session_state.metadata_templates.get(template_id)
