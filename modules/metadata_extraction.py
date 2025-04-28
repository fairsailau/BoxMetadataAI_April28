import streamlit as st
import logging
import json
import requests
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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
            dict: Extracted metadata with confidence scores
        """
        try:
            # Get client from session state
            client = st.session_state.client
            
            # Get access token from client
            access_token = None
            if hasattr(client, "_oauth"):
                access_token = client._oauth.access_token
            elif hasattr(client, "auth") and hasattr(client.auth, "access_token"):
                access_token = client.auth.access_token
            
            if not access_token:
                raise ValueError("Could not retrieve access token from client")
            
            # Set headers
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            # Create AI agent configuration with proper format for structured extraction
            # Enhanced with confidence scoring instructions
            ai_agent = {
                "type": "ai_agent_extract_structured",
                "long_text": {
                    "model": ai_model,
                    "mode": "default",
                    "system_message": "You are an AI assistant specialized in extracting metadata from documents based on provided field definitions. For each field, analyze the document content and extract the corresponding value. CRITICALLY IMPORTANT: Respond for EACH field with a JSON object containing two keys: 1. \\\"value\\\": The extracted metadata value as a string. 2. \\\"confidence\\\": Your confidence level for this specific extraction, chosen from ONLY these three options: \\\"High\\\", \\\"Medium\\\", or \\\"Low\\\". Base your confidence on how certain you are about the extracted value given the document content and field definition. Example Response for a field: {\\\"value\\\": \\\"INV-12345\\\", \\\"confidence\\\": \\\"High\\\"}"
                },
                "basic_text": {
                    "model": ai_model,
                    "mode": "default",
                    "system_message": "You are an AI assistant specialized in extracting metadata from documents based on provided field definitions. For each field, analyze the document content and extract the corresponding value. CRITICALLY IMPORTANT: Respond for EACH field with a JSON object containing two keys: 1. \\\"value\\\": The extracted metadata value as a string. 2. \\\"confidence\\\": Your confidence level for this specific extraction, chosen from ONLY these three options: \\\"High\\\", \\\"Medium\\\", or \\\"Low\\\". Base your confidence on how certain you are about the extracted value given the document content and field definition. Example Response for a field: {\\\"value\\\": \\\"INV-12345\\\", \\\"confidence\\\": \\\"High\\\"}"
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
            logger.info(f"Raw Box AI structured extraction response data: {json.dumps(response_data)}") # Added logging
            
            # Process the response to extract confidence levels
            processed_response = {}
            
            # *** REVISED LOGIC TO HANDLE DIFFERENT RESPONSE FORMATS ***
            if "answer" in response_data and isinstance(response_data["answer"], dict):
                answer_dict = response_data["answer"]
                
                # Check for 'fields' array format FIRST
                if "fields" in answer_dict and isinstance(answer_dict["fields"], list):
                    logger.info("Processing 'answer' with 'fields' array format.")
                    fields_array = answer_dict["fields"]
                    for field_item in fields_array:
                        if isinstance(field_item, dict) and "key" in field_item and "value" in field_item:
                            field_key = field_item["key"]
                            extracted_value = field_item["value"]
                            # Get confidence if available, otherwise default to Medium
                            confidence_level = field_item.get("confidence", "Medium")
                            
                            # Validate confidence value
                            if confidence_level not in ["High", "Medium", "Low"]:
                                logger.warning(f"Field {field_key}: Unexpected confidence value '{confidence_level}', defaulting to Medium.")
                                confidence_level = "Medium"
                                
                            # Store the extracted value and confidence
                            processed_response[field_key] = extracted_value
                            processed_response[f"{field_key}_confidence"] = confidence_level
                        else:
                            logger.warning(f"Skipping invalid item in 'fields' array: {field_item}")
                
                # Else, process as standard key-value pairs
                else:
                    logger.info("Processing 'answer' as standard key-value dictionary.")
                    for field_key, field_data in answer_dict.items():
                        # Default values
                        extracted_value = None
                        confidence_level = "Medium" # Default if parsing fails or not provided

                        try:
                            # Check if field_data is a dictionary with 'value' and 'confidence'
                            if isinstance(field_data, dict) and "value" in field_data and "confidence" in field_data:
                                extracted_value = field_data["value"]
                                confidence_level = field_data["confidence"]
                                
                                # Validate confidence value
                                if confidence_level not in ["High", "Medium", "Low"]:
                                    logger.warning(f"Field {field_key}: Unexpected confidence value '{confidence_level}', defaulting to Medium.")
                                    confidence_level = "Medium"
                            
                            # Handle cases where field_data might be None
                            elif field_data is None:
                                logger.info(f"Field {field_key}: Received null value. Setting value to None and confidence to Low.")
                                extracted_value = None
                                confidence_level = "Low"
                            
                            # Handle cases where field_data is a dict with only 'value'
                            elif isinstance(field_data, dict) and "value" in field_data and len(field_data) == 1:
                                logger.warning(f"Field {field_key}: Found dict with only 'value' key: {field_data}. Extracting value directly.")
                                extracted_value = field_data["value"]
                                confidence_level = "Medium" # Default confidence as it was missing
                            
                            # Otherwise treat the whole field_data as the value
                            else:
                                logger.warning(f"Field {field_key}: Unexpected data format: {field_data}. Using raw data as value and Medium confidence.")
                                extracted_value = field_data # Use the raw data
                                confidence_level = "Medium"

                            # Store the extracted value and confidence
                            processed_response[field_key] = extracted_value
                            processed_response[f"{field_key}_confidence"] = confidence_level

                        except Exception as e:
                            logger.error(f"Error processing field {field_key} with data '{field_data}': {str(e)}")
                            processed_response[field_key] = field_data # Store original data on error
                            processed_response[f"{field_key}_confidence"] = "Low" # Low confidence due to processing error
            
            # Handle freeform response format (string in 'answer')
            elif "answer" in response_data and isinstance(response_data["answer"], str):
                 logger.info("Processing 'answer' as string (potential freeform JSON).")
                 response_text = response_data["answer"]
                 # Try to parse JSON from the text
                 try:
                     # Look for JSON-like content in the string
                     json_start = response_text.find('{')
                     json_end = response_text.rfind('}') + 1
                     
                     if json_start != -1 and json_end > json_start:
                         json_str = response_text[json_start:json_end]
                         parsed_json = json.loads(json_str)
                         
                         if isinstance(parsed_json, dict):
                             # Process each field in the parsed JSON
                             for field_key, field_data in parsed_json.items():
                                 if isinstance(field_data, dict) and "value" in field_data and "confidence" in field_data:
                                     # Extract value and confidence
                                     extracted_value = field_data["value"]
                                     confidence_level = field_data["confidence"]
                                     
                                     # Validate confidence
                                     if confidence_level not in ["High", "Medium", "Low"]:
                                         confidence_level = "Medium"
                                         
                                     # Store in processed response
                                     processed_response[field_key] = extracted_value
                                     processed_response[f"{field_key}_confidence"] = confidence_level
                                 else:
                                     # Use the field_data as is with Medium confidence
                                     processed_response[field_key] = field_data
                                     processed_response[f"{field_key}_confidence"] = "Medium"
                         else:
                             # Not a dictionary, store raw text
                             logger.warning(f"Parsed JSON from 'answer' string is not a dictionary: {parsed_json}")
                             processed_response["_raw_response"] = response_text
                             processed_response["_confidence_processing_failed"] = True
                     else:
                         # No JSON found, store raw text
                         logger.warning("No JSON object found in 'answer' string.")
                         processed_response["_raw_response"] = response_text
                         processed_response["_confidence_processing_failed"] = True
                 except Exception as e:
                     logger.error(f"Error parsing JSON from answer string: {str(e)}")
                     processed_response["_raw_response"] = response_text
                     processed_response["_confidence_processing_failed"] = True
            
            # Fall back to entries format if answer is not present (for backward compatibility)
            elif "entries" in response_data and len(response_data["entries"]) > 0:
                logger.info("Processing response using fallback 'entries' format.")
                # Get the first entry (should be our file)
                entry = response_data["entries"][0]
                
                # Check if we have metadata fields
                if "metadata" in entry:
                    metadata = entry["metadata"]
                    
                    # Process each field to extract value and confidence
                    for field_key, field_value in metadata.items():
                        # Default values
                        extracted_value = field_value
                        confidence_level = "Medium" # Default if parsing fails or not provided

                        try:
                            # Check if field_value is a string that looks like our requested JSON
                            if isinstance(field_value, str) and field_value.strip().startswith('{') and field_value.strip().endswith('}'):
                                try:
                                    # Attempt to parse as JSON
                                    parsed_value = json.loads(field_value)
                                    if isinstance(parsed_value, dict) and "value" in parsed_value and "confidence" in parsed_value:
                                        # Successfully parsed the expected JSON structure
                                        extracted_value = parsed_value["value"]
                                        confidence_level = parsed_value["confidence"]
                                        # Validate confidence value
                                        if confidence_level not in ["High", "Medium", "Low"]:
                                            logger.warning(f"Field {field_key}: Unexpected confidence value '{confidence_level}', defaulting to Medium.")
                                            confidence_level = "Medium"
                                    else:
                                        # Parsed JSON but not the expected structure
                                        logger.warning(f"Field {field_key}: Parsed JSON but keys 'value' and 'confidence' not found. Using raw value.")
                                        extracted_value = field_value # Keep original string
                                        confidence_level = "Medium" # Default confidence
                                except json.JSONDecodeError:
                                    # String looked like JSON but failed to parse
                                    logger.warning(f"Field {field_key}: Failed to parse potential JSON value '{field_value}'. Using raw value.")
                                    extracted_value = field_value # Keep original string
                                    confidence_level = "Medium" # Default confidence
                            else:
                                # field_value is not a string or doesn't look like JSON
                                # Assume the value is the direct extraction result
                                extracted_value = field_value
                                # We didn't get confidence in the expected format, default to Medium
                                confidence_level = "Medium"
                                logger.info(f"Field {field_key}: Value is not the expected JSON format. Using raw value and Medium confidence.")

                            # Store the extracted value and confidence
                            processed_response[field_key] = extracted_value
                            processed_response[f"{field_key}_confidence"] = confidence_level

                        except Exception as e:
                            logger.error(f"Error processing field {field_key} with value '{field_value}': {str(e)}")
                            processed_response[field_key] = field_value # Store original value on error
                            processed_response[f"{field_key}_confidence"] = "Low" # Low confidence due to processing error
                else:
                    logger.warning(f"No 'metadata' field found in the structured API entry: {entry}")
                    processed_response["_error"] = "No 'metadata' field in API entry"
                    processed_response["_confidence_processing_failed"] = True
            
            # Original error case if neither 'answer' nor 'entries' is found
            else:
                logger.warning(f"Neither 'answer' nor 'entries' field found in the structured API response: {response_data}")
                processed_response["_error"] = "Neither 'answer' nor 'entries' field in API response"
                processed_response["_confidence_processing_failed"] = True
            # *** END OF REVISED LOGIC ***
            
            # Return the processed response
            return processed_response
        
        except Exception as e:
            logger.error(f"Error in structured metadata extraction call: {str(e)}")
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
            dict: Extracted metadata with confidence scores
        """
        try:
            # Get client from session state
            client = st.session_state.client
            
            # Get access token from client
            access_token = None
            if hasattr(client, "_oauth"):
                access_token = client._oauth.access_token
            elif hasattr(client, "auth") and hasattr(client.auth, "access_token"):
                access_token = client.auth.access_token
            
            if not access_token:
                raise ValueError("Could not retrieve access token from client")
            
            # Set headers
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
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
                    "system_message": "You are an AI assistant specialized in extracting metadata from documents. Extract the requested information and for EACH extracted field, provide both the value and your confidence level (High, Medium, or Low). Format your response as a JSON object where each field has a nested object containing \\\"value\\\" and \\\"confidence\\\" keys. Example: {\\\"InvoiceNumber\\\": {\\\"value\\\": \\\"INV-12345\\\", \\\"confidence\\\": \\\"High\\\"}, \\\"Date\\\": {\\\"value\\\": \\\"2023-04-15\\\", \\\"confidence\\\": \\\"Medium\\\"}}"
                },
                "basic_text": {
                    "model": ai_model,
                    "system_message": "You are an AI assistant specialized in extracting metadata from documents. Extract the requested information and for EACH extracted field, provide both the value and your confidence level (High, Medium, or Low). Format your response as a JSON object where each field has a nested object containing \\\"value\\\" and \\\"confidence\\\" keys. Example: {\\\"InvoiceNumber\\\": {\\\"value\\\": \\\"INV-12345\\\", \\\"confidence\\\": \\\"High\\\"}, \\\"Date\\\": {\\\"value\\\": \\\"2023-04-15\\\", \\\"confidence\\\": \\\"Medium\\\"}}"
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
            logger.info(f"Raw Box AI freeform extraction response data: {json.dumps(response_data)}") # Added logging
            
            # Process the response to extract confidence levels
            processed_response = {}
            
            # *** MODIFIED LOGIC TO HANDLE BOTH RESPONSE FORMATS ***
            # First check for 'answer' field (new format)
            if "answer" in response_data:
                if isinstance(response_data["answer"], str):
                    # Answer is a string, try to parse JSON from it
                    logger.info("Processing freeform 'answer' as string.")
                    response_text = response_data["answer"]
                    parsed_successfully = False
                    
                    try:
                        # Look for JSON-like content in the string
                        json_start = response_text.find('{')
                        json_end = response_text.rfind('}') + 1
                        
                        if json_start != -1 and json_end > json_start:
                            json_str = response_text[json_start:json_end]
                            parsed_json = json.loads(json_str)
                            
                            if isinstance(parsed_json, dict):
                                # Process each field in the parsed JSON
                                for field_key, field_data in parsed_json.items():
                                    if isinstance(field_data, dict) and "value" in field_data and "confidence" in field_data:
                                        # Extract value and confidence
                                        extracted_value = field_data["value"]
                                        confidence_level = field_data["confidence"]
                                        
                                        # Validate confidence
                                        if confidence_level not in ["High", "Medium", "Low"]:
                                            confidence_level = "Medium"
                                            
                                        # Store in processed response
                                        processed_response[field_key] = extracted_value
                                        processed_response[f"{field_key}_confidence"] = confidence_level
                                    else:
                                        # Use the field_data as is with Medium confidence
                                        processed_response[field_key] = field_data
                                        processed_response[f"{field_key}_confidence"] = "Medium"
                                parsed_successfully = True
                            else:
                                # Not a dictionary, store raw text
                                logger.warning(f"Parsed JSON from freeform 'answer' string is not a dictionary: {parsed_json}")
                        else:
                            # No JSON found, store raw text
                            logger.warning(f"No JSON object found in freeform 'answer' string.")
                            
                        if not parsed_successfully:
                            processed_response["_raw_response"] = response_text
                            processed_response["_confidence_processing_failed"] = True
                    except Exception as e:
                        logger.error(f"Error parsing JSON from freeform answer string: {str(e)}")
                        processed_response["_raw_response"] = response_text
                        processed_response["_confidence_processing_failed"] = True
                
                elif isinstance(response_data["answer"], dict):
                    # Answer is already a dictionary, process directly (assuming same structure as structured)
                    logger.info("Processing freeform 'answer' as dictionary.")
                    for field_key, field_data in response_data["answer"].items():
                        if isinstance(field_data, dict) and "value" in field_data and "confidence" in field_data:
                            # Extract value and confidence
                            extracted_value = field_data["value"]
                            confidence_level = field_data["confidence"]
                            
                            # Validate confidence
                            if confidence_level not in ["High", "Medium", "Low"]:
                                confidence_level = "Medium"
                                
                            # Store in processed response
                            processed_response[field_key] = extracted_value
                            processed_response[f"{field_key}_confidence"] = confidence_level
                        # Handle dict with only 'value'
                        elif isinstance(field_data, dict) and "value" in field_data and len(field_data) == 1:
                            extracted_value = field_data["value"]
                            confidence_level = "Medium"
                            processed_response[field_key] = extracted_value
                            processed_response[f"{field_key}_confidence"] = confidence_level
                        else:
                            # Use the field_data as is with Medium confidence
                            processed_response[field_key] = field_data
                            processed_response[f"{field_key}_confidence"] = "Medium"
                else:
                    logger.warning(f"Unexpected freeform 'answer' format: {response_data['answer']}")
                    processed_response["_error"] = "Unexpected freeform 'answer' format"
                    processed_response["_confidence_processing_failed"] = True
            
            # Fall back to entries format if answer is not present (for backward compatibility)
            elif "entries" in response_data and len(response_data["entries"]) > 0:
                logger.info("Processing freeform response using fallback 'entries' format.")
                # Get the first entry
                entry = response_data["entries"][0]
                
                # Check if we have a response field
                if "response" in entry:
                    response_text = entry["response"]
                    parsed_successfully = False

                    # Try to parse the response text as JSON
                    try:
                        # Attempt to find and parse JSON within the response text
                        json_start = response_text.find('{')
                        json_end = response_text.rfind('}') + 1

                        if json_start != -1 and json_end > json_start:
                            json_str = response_text[json_start:json_end]
                            parsed_json = json.loads(json_str)

                            if isinstance(parsed_json, dict):
                                # Successfully parsed a JSON object
                                for field_key, field_data in parsed_json.items():
                                    if isinstance(field_data, dict) and "value" in field_data and "confidence" in field_data:
                                        # Found the expected nested structure
                                        extracted_value = field_data["value"]
                                        confidence_level = field_data["confidence"]
                                        # Validate confidence
                                        if confidence_level not in ["High", "Medium", "Low"]:
                                            logger.warning(f"Field {field_key}: Unexpected confidence value '{confidence_level}', defaulting to Medium.")
                                            confidence_level = "Medium"

                                        processed_response[field_key] = extracted_value
                                        processed_response[f"{field_key}_confidence"] = confidence_level
                                    else:
                                        # Field data is not the expected nested dict
                                        logger.warning(f"Field {field_key}: Unexpected structure in parsed JSON: {field_data}. Storing as is with Medium confidence.")
                                        processed_response[field_key] = field_data # Store whatever was returned
                                        processed_response[f"{field_key}_confidence"] = "Medium"
                                parsed_successfully = True
                            else:
                                # Parsed JSON but it's not a dictionary (e.g., a list)
                                logger.warning(f"Parsed JSON from 'entries' response string is not a dictionary: {json_str}. Storing raw response.")
                        else:
                            # No JSON object found in the response text
                            logger.warning(f"No JSON object found in 'entries' response string. Storing raw response.")

                    except json.JSONDecodeError as e:
                        # Failed to parse the potential JSON string
                        logger.warning(f"Failed to parse JSON from 'entries' response string. Error: {e}. Storing raw response.")
                    except Exception as e:
                        # General error during parsing
                        logger.error(f"Error processing freeform 'entries' response. Error: {e}. Storing raw response.")

                    # If parsing failed or no JSON found, store the raw response text
                    if not parsed_successfully:
                        processed_response["_raw_response"] = response_text
                        processed_response["_confidence_processing_failed"] = True # Indicate failure
                else:
                    # No 'response' field in the entry
                    logger.warning(f"No 'response' field found in the freeform API entry: {entry}")
                    processed_response["_error"] = "No 'response' field in API entry"
                    processed_response["_confidence_processing_failed"] = True
            else:
                # Neither 'answer' nor 'entries' found
                logger.warning(f"Neither 'answer' nor 'entries' field found in the freeform API response: {response_data}")
                processed_response["_error"] = "Neither 'answer' nor 'entries' field in API response"
                processed_response["_confidence_processing_failed"] = True
            # *** END OF MODIFIED LOGIC ***
            
            # Return the processed response
            return processed_response
        
        except Exception as e:
            logger.error(f"Error in freeform metadata extraction call: {str(e)}")
            return {"error": str(e)}

    # Return dictionary of functions
    return {
        "extract_structured_metadata": extract_structured_metadata,
        "extract_freeform_metadata": extract_freeform_metadata
    }

# Example usage (for testing purposes)
if __name__ == "__main__":
    # This part would require a running Streamlit app context with authentication
    # and session state, so it's primarily for structural reference.
    
    # Mock Streamlit session state for local testing (if needed)
    # st.session_state.client = ... # Initialize with a mock or real client
    # st.session_state.authenticated = True
    
    extraction_funcs = metadata_extraction()
    
    # Example call (replace with actual file ID and config)
    # file_id = "YOUR_FILE_ID"
    # fields = [{"key": "InvoiceNumber", "displayName": "Invoice Number", "type": "string"}]
    # result = extraction_funcs["extract_structured_metadata"](file_id, fields=fields)
    # print(result)
    
    # prompt = "Extract the invoice number and total amount."
    # result_freeform = extraction_funcs["extract_freeform_metadata"](file_id, prompt=prompt)
    # print(result_freeform)
    pass
