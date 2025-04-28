"""
Centralized API client for Box API interactions.
This module provides a unified interface for all Box API operations,
handling authentication, request formatting, and error handling consistently.
"""

import requests
import time
import logging
import json
import threading
import random
from typing import Dict, Any, Optional, Union, List, Tuple

# Configure logging
logger = logging.getLogger(__name__)

class BoxAPIClient:
    """
    Centralized client for Box API interactions with consistent error handling,
    authentication management, request formatting, and optimization features.
    """
    
    def __init__(self, client):
        """
        Initialize the API client with a Box SDK client.
        
        Args:
            client: Box SDK client instance
        """
        self.client = client
        self._access_token = None
        self._token_lock = threading.RLock()
        self.session = requests.Session()
        
        # Configure connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,  # Number of connection pools
            pool_maxsize=100,     # Max connections per pool
            max_retries=0         # We'll handle retries ourselves
        )
        self.session.mount("https://", adapter)
        
        # Initialize metrics tracking
        self.metrics = {
            'requests': 0,
            'successes': 0,
            'failures': 0,
            'retries': 0,
            'total_time': 0,
            'endpoints': {}
        }
        self.metrics_lock = threading.RLock()
        
        # Initialize template schema cache
        self.template_cache = {}
        self.template_cache_lock = threading.RLock()
    
    def get_access_token(self) -> str:
        """
        Get the current access token, extracting it from the client if needed.
        
        Returns:
            str: Access token
        """
        with self._token_lock:
            if not self._access_token:
                if hasattr(self.client, '_oauth'):
                    self._access_token = self.client._oauth.access_token
                elif hasattr(self.client, 'auth') and hasattr(self.client.auth, 'access_token'):
                    self._access_token = self.client.auth.access_token
                else:
                    # Attempt to refresh if possible
                    try:
                        refreshed = self.client.auth.refresh(None)
                        if refreshed:
                            self._access_token = self.client.auth.access_token
                        else:
                             raise ValueError("Could not retrieve or refresh access token from Box client")
                    except Exception as e:
                         raise ValueError(f"Could not retrieve access token from Box client: {e}")
            
            return self._access_token
    
    def refresh_token(self) -> None:
        """
        Force a token refresh by clearing the cached token.
        The next call to get_access_token will extract a fresh token.
        """
        with self._token_lock:
            self._access_token = None
            # Optionally, trigger SDK refresh immediately
            try:
                if hasattr(self.client, 'auth') and hasattr(self.client.auth, 'refresh'):
                    self.client.auth.refresh(None)
                    logger.info("Box SDK token refreshed.")
            except Exception as e:
                logger.error(f"Error refreshing Box SDK token: {e}")

    
    def call_api(self, 
                endpoint: str, 
                method: str = "GET", 
                data: Optional[Dict[str, Any]] = None,
                params: Optional[Dict[str, Any]] = None,
                headers: Optional[Dict[str, str]] = None,
                files: Optional[Dict[str, Any]] = None,
                max_retries: int = 3,
                retry_codes: List[int] = [401, 429, 500, 502, 503, 504],
                timeout: int = 120) -> Dict[str, Any]: # Increased timeout for AI calls
        """
        Make an API call to the Box API with consistent error handling and retries.
        
        Args:
            endpoint: API endpoint (without base URL)
            method: HTTP method (GET, POST, PUT, DELETE)
            data: Request body data (will be JSON-encoded)
            params: Query parameters
            headers: Additional headers
            files: Files to upload
            max_retries: Maximum number of retry attempts
            retry_codes: HTTP status codes that should trigger a retry
            timeout: Request timeout in seconds
            
        Returns:
            dict: API response data
        """
        url = f"https://api.box.com/2.0/{endpoint.lstrip('/')}"
        
        # Prepare headers
        request_headers = {
            'Authorization': f'Bearer {self.get_access_token()}',
            'Content-Type': 'application/json'
        }
        if headers:
            request_headers.update(headers)
        
        # Start metrics tracking
        start_time = time.time()
        # Extract base endpoint for metrics (e.g., 'files', 'ai')
        endpoint_parts = endpoint.split('/')
        endpoint_key = endpoint_parts[0] if endpoint_parts else endpoint
        retries = 0
        
        try:
            while True:
                try:
                    # Make the request
                    if method.upper() in ['GET', 'DELETE']:
                        response = self.session.request(
                            method=method,
                            url=url,
                            headers=request_headers,
                            params=params,
                            timeout=timeout
                        )
                    else:
                        # For POST, PUT, PATCH
                        if files:
                            # Don't JSON-encode if sending files
                            json_data = None
                            # Remove Content-Type if sending files, let requests set it
                            if 'Content-Type' in request_headers and 'multipart/form-data' not in request_headers['Content-Type']:
                                del request_headers['Content-Type']
                        else:
                            json_data = data
                        
                        response = self.session.request(
                            method=method,
                            url=url,
                            headers=request_headers,
                            params=params,
                            json=json_data,
                            data=data if files else None, # Use data for form-encoded if needed, json otherwise
                            files=files,
                            timeout=timeout
                        )
                    
                    # Check for success
                    response.raise_for_status()
                    
                    # Parse response
                    if response.content:
                        try:
                            result = response.json()
                        except json.JSONDecodeError:
                            logger.error(f"Failed to decode JSON response from {method} {url}")
                            result = {"error": "Invalid JSON response", "status_code": response.status_code, "content": response.text}
                    else:
                        result = {"success": True, "status_code": response.status_code}
                    
                    # Update metrics for success
                    self._update_metrics(endpoint_key, True, time.time() - start_time, retries)
                    
                    return result
                
                except requests.exceptions.HTTPError as e:
                    status_code = e.response.status_code
                    
                    # Check if we should retry
                    if status_code in retry_codes and retries < max_retries:
                        retries += 1
                        
                        # Calculate backoff time with exponential backoff and jitter
                        backoff = min(1 * (2 ** retries), 60)  # Start with 1s, cap at 60s
                        jitter = random.uniform(0, 0.1 * backoff) # Add positive jitter up to 10%
                        sleep_time = backoff + jitter
                        
                        logger.warning(
                            f"API request {method} {url} failed with status {status_code}, "
                            f"retrying in {sleep_time:.2f}s (attempt {retries}/{max_retries})"
                        )
                        
                        # Check if token expired (401)
                        if status_code == 401:
                            logger.info("Access token may have expired, refreshing")
                            self.refresh_token()
                            # Update header for next retry
                            request_headers['Authorization'] = f'Bearer {self.get_access_token()}'
                        
                        time.sleep(sleep_time)
                        continue
                    
                    # No more retries or non-retryable status
                    logger.error(f"API request {method} {url} failed after {retries} retries: {str(e)}")
                    
                    # Try to parse error response
                    error_data = {"error": str(e), "status_code": status_code}
                    try:
                        error_json = e.response.json()
                        if isinstance(error_json, dict):
                            error_data.update(error_json)
                        else:
                            error_data["response_text"] = e.response.text
                    except json.JSONDecodeError:
                         error_data["response_text"] = e.response.text
                    
                    # Update metrics for failure
                    self._update_metrics(endpoint_key, False, time.time() - start_time, retries)
                    
                    return error_data
                
                except (requests.exceptions.ConnectionError, 
                        requests.exceptions.Timeout,
                        requests.exceptions.RequestException) as e:
                    # Network-related errors
                    if retries < max_retries:
                        retries += 1
                        
                        # Calculate backoff time
                        backoff = min(1 * (2 ** retries), 60)
                        jitter = random.uniform(0, 0.1 * backoff)
                        sleep_time = backoff + jitter
                        
                        logger.warning(
                            f"Network error on {method} {url}: {str(e)}, "
                            f"retrying in {sleep_time:.2f}s (attempt {retries}/{max_retries})"
                        )
                        
                        time.sleep(sleep_time)
                        continue
                    
                    # No more retries
                    logger.error(f"Network error on {method} {url} after {max_retries} retries: {str(e)}")
                    
                    # Update metrics for failure
                    self._update_metrics(endpoint_key, False, time.time() - start_time, retries)
                    
                    return {"error": str(e), "type": "network_error"}
        
        except Exception as e:
            # Unexpected errors
            logger.exception(f"Unexpected error in API call {method} {url}: {str(e)}")
            
            # Update metrics for failure
            self._update_metrics(endpoint_key, False, time.time() - start_time, retries)
            
            return {"error": str(e), "type": "unexpected_error"}
    
    def _update_metrics(self, endpoint: str, success: bool, duration: float, retries: int) -> None:
        """
        Update API metrics.
        
        Args:
            endpoint: API endpoint key (e.g., 'files', 'ai')
            success: Whether the call was successful
            duration: Call duration in seconds
            retries: Number of retries performed
        """
        with self.metrics_lock:
            # Update global metrics
            self.metrics['requests'] += 1
            self.metrics['total_time'] += duration
            
            if success:
                self.metrics['successes'] += 1
            else:
                self.metrics['failures'] += 1
            
            self.metrics['retries'] += retries
            
            # Update endpoint-specific metrics
            if endpoint not in self.metrics['endpoints']:
                self.metrics['endpoints'][endpoint] = {
                    'requests': 0,
                    'successes': 0,
                    'failures': 0,
                    'total_time': 0,
                    'min_time': float('inf'),
                    'max_time': 0
                }
            
            endpoint_metrics = self.metrics['endpoints'][endpoint]
            endpoint_metrics['requests'] += 1
            endpoint_metrics['total_time'] += duration
            
            if success:
                endpoint_metrics['successes'] += 1
            else:
                endpoint_metrics['failures'] += 1
            
            endpoint_metrics['min_time'] = min(endpoint_metrics['min_time'], duration)
            endpoint_metrics['max_time'] = max(endpoint_metrics['max_time'], duration)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current API metrics.
        
        Returns:
            dict: API metrics
        """
        with self.metrics_lock:
            # Create a copy of metrics with calculated values
            metrics_copy = {
                'requests': self.metrics['requests'],
                'successes': self.metrics['successes'],
                'failures': self.metrics['failures'],
                'retries': self.metrics['retries'],
                'total_time': self.metrics['total_time'],
                'avg_time': self.metrics['total_time'] / max(1, self.metrics['requests']),
                'success_rate': (self.metrics['successes'] / max(1, self.metrics['requests'])) * 100,
                'endpoints': {}
            }
            
            # Add endpoint-specific metrics with calculated values
            for endpoint, data in self.metrics['endpoints'].items():
                metrics_copy['endpoints'][endpoint] = {
                    'requests': data['requests'],
                    'successes': data['successes'],
                    'failures': data['failures'],
                    'total_time': data['total_time'],
                    'avg_time': data['total_time'] / max(1, data['requests']),
                    'min_time': data['min_time'] if data['min_time'] != float('inf') else 0,
                    'max_time': data['max_time'],
                    'success_rate': (data['successes'] / max(1, data['requests'])) * 100
                }
            
            return metrics_copy
    
    def reset_metrics(self) -> None:
        """Reset all API metrics."""
        with self.metrics_lock:
            self.metrics = {
                'requests': 0,
                'successes': 0,
                'failures': 0,
                'retries': 0,
                'total_time': 0,
                'endpoints': {}
            }
    
    # --- Convenience methods for common API operations --- 
    
    def get_file_info(self, file_id: str, fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get file information.
        
        Args:
            file_id: Box file ID
            fields: Specific fields to retrieve (or None for all)
            
        Returns:
            dict: File information
        """
        params = {}
        if fields:
            params['fields'] = ','.join(fields)
        
        return self.call_api(f"files/{file_id}", params=params)
    
    def get_folder_items(self, 
                        folder_id: str, 
                        limit: int = 100, 
                        offset: int = 0,
                        fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get items in a folder.
        
        Args:
            folder_id: Box folder ID
            limit: Maximum number of items to return
            offset: Pagination offset
            fields: Specific fields to retrieve (or None for all)
            
        Returns:
            dict: Folder items
        """
        params = {
            'limit': limit,
            'offset': offset
        }
        
        if fields:
            params['fields'] = ','.join(fields)
        
        return self.call_api(f"folders/{folder_id}/items", params=params)
    
    def get_metadata_templates(self, scope: str = "enterprise") -> Dict[str, Any]:
        """
        Get metadata templates.
        
        Args:
            scope: Template scope (enterprise or global)
            
        Returns:
            dict: Metadata templates
        """
        return self.call_api(f"metadata_templates/{scope}")
    
    def get_metadata_template_schema(self, scope: str, template_key: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get a specific metadata template schema, using cache if enabled.
        
        Args:
            scope: Template scope (enterprise or global)
            template_key: Template key
            use_cache: Whether to use the cache (default: True)
            
        Returns:
            dict: Metadata template schema
        """
        cache_key = f"{scope}_{template_key}"
        
        if use_cache:
            with self.template_cache_lock:
                if cache_key in self.template_cache:
                    logger.info(f"Using cached schema for template: {cache_key}")
                    return self.template_cache[cache_key]
        
        # Fetch from API if not in cache or cache disabled
        logger.info(f"Fetching schema for template from API: {cache_key}")
        schema = self.call_api(f"metadata_templates/{scope}/{template_key}/schema")
        
        # Store in cache if successful and caching enabled
        if use_cache and 'error' not in schema:
            with self.template_cache_lock:
                self.template_cache[cache_key] = schema
                
        return schema

    def clear_template_cache(self) -> None:
        """Clear the metadata template schema cache."""
        with self.template_cache_lock:
            self.template_cache = {}
            logger.info("Metadata template schema cache cleared.")

    def get_file_metadata(self, file_id: str, scope: str, template: str) -> Dict[str, Any]:
        """
        Get file metadata.
        
        Args:
            file_id: Box file ID
            scope: Metadata scope (enterprise or global)
            template: Template key
            
        Returns:
            dict: File metadata
        """
        return self.call_api(f"files/{file_id}/metadata/{scope}/{template}")
    
    def apply_metadata(self, 
                      file_id: str, 
                      metadata: Dict[str, Any], 
                      scope: str, 
                      template_key: str) -> Dict[str, Any]:
        """
        Apply metadata to a file (create or update).
        
        Args:
            file_id: Box file ID
            metadata: Metadata key-value pairs to apply
            scope: Metadata scope (enterprise or global)
            template_key: Template key
            
        Returns:
            dict: API response
        """
        endpoint = f"files/{file_id}/metadata/{scope}/{template_key}"
        
        # Try to create first
        create_response = self.call_api(endpoint, method="POST", data=metadata, retry_codes=[429, 500, 502, 503, 504]) # Don't retry on 409 Conflict
        
        if 'error' in create_response and create_response.get('status_code') == 409: # Conflict means metadata exists
            logger.info(f"Metadata already exists for file {file_id}, attempting update.")
            # Prepare update payload (RFC 6902 JSON Patch)
            update_payload = []
            for key, value in metadata.items():
                # Ensure value is string for Box API compatibility
                update_payload.append({"op": "add", "path": f"/{key}", "value": str(value)})
            
            # Use PUT for update with JSON Patch headers
            headers = {'Content-Type': 'application/json-patch+json'}
            update_response = self.call_api(endpoint, method="PUT", data=update_payload, headers=headers)
            return update_response
        else:
            # Return the create response (success or other error)
            return create_response

    # --- NEW Batch AI Extraction Methods --- 

    def batch_extract_metadata_freeform(self, 
                                       items: List[Dict[str, str]], 
                                       prompt: str, 
                                       ai_model: str = "google__gemini_2_0_flash_001") -> Dict[str, Any]:
        """
        Perform batch freeform metadata extraction using Box AI.
        
        Args:
            items: List of item dictionaries, e.g., [{'id': 'file_id_1', 'type': 'file'}, ...]
            prompt: The prompt for the AI model.
            ai_model: The AI model to use (default: google__gemini_2_0_flash_001).
            
        Returns:
            dict: API response containing batch results or error.
        """
        endpoint = "ai/extract"
        data = {
            "items": items,
            "prompt": prompt,
            "ai_agent": {
                "type": "ai_agent_extract",
                "basic_text": {
                    "model": ai_model
                },
                "long_text": {
                    "model": ai_model
                }
            }
        }
        logger.info(f"Calling batch freeform extraction for {len(items)} items.")
        return self.call_api(endpoint, method="POST", data=data)

    def batch_extract_metadata_structured(self, 
                                         items: List[Dict[str, str]], 
                                         template_scope: str, 
                                         template_key: str, 
                                         fields: List[Dict[str, Any]], # FIX: Added fields parameter
                                         ai_model: str = "google__gemini_2_0_flash_001") -> Dict[str, Any]:
        """
        Perform batch structured metadata extraction using Box AI.
        
        Args:
            items: List of item dictionaries, e.g., [{'id': 'file_id_1', 'type': 'file'}, ...]
            template_scope: The scope of the metadata template (e.g., 'enterprise_12345').
            template_key: The key of the metadata template.
            fields: List of field definitions from the template schema.
            ai_model: The AI model to use (default: google__gemini_2_0_flash_001).
            
        Returns:
            dict: API response containing batch results or error.
        """
        endpoint = "ai/extract_structured"
        data = {
            "items": items,
            "metadata_template": {
                "template_key": template_key,
                "scope": template_scope,
                "type": "metadata_template"
            },
            "fields": fields, # FIX: Include fields in the request payload
            "ai_agent": {
                "type": "ai_agent_extract", 
                "basic_text": {
                    "model": ai_model
                },
                "long_text": {
                    "model": ai_model
                }
            }
        }
        logger.info(f"Calling batch structured extraction for {len(items)} items using template {template_scope}.{template_key} with {len(fields)} fields.")
        return self.call_api(endpoint, method="POST", data=data)
