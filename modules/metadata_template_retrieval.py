import streamlit as st
import logging
import requests
import time
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_metadata_templates(client, force_refresh=False):
    """
    Retrieve metadata templates from Box
    
    Args:
        client: Box client
        force_refresh: Force refresh of templates
        
    Returns:
        dict: Metadata templates
    """
    # Check if templates are already cached
    if not force_refresh and hasattr(st.session_state, "metadata_templates") and st.session_state.metadata_templates:
        logger.info(f"Using cached metadata templates: {len(st.session_state.metadata_templates)} templates")
        return st.session_state.metadata_templates
    
    try:
        # Get access token from client
        access_token = None
        if hasattr(client, '_oauth'):
            access_token = client._oauth.access_token
        elif hasattr(client, 'auth') and hasattr(client.auth, 'access_token'):
            access_token = client.auth.access_token
        
        if not access_token:
            raise ValueError("Could not retrieve access token from client")
        
        # Get metadata templates using direct API calls
        templates = {}
        
        # Retrieve enterprise templates
        enterprise_templates = retrieve_templates_by_scope(access_token, "enterprise")
        
        # Process enterprise templates
        for template in enterprise_templates:
            if "templateKey" in template and "scope" in template:
                template_key = template["templateKey"]
                scope = template["scope"]
                template_id = f"{scope}_{template_key}"
                
                # Store template
                templates[template_id] = {
                    "id": template_id,
                    "key": template_key,
                    "displayName": template.get("displayName", template_key),
                    "fields": template.get("fields", []),
                    "hidden": template.get("hidden", False)
                }
        
        # Cache templates
        st.session_state.metadata_templates = templates
        st.session_state.template_cache_timestamp = time.time()
        
        logger.info(f"Retrieved {len(templates)} metadata templates")
        return templates
    
    except Exception as e:
        logger.error(f"Error retrieving metadata templates: {str(e)}")
        st.session_state.metadata_templates = {}
        return {}

def retrieve_templates_by_scope(access_token, scope):
    """
    Retrieve metadata templates for a specific scope using direct API call
    
    Args:
        access_token: Box API access token
        scope: Template scope (enterprise or global)
        
    Returns:
        list: List of metadata templates for the specified scope
    """
    templates = []
    next_marker = None
    
    try:
        # Make API calls until all templates are retrieved
        while True:
            # Construct API URL
            api_url = f"https://api.box.com/2.0/metadata_templates/{scope}"
            if next_marker:
                api_url += f"?marker={next_marker}"
            
            # Set headers
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Content-Type': 'application/json'
            }
            
            # Make API call
            response = requests.get(api_url, headers=headers)
            
            # Check for errors
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Add templates to list
            if 'entries' in data:
                templates.extend(data['entries'])
            
            # Check for next marker
            if 'next_marker' in data and data['next_marker']:
                next_marker = data['next_marker']
            else:
                break
        
        return templates
    
    except Exception as e:
        logger.error(f"Error retrieving {scope} templates: {str(e)}")
        return []

def initialize_template_state():
    """
    Initialize template-related session state variables
    """
    # Template cache
    if not hasattr(st.session_state, "metadata_templates"):
        st.session_state.metadata_templates = {}
        logger.info("Initialized metadata_templates in session state")
    
    # Template cache timestamp
    if not hasattr(st.session_state, "template_cache_timestamp"):
        st.session_state.template_cache_timestamp = None
        logger.info("Initialized template_cache_timestamp in session state")
    
    # Document type to template mapping
    if not hasattr(st.session_state, "document_type_to_template"):
        st.session_state.document_type_to_template = {
            "Sales Contract": None,
            "Invoices": None,
            "Tax": None,
            "Financial Report": None,
            "Employment Contract": None,
            "PII": None,
            "Other": None
        }
        logger.info("Initialized document_type_to_template in session state")

def get_template_by_id(template_id):
    """
    Get template by ID
    
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

def get_template_by_document_type(document_type):
    """
    Get template by document type
    
    Args:
        document_type: Document type
        
    Returns:
        dict: Template or None if not found
    """
    if not document_type:
        return None
    
    if not hasattr(st.session_state, "document_type_to_template"):
        return None
    
    template_id = st.session_state.document_type_to_template.get(document_type)
    if not template_id:
        return None
    
    return get_template_by_id(template_id)

def map_document_type_to_template(document_type, template_id):
    """
    Map document type to template
    
    Args:
        document_type: Document type
        template_id: Template ID
    """
    if not hasattr(st.session_state, "document_type_to_template"):
        st.session_state.document_type_to_template = {}
    
    st.session_state.document_type_to_template[document_type] = template_id
    logger.info(f"Mapped document type '{document_type}' to template '{template_id}'")
