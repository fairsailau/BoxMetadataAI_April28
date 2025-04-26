import streamlit as st
from boxsdk import OAuth2, Client, JWTAuth
import os
import json
import webbrowser
from urllib.parse import parse_qs, urlparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def authenticate():
    """
    Handle Box authentication using OAuth2 or JWT
    """
    st.title("Box Authentication")
    
    # Check if already authenticated
    if st.session_state.authenticated and st.session_state.client:
        st.success(f"You are already authenticated as {st.session_state.user.name}!")
        return
    
    st.write("""
    ## Connect to Box
    
    This app requires authentication with Box to access your files and use the Box AI API.
    You can authenticate using OAuth2 (for user-specific access) or JWT (for enterprise-wide access).
    """)
    
    # Authentication method selection
    auth_method = st.radio(
        "Select authentication method:",
        options=["OAuth 2.0", "JWT", "Developer Token (Testing Only)"],
        index=0,
        help="OAuth 2.0 is recommended for user-specific access. JWT is for enterprise-wide access."
    )
    
    if auth_method == "OAuth 2.0":
        oauth2_authentication()
    elif auth_method == "JWT":
        jwt_authentication()
    else:
        developer_token_authentication()

def oauth2_authentication():
    """
    Implement OAuth 2.0 authentication flow
    """
    with st.form("oauth2_form"):
        st.subheader("OAuth 2.0 Authentication")
        
        client_id = st.text_input("Client ID", type="password")
        client_secret = st.text_input("Client Secret", type="password")
        redirect_uri = st.text_input("Redirect URI", value="http://localhost:8501/")
        
        submitted = st.form_submit_button("Authenticate")
        
        if submitted:
            if not client_id or not client_secret:
                st.error("Please provide both Client ID and Client Secret")
            else:
                try:
                    # Initialize OAuth2 object
                    oauth = OAuth2(
                        client_id=client_id,
                        client_secret=client_secret,
                        store_tokens=store_tokens,
                    )
                    
                    # Store credentials for re-authentication
                    if "auth_credentials" not in st.session_state:
                        st.session_state.auth_credentials = {}
                    st.session_state.auth_credentials["client_id"] = client_id
                    st.session_state.auth_credentials["client_secret"] = client_secret
                    
                    # Get authorization URL
                    auth_url, csrf_token = oauth.get_authorization_url(redirect_uri)
                    
                    # Store CSRF token in session state
                    st.session_state.csrf_token = csrf_token
                    st.session_state.oauth = oauth
                    
                    # Display authorization URL
                    st.write("Please authorize the app by clicking the link below:")
                    st.markdown(f"[Authorize App]({auth_url})")
                    
                    # Open browser automatically
                    if st.button("Open in Browser"):
                        webbrowser.open(auth_url)
                    
                    # Input field for authorization code
                    st.write("After authorization, you'll be redirected to your redirect URI. Copy the full URL and paste it below:")
                    auth_code_url = st.text_input("Redirect URL")
                    
                    if auth_code_url:
                        try:
                            # Parse the URL to get the authorization code
                            parsed_url = urlparse(auth_code_url)
                            query_params = parse_qs(parsed_url.query)
                            
                            if 'code' in query_params:
                                auth_code = query_params['code'][0]
                                
                                # Exchange authorization code for access token
                                access_token, refresh_token = oauth.authenticate(auth_code)
                                
                                # Create client
                                client = Client(oauth)
                                
                                # Test the connection by getting current user info
                                current_user = client.user().get()
                                
                                # Store in session state
                                st.session_state.authenticated = True
                                st.session_state.client = client
                                st.session_state.user = current_user
                                
                                # Log authentication success
                                logger.info(f"Successfully authenticated as {current_user.name}")
                                
                                st.success(f"Successfully authenticated as {current_user.name}!")
                                st.rerun()
                            else:
                                st.error("Could not find authorization code in the URL")
                        except Exception as e:
                            st.error(f"Error processing authorization: {str(e)}")
                            logger.error(f"Error processing authorization: {str(e)}")
                
                except Exception as e:
                    st.error(f"Authentication initialization failed: {str(e)}")
                    logger.error(f"Authentication initialization failed: {str(e)}")

def jwt_authentication():
    """
    Implement JWT authentication flow
    """
    with st.form("jwt_form"):
        st.subheader("JWT Authentication")
        
        st.write("""
        JWT authentication requires a config.json file with your enterprise ID and private key.
        You can upload the file or paste the contents directly.
        """)
        
        # Option to upload config file or paste JSON
        upload_method = st.radio(
            "Config method:",
            options=["Upload config.json", "Paste JSON content"],
            index=0
        )
        
        config_json = None
        
        if upload_method == "Upload config.json":
            uploaded_file = st.file_uploader("Upload config.json", type=["json"])
            if uploaded_file:
                try:
                    config_json = json.load(uploaded_file)
                except Exception as e:
                    st.error(f"Error parsing JSON file: {str(e)}")
                    logger.error(f"Error parsing JSON file: {str(e)}")
        else:
            json_content = st.text_area("Paste JSON content", height=200)
            if json_content:
                try:
                    config_json = json.loads(json_content)
                except Exception as e:
                    st.error(f"Error parsing JSON content: {str(e)}")
                    logger.error(f"Error parsing JSON content: {str(e)}")
        
        submitted = st.form_submit_button("Authenticate")
        
        if submitted and config_json:
            try:
                # Initialize JWT auth
                auth = JWTAuth.from_settings_dictionary(config_json)
                
                # Authenticate
                auth.authenticate_instance()
                
                # Create client
                client = Client(auth)
                
                # Test the connection by getting service account info
                service_account = client.user().get()
                
                # Store in session state
                st.session_state.authenticated = True
                st.session_state.client = client
                st.session_state.user = service_account
                
                # Store JWT config for re-authentication
                if "auth_credentials" not in st.session_state:
                    st.session_state.auth_credentials = {}
                st.session_state.auth_credentials["jwt_config"] = config_json
                
                # Log authentication success
                logger.info(f"Successfully authenticated as {service_account.name} (Service Account)")
                
                st.success(f"Successfully authenticated as {service_account.name} (Service Account)!")
                st.rerun()
            
            except Exception as e:
                st.error(f"JWT Authentication failed: {str(e)}")
                logger.error(f"JWT Authentication failed: {str(e)}")

def developer_token_authentication():
    """
    Implement developer token authentication (for testing only)
    """
    with st.form("dev_token_form"):
        st.subheader("Developer Token Authentication")
        
        st.warning("Developer tokens expire after 60 minutes and are for testing only.")
        
        client_id = st.text_input("Client ID", type="password")
        client_secret = st.text_input("Client Secret", type="password")
        developer_token = st.text_input("Developer Token", type="password")
        
        submitted = st.form_submit_button("Authenticate")
        
        if submitted:
            if not client_id or not client_secret or not developer_token:
                st.error("Please provide Client ID, Client Secret, and Developer Token")
            else:
                try:
                    # Initialize OAuth2 with developer token
                    auth = OAuth2(
                        client_id=client_id,
                        client_secret=client_secret,
                        access_token=developer_token,
                        store_tokens=store_tokens  # Added store_tokens parameter
                    )
                    
                    # Store credentials for re-authentication
                    if "auth_credentials" not in st.session_state:
                        st.session_state.auth_credentials = {}
                    st.session_state.auth_credentials["client_id"] = client_id
                    st.session_state.auth_credentials["client_secret"] = client_secret
                    st.session_state.auth_credentials["access_token"] = developer_token
                    
                    # Create client
                    client = Client(auth)
                    
                    # Test the connection by getting current user info
                    current_user = client.user().get()
                    
                    # Store in session state
                    st.session_state.authenticated = True
                    st.session_state.client = client
                    st.session_state.user = current_user
                    
                    # Log authentication success
                    logger.info(f"Successfully authenticated as {current_user.name}")
                    
                    st.success(f"Successfully authenticated as {current_user.name}!")
                    st.rerun()
                
                except Exception as e:
                    st.error(f"Authentication failed: {str(e)}")
                    logger.error(f"Authentication failed: {str(e)}")

def store_tokens(access_token, refresh_token=None):
    """
    Store tokens in session state and return them as required by Box SDK
    """
    logger.info("Storing authentication tokens in session state")
    
    # Store in session state
    st.session_state.access_token = access_token
    if refresh_token:
        st.session_state.refresh_token = refresh_token
    
    # Store in auth_credentials for re-authentication
    if "auth_credentials" not in st.session_state:
        st.session_state.auth_credentials = {}
    
    # Make sure we have client_id and client_secret from the current session
    if hasattr(st.session_state, 'oauth') and hasattr(st.session_state.oauth, '_client_id'):
        st.session_state.auth_credentials["client_id"] = st.session_state.oauth._client_id
        st.session_state.auth_credentials["client_secret"] = st.session_state.oauth._client_secret
        logger.info("Captured client_id and client_secret from OAuth object")
    
    st.session_state.auth_credentials["access_token"] = access_token
    if refresh_token:
        st.session_state.auth_credentials["refresh_token"] = refresh_token
    
    # Log what we've stored (without revealing sensitive data)
    logger.info(f"Auth credentials keys stored: {list(st.session_state.auth_credentials.keys())}")
    
    # Must return tokens for Box SDK
    return access_token, refresh_token

# Instructions for creating a Box app
with st.expander("How to create a Box app and get credentials"):
    st.write("""
    ### Creating a Box App
    
    1. Go to the [Box Developer Console](https://app.box.com/developers/console)
    2. Click on "Create New App"
    3. Select "Custom App" and click "Next"
    4. Select "Standard OAuth 2.0" or "Server Authentication (with JWT)" and click "Next"
    5. Name your app and click "Create App"
    6. In the app configuration:
       - Under "OAuth 2.0 Redirect URI", add: `http://localhost:8501/`
       - Under "Application Scopes", select the necessary permissions:
         - Read and write all files and folders
         - Manage metadata
         - Use Box AI
    7. Save changes
    8. Note your "Client ID" and "Client Secret" for use in this app
    
    ### Getting a Developer Token (for testing)
    
    1. In your app configuration, click on "Developer Token"
    2. Generate a developer token
    3. Copy the token and use it in the form above
    
    Note: Developer tokens expire after 60 minutes and are for testing only.
    
    ### Setting up JWT Authentication
    
    1. In your app configuration, select "Server Authentication (with JWT)"
    2. Under "App Access Level", select "App + Enterprise Access"
    3. Under "Advanced Features", enable "Generate User Access Tokens"
    4. Save changes
    5. Under "Configuration", click "Generate a Public/Private Keypair"
    6. This will download a config.json file with your credentials
    7. Use this file for JWT authentication
    """)

# Add debug functionality
if st.sidebar.checkbox("Debug Authentication"):
    st.sidebar.write("### Authentication Debug")
    
    st.sidebar.write("**Session State Keys:**")
    st.sidebar.write(list(st.session_state.keys()))
    
    if "authenticated" in st.session_state:
        st.sidebar.write(f"**Authenticated:** {st.session_state.authenticated}")
    
    if "client" in st.session_state:
        st.sidebar.write("**Client:** Available")
    else:
        st.sidebar.write("**Client:** Not available")
    
    if "auth_credentials" in st.session_state:
        st.sidebar.write("**Auth Credentials Keys:**")
        st.sidebar.write(list(st.session_state.auth_credentials.keys()))
    else:
        st.sidebar.write("**Auth Credentials:** Not available")
