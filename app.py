import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
import chromadb
import json
from typing import List, Dict, Any
import pandas as pd
from translations import TRANSLATIONS

# Load environment variables
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("Please set your Google API key in the .env file")
    GOOGLE_API_KEY = st.text_input("Enter your Google API key:", type="password")
    if GOOGLE_API_KEY:
        with open(".env", "w") as f:
            f.write(f"GOOGLE_API_KEY={GOOGLE_API_KEY}")
        st.success("API key saved! Please restart the application.")
        st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-pro')

# Initialize ChromaDB
chroma_client = chromadb.Client()
try:
    collection = chroma_client.get_collection(name="electrician_docs")
except:
    collection = chroma_client.create_collection(name="electrician_docs")

def get_text(key: str) -> str:
    """Get translated text based on current language"""
    return TRANSLATIONS[st.session_state.language][key]

def initialize_session_state():
    """Initialize session state variables"""
    if "language" not in st.session_state:
        st.session_state.language = "en"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_tab" not in st.session_state:
        st.session_state.current_tab = get_text('nec_assistant')
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'conversation_context' not in st.session_state:
        st.session_state.conversation_context = []
    if 'estimates_history' not in st.session_state:
        st.session_state.estimates_history = []

def update_chat_history(role: str, content: str):
    st.session_state.chat_history.append({"role": role, "content": content})
    # Keep last 10 messages for context
    if len(st.session_state.chat_history) > 10:
        st.session_state.chat_history = st.session_state.chat_history[-10:]

def get_gemini_response(prompt: str, context: str = "") -> str:
    """Get response from Gemini model"""
    try:
        structured_prompt = f"""As an expert electrician assistant, analyze the following question and provide a response in this specific format:

SUMMARY:
- Provide a brief summary of the key points and concerns in the question

RELEVANT NEC CODES:
- List each relevant NEC code number and title
- For each code:
  * Explain the specific requirements
  * Detail what needs to be done for compliance
  * Note any exceptions or special conditions

DETAILED RESPONSE:
- Provide a comprehensive answer addressing all aspects of the question
- Include practical advice and best practices
- Mention any safety considerations

Question: {prompt}
{context}"""
        
        response = model.generate_content(structured_prompt)
        return response.text
    except Exception as e:
        st.error(f"Error getting response from Gemini: {str(e)}")
        return ""

def is_greeting(text: str) -> bool:
    """Check if the input is a greeting or introduction"""
    greetings = ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'howdy']
    return any(text.lower().startswith(word) for word in greetings)

def is_electrical_question(text: str) -> bool:
    """Check if the input is likely an electrical-related question"""
    electrical_keywords = ['electrical', 'wire', 'circuit', 'voltage', 'amp', 'panel', 'breaker', 
                         'outlet', 'switch', 'power', 'install', 'repair', 'light', 'socket', 'nec']
    return any(keyword in text.lower() for keyword in electrical_keywords)

def format_nec_references(text: str) -> str:
    """
    Extract NEC code references from text and make them clickable.
    Matches patterns like 'NEC 210.12', 'Article 210.12', etc.
    """
    import re
    
    # Pattern to match NEC references
    nec_pattern = r'(?:NEC|Article)\s+(\d+\.?\d*(?:\([a-z]\))?(?:\(\d+\))?)'
    
    def replace_with_link(match):
        code = match.group(1)
        # Using NFPA's free access portal
        return f'[{match.group(0)}](https://www.nfpa.org/codes-and-standards/all-codes-and-standards/list-of-codes-and-standards/detail?code=70&section={code}) 📖'
    
    # Replace NEC references with clickable links
    text_with_links = re.sub(nec_pattern, replace_with_link, text)
    return text_with_links

def get_chat_response(prompt: str) -> str:
    """Get response from Gemini model with enhanced NEC reference handling"""
    context = ""
    
    # Check if it's an electrical code question
    if is_electrical_question(prompt):
        # Enhance the prompt to encourage NEC references
        prompt = f"""As an electrical code expert, please answer the following question, 
        citing specific NEC (National Electrical Code) articles and sections where applicable. 
        When referencing code sections, use the format 'NEC XXX.XX' or 'Article XXX.XX' 
        (for example: 'NEC 210.12' or 'Article 210.12').

        Question: {prompt}

        Please provide:
        1. A clear answer with specific NEC code references
        2. Brief explanation of why these code sections are relevant
        3. Any important safety considerations"""
    
    response = get_gemini_response(prompt, context)
    
    # Format NEC references as clickable links
    response_with_links = format_nec_references(response)
    return response_with_links

def display_chat_history():
    """Display chat history with clickable NEC references"""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                # Format NEC references in assistant's responses
                formatted_message = format_nec_references(message["content"])
                st.markdown(formatted_message, unsafe_allow_html=True)
            else:
                st.write(message["content"])

def create_cost_estimate(description: str):
    """Generate a cost estimate based on the job description"""
    # Initialize estimate conversation if not exists
    if 'estimate_conversation' not in st.session_state:
        st.session_state.estimate_conversation = []
        
    # Add user's description to conversation
    if not st.session_state.estimate_conversation:
        st.session_state.estimate_conversation.append({"role": "user", "content": description})
    
    # Prepare the prompt with required information checklist
    required_info = {
        'location': ['city', 'state', 'property type (residential/commercial)'],
        'electrical': ['voltage requirements', 'amperage', 'number of circuits'],
        'scope': ['timeline', 'accessibility', 'existing conditions'],
        'materials': ['specific equipment preferences', 'quality grade (standard/premium)']
    }
    
    # Check if we have all required information
    missing_info = []
    description_lower = description.lower()
    
    for category, items in required_info.items():
        for item in items:
            # Simple check for presence of keywords
            if not any(keyword in description_lower for keyword in item.lower().split()):
                missing_info.append(item)
    
    # If information is missing, generate follow-up questions
    if missing_info:
        prompt = f"""Based on the following job description: '{description}', 
        I notice some important details are missing. Please help me gather the following information:
        
        {', '.join(missing_info)}
        
        Please provide a friendly response asking for these details, explaining why they're important for an accurate estimate."""
        
        response = model.generate_content(prompt).text
        
        # Add the response to conversation
        st.session_state.estimate_conversation.append({"role": "assistant", "content": response})
        return {"status": "need_more_info", "message": response}
    
    # If we have all information, generate the estimate
    prompt = f"""As an experienced electrical contractor, create a detailed cost estimate for the following job:
    {description}
    
    Please include:
    1. Labor costs (breakdown of hours and rates)
    2. Material costs (itemized list with prices)
    3. Permit fees and inspections
    4. Overhead and profit
    5. Timeline for completion
    6. Any potential additional costs or variables
    
    Format the response as a JSON object with these categories."""
    
    try:
        response = model.generate_content(prompt).text
        # Parse the response into a dictionary
        estimate_data = json.loads(response)
        return {"status": "complete", "data": estimate_data}
    except Exception as e:
        return {"status": "error", "message": f"Error generating estimate: {str(e)}"}

def display_cost_estimate(estimate_data):
    """Display the cost estimate or follow-up questions"""
    if estimate_data["status"] == "need_more_info":
        st.write("📝 Additional Information Needed:")
        st.write(estimate_data["message"])
        
        # Display the conversation history
        for message in st.session_state.estimate_conversation:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
        # Add input for user's response
        if user_response := st.chat_input("Your response:"):
            st.session_state.estimate_conversation.append({"role": "user", "content": user_response})
            # Combine all user responses for a complete description
            full_description = " ".join([msg["content"] for msg in st.session_state.estimate_conversation if msg["role"] == "user"])
            new_estimate = create_cost_estimate(full_description)
            display_cost_estimate(new_estimate)
            
    elif estimate_data["status"] == "complete":
        st.success("✅ Cost Estimate Generated")
        
        # Display the estimate in a structured format
        data = estimate_data["data"]
        
        # Labor Costs
        st.subheader("👷 Labor Costs")
        if "labor_costs" in data:
            st.write(data["labor_costs"])
            
        # Material Costs
        st.subheader("🛠️ Materials")
        if "material_costs" in data:
            st.write(data["material_costs"])
            
        # Permits and Inspections
        st.subheader("📋 Permits and Inspections")
        if "permit_fees" in data:
            st.write(data["permit_fees"])
            
        # Timeline
        st.subheader("⏱️ Timeline")
        if "timeline" in data:
            st.write(data["timeline"])
            
        # Total Cost
        st.subheader("💰 Total Cost")
        if "total_cost" in data:
            st.metric("Estimated Total", f"${data['total_cost']:,.2f}")
            
        # Additional Notes
        if "additional_costs" in data:
            st.subheader("📝 Additional Notes")
            st.write(data["additional_costs"])
    else:
        st.error(estimate_data["message"])

def main():
    initialize_session_state()  # Initialize first!
    
    # Language selector in sidebar
    current_language = 'English' if st.session_state.language == 'en' else 'Español'
    selected_language = st.sidebar.selectbox(
        "🌐 Select Language / Seleccionar Idioma",
        options=['English', 'Español'],
        index=['English', 'Español'].index(current_language)
    )
    
    # Update language state if changed
    if (selected_language == 'English' and st.session_state.language != 'en') or \
       (selected_language == 'Español' and st.session_state.language != 'es'):
        st.session_state.language = 'en' if selected_language == 'English' else 'es'
        st.rerun()  # Rerun to update all translations
    
    # Now we can safely use get_text since language is initialized
    st.title(get_text('app_title'))
    
    # Create mode selector in the main content area
    st.session_state.current_tab = st.radio(
        "",  # Empty label for cleaner look
        [get_text('nec_assistant'), get_text('cost_estimator')],
        horizontal=True,  # Make the radio buttons horizontal
        label_visibility="collapsed"  # Hide the empty label
    )
    
    st.divider()  # Add a visual separator
    
    # Main content
    if st.session_state.current_tab == get_text('nec_assistant'):
        st.write(f"## {get_text('nec_assistant')}")
        display_chat_history()
        
        if prompt := st.chat_input(get_text('chat_placeholder')):
            update_chat_history("user", prompt)
            with st.chat_message("user"):
                st.write(prompt)
                
            with st.chat_message("assistant"):
                with st.spinner(get_text('thinking')):
                    response = get_chat_response(prompt)
                    st.write(response)
    
    else:  # Cost Estimator
        st.write(f"## {get_text('cost_estimator_title')}")
        st.write(get_text('cost_estimator_description'))
        
        job_description = st.text_area(
            get_text('job_description'),
            height=100,
            placeholder=get_text('job_description_placeholder')
        )
        
        if st.button(get_text('generate_estimate'), type="primary"):
            if job_description:
                with st.spinner(get_text('generating_estimate')):
                    estimate = create_cost_estimate(job_description)
                    display_cost_estimate(estimate)
            else:
                st.warning(get_text('provide_description'))

if __name__ == "__main__":
    main()
