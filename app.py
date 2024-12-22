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
import re

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

def get_nec_assistant_prompt(query: str) -> str:
    """Generate the prompt for the NEC Electrical Assistant"""
    return f"""You are an expert electrical contractor and NEC code specialist based in Los Angeles County. Answer the following question about electrical work:

{query}

Provide your response in this format:

[No title - start with the practical explanation]
Give a brief, practical explanation of how to accomplish the task. Focus on the key steps and important safety considerations. This should be 2-3 sentences that give a high-level overview of the process.

RELEVANT NEC CODES:
List and explain the relevant NEC codes that apply to this work. For each code, include:
- Code number and title
- Brief explanation of the requirement
- Any specific compliance details

PERMITS AND INSPECTIONS (Los Angeles County):
1. List all required permits from LA County Building & Safety
2. Specific forms or documentation needed
3. Inspection requirements and process
4. Typical timeline and costs
5. Any special LA County requirements or restrictions

TECHNICAL DETAILS:
1. Safety Requirements:
   - Required PPE
   - Safety procedures
   - Lock-out/tag-out requirements

2. Tools and Materials:
   - List of required tools
   - Required materials and specifications
   - Recommended brands or types

3. Step-by-Step Process:
   - Detailed installation/repair steps
   - Critical measurements and specifications
   - Testing procedures
   - Quality control checks

4. Best Practices:
   - Industry-standard techniques
   - Common mistakes to avoid
   - Tips from experienced professionals
   - Future maintenance considerations

Ensure all technical information is accurate and code-compliant."""

def format_nec_response(response: str) -> str:
    """Format the NEC assistant response with proper styling and links"""
    
    # Remove section titles while keeping the content
    response = re.sub(r'^SUMMARY:\s*', '', response, flags=re.MULTILINE)
    response = re.sub(r'^DETAILED RESPONSE:\s*', '', response, flags=re.MULTILINE)
    
    # Format NEC code references as links with proper styling
    def replace_nec_reference(match):
        code = match.group(1)
        return f'<a href="https://www.nfpa.org/codes-and-standards/all-codes-and-standards/list-of-codes-and-standards/detail?code=70&section={code}" target="_blank" style="text-decoration: none; color: #0366d6;">NEC {code} 📖</a>'
    
    # Update the pattern to match various NEC reference formats
    response = re.sub(r'NEC (\d+\.\d+(?:\([a-z]\))?(?:\(\d+\))?)', replace_nec_reference, response)
    
    # Add subtle separator before each main section
    response = re.sub(r'^RELEVANT NEC CODES:', '\n---\n\n📚 RELEVANT NEC CODES:', response, flags=re.MULTILINE)
    response = re.sub(r'^PERMITS AND INSPECTIONS', '\n---\n\n📋 PERMITS AND INSPECTIONS', response, flags=re.MULTILINE)
    response = re.sub(r'^TECHNICAL DETAILS:', '\n---\n\n🔧 TECHNICAL DETAILS:', response, flags=re.MULTILINE)
    
    return response

def get_gemini_response(prompt: str, context: str = "") -> str:
    """Get response from Gemini model"""
    try:
        response = model.generate_content(prompt)
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

def get_chat_response(prompt: str) -> str:
    """Get response from Gemini model with enhanced NEC reference handling"""
    context = ""
    
    # Check if it's an electrical code question
    if is_electrical_question(prompt):
        # Enhance the prompt to encourage NEC references
        prompt = get_nec_assistant_prompt(prompt)
    
    response = get_gemini_response(prompt, context)
    
    # Format NEC references as clickable links
    response_with_links = format_nec_response(response)
    return response_with_links

def display_chat_history():
    """Display chat history with clickable NEC references"""
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                # Format NEC references in assistant's responses
                formatted_message = format_nec_response(message["content"])
                st.markdown(formatted_message, unsafe_allow_html=True)
            else:
                st.write(message["content"])

def create_cost_estimate(description: str, additional_info: dict = None):
    """Generate a cost estimate based on the job description and optional additional info"""
    if additional_info is None:
        additional_info = {}
    
    # Combine description with any additional info provided
    full_description = description + "\n\n"
    if additional_info:
        full_description += "Additional Details:\n"
        for key, value in additional_info.items():
            if value:  # Only add if value is provided
                full_description += f"- {key}: {value}\n"
    
    prompt = f"""As an experienced electrical contractor, create a detailed cost estimate for the following job:
    {full_description}
    
    Note: This is a {'detailed' if additional_info else 'general'} estimate based on the information provided.
    
    Please include:
    1. Labor costs (breakdown of hours and rates)
    2. Material costs (itemized list with prices)
    3. Permit fees and inspections
    4. Overhead and profit
    5. Timeline for completion
    6. Any potential additional costs or variables
    7. Confidence level of the estimate based on information provided
    
    Format the response as a JSON object with these categories."""
    
    try:
        response = model.generate_content(prompt).text
        # Parse the response into a dictionary
        estimate_data = json.loads(response)
        return {"status": "complete", "data": estimate_data}
    except Exception as e:
        return {"status": "error", "message": f"Error generating estimate: {str(e)}"}

def display_cost_estimate_form():
    """Display the cost estimate form with optional fields"""
    st.write(f"## {get_text('cost_estimator_title')}")
    st.write(get_text('cost_estimator_description'))
    
    # Basic job description
    job_description = st.text_area(
        get_text('job_description'),
        height=100,
        placeholder=get_text('job_description_placeholder')
    )
    
    # Optional details section
    with st.expander("📝 Additional Details (Optional)", expanded=False):
        st.info("These details will help provide a more accurate estimate, but all fields are optional.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Location details
            st.subheader("📍 Location")
            location_type = st.selectbox(
                "Property Type",
                options=["", "Residential", "Commercial", "Industrial"],
                index=0
            )
            city = st.text_input("City")
            state = st.selectbox(
                "State",
                options=["", "CA", "Other"],
                index=0
            )
            
            # Electrical specifications
            st.subheader("⚡ Electrical Specs")
            voltage = st.selectbox(
                "Voltage Requirements",
                options=["", "120V", "240V", "208V", "480V"],
                index=0
            )
            amperage = st.selectbox(
                "Amperage",
                options=["", "15A", "20A", "30A", "50A", "100A", "200A", "400A"],
                index=0
            )
            
        with col2:
            # Project scope
            st.subheader("🔨 Project Scope")
            timeline_urgency = st.select_slider(
                "Timeline Urgency",
                options=["", "Flexible", "Normal", "Urgent"],
                value=""
            )
            accessibility = st.select_slider(
                "Job Site Accessibility",
                options=["", "Easy", "Moderate", "Difficult"],
                value=""
            )
            
            # Material preferences
            st.subheader("🛠️ Materials")
            quality_grade = st.select_slider(
                "Material Quality Grade",
                options=["", "Standard", "Premium", "Luxury"],
                value=""
            )
            
    # Collect all the additional information
    additional_info = {
        "Property Type": location_type,
        "City": city,
        "State": state,
        "Voltage": voltage,
        "Amperage": amperage,
        "Timeline": timeline_urgency,
        "Accessibility": accessibility,
        "Material Grade": quality_grade
    }
    
    # Remove empty values
    additional_info = {k: v for k, v in additional_info.items() if v}
    
    if st.button(get_text('generate_estimate'), type="primary"):
        if job_description:
            with st.spinner(get_text('generating_estimate')):
                estimate = create_cost_estimate(job_description, additional_info)
                display_cost_estimate(estimate)
        else:
            st.warning(get_text('provide_description'))

def display_cost_estimate(estimate_data):
    """Display the cost estimate results"""
    if estimate_data["status"] == "complete":
        st.success("✅ Cost Estimate Generated")
        
        data = estimate_data["data"]
        
        # Confidence Level (if provided)
        if "confidence_level" in data:
            confidence_color = {
                "high": "green",
                "medium": "orange",
                "low": "red"
            }.get(data.get("confidence_level", "").lower(), "grey")
            st.markdown(f"*Confidence Level: <span style='color:{confidence_color}'>{data['confidence_level']}</span>*", unsafe_allow_html=True)
        
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
            
        # Disclaimer for generic estimates
        if not any(k for k in estimate_data.get("data", {}).keys() if k.startswith("user_")):
            st.info("Note: This is a general estimate based on limited information. Actual costs may vary significantly. Contact us for a more accurate quote.")
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
        display_cost_estimate_form()

if __name__ == "__main__":
    main()
