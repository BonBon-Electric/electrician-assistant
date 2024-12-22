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

def get_nec_code_info(query: str) -> dict:
    """Get relevant NEC code information based on the query"""
    
    # Define NEC code sections with detailed explanations
    nec_codes = {
        "210.12": {
            "title": "Arc-Fault Circuit-Interrupter Protection",
            "description": """Key requirements for AFCI protection:
            • Required for all 120V circuits in residential dwellings
            • Must protect bedrooms, living rooms, kitchens, family rooms
            • Protection required for branch circuits and feeders
            • AFCI devices must be readily accessible
            • Must be listed and marked as AFCI protection devices
            • Required for replacement receptacles in these areas
            • Must meet UL 1699 testing standards"""
        },
        "210.24": {
            "title": "Branch Circuit Requirements",
            "description": """Key requirements for branch circuits:
            • Circuit rating must match standard ampere rating
            • Conductor size must match overcurrent protection
            • Maximum load must not exceed circuit rating
            • Multi-wire circuits require common disconnect
            • Must maintain proper wire gauge for length
            • Voltage drop limited to 3% for branch circuits
            • Must provide separate neutral for AFCI circuits"""
        },
        "240.4": {
            "title": "Protection of Conductors",
            "description": """Key requirements for conductor protection:
            • Conductors must be protected at their ampacity
            • Small conductors require specific protection
            • Tap conductors have special sizing rules
            • Must account for temperature correction factors
            • Requires coordination with terminal temperature ratings
            • Must protect all ungrounded conductors
            • Cannot exceed maximum overcurrent protection size"""
        },
        "250.122": {
            "title": "Equipment Grounding Conductor Sizing",
            "description": """Key requirements for EGC sizing:
            • Must be sized based on overcurrent device rating
            • Minimum size specified in Table 250.122
            • Must be increased for voltage drop compensation
            • Parallel conductors require proper sizing
            • Must maintain proper termination methods
            • Cannot be smaller than specified minimums
            • Must account for conduit fill requirements"""
        },
        "300.5": {
            "title": "Underground Installation Requirements",
            "description": """Key requirements for underground installations:
            • Minimum burial depths specified by circuit type
            • Must protect cables from physical damage
            • Requires proper warning tape placement
            • Must maintain separation from other utilities
            • Requires proper raceway selection for soil
            • Must account for frost line depth
            • Requires proper drainage considerations"""
        },
        "310.15": {
            "title": "Ampacities for Conductors",
            "description": """Key requirements for conductor ampacity:
            • Must use proper temperature correction factors
            • Must account for ambient temperature
            • Requires adjustment for conductor bundling
            • Must consider raceway fill restrictions
            • Solar heating must be considered
            • Continuous loads require 125% sizing
            • Must use proper terminal temperature ratings"""
        },
        "314.16": {
            "title": "Box Fill Calculations",
            "description": """Key requirements for box fill:
            • Must count all conductors entering box
            • Device yokes count as double volume
            • Must include space for wire connectors
            • Support fittings require volume allowance
            • Equipment grounding conductors count as one
            • Must maintain proper wire bending space
            • Cannot exceed maximum fill percentage"""
        },
        "404.2": {
            "title": "Switch Locations",
            "description": """Key requirements for switch installation:
            • Must be readily accessible
            • Height restrictions for accessibility
            • Requires proper workspace clearances
            • Must be grounded if in metallic box
            • Wet locations require special enclosures
            • Must identify switched neutral when used
            • Requires proper wire bending space"""
        },
        "406.4": {
            "title": "Receptacle Requirements",
            "description": """Key requirements for receptacles:
            • Must be listed for the location
            • Proper orientation must be maintained
            • GFCI protection where required
            • Tamper-resistant in specified locations
            • Weather-resistant in wet locations
            • Must be properly secured to box
            • Requires proper grounding methods"""
        },
        "408.36": {
            "title": "Panelboard Overcurrent Protection",
            "description": """Key requirements for panelboard protection:
            • Main breaker required unless specific exceptions met
            • Must not exceed panelboard rating
            • Back-fed breakers require securing means
            • Must maintain proper wire bending space
            • Series-rated systems must be labeled
            • Must maintain proper working clearances
            • Requires proper grounding methods"""
        }
    }
    
    # Process the query to find relevant codes
    relevant_codes = {}
    query_lower = query.lower()
    
    for code, info in nec_codes.items():
        # Check if code number is directly mentioned
        if code in query:
            relevant_codes[code] = info
            continue
            
        # Check if title matches
        if info["title"].lower() in query_lower:
            relevant_codes[code] = info
            continue
            
        # Check for keyword matches in description
        keywords = query_lower.split()
        desc_lower = info["description"].lower()
        if any(keyword in desc_lower for keyword in keywords):
            relevant_codes[code] = info
    
    # Format the response
    if relevant_codes:
        response = "Here are the relevant NEC code sections:\n\n"
        for code, info in relevant_codes.items():
            response += f"**{code} {info['title']}**\n"
            response += "\n".join([f"- {line}" for line in info["description"].split("\n")]) + "\n\n"
    else:
        response = "No directly relevant NEC codes found for your query. Please try rephrasing or being more specific."
    
    return {
        "found": bool(relevant_codes),
        "codes": relevant_codes,
        "response": response
    }

def format_nec_response(response: str) -> str:
    """Format the NEC assistant response with detailed code explanations."""
    # Regular expression to find NEC code references in various formats
    nec_pattern = r'NEC (\d+\.\d+)(?:\s*\([A-Za-z]\))?'
    
    def format_nec_code(match):
        code = match.group(1)
        # Replace with detailed summaries based on the NEC code
        nec_info = get_nec_code_info(code)
        if nec_info["found"]:
            return nec_info["response"]
        else:
            return f"📖 NEC {code} - Refer to official NEC documentation for detailed requirements"
    
    # Replace NEC references with detailed summaries
    formatted_response = re.sub(nec_pattern, format_nec_code, response)
    
    return formatted_response

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
    
    # Format NEC references as detailed summaries
    response_with_summaries = format_nec_response(response)
    return response_with_summaries

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
    
    prompt = f"""As an experienced electrical contractor in Los Angeles County, create a detailed cost estimate for the following job:
    {full_description}
    
    Note: This is a {'detailed' if additional_info else 'general'} estimate based on the information provided.
    
    Please include:
    1. Labor costs (breakdown of hours and rates)
    2. Material costs (itemized list with prices)
    3. Los Angeles County permits and inspections required for this specific job
    4. Timeline for completion
    5. Any potential additional costs or variables
    
    Format the response in this exact JSON structure:
    {{
        "labor_costs": {{
            "description": "string",
            "total_hours": number,
            "rate_per_hour": number,
            "total": number
        }},
        "material_costs": {{
            "items": [
                {{
                    "item": "string",
                    "quantity": number,
                    "price": number,
                    "total": number
                }}
            ],
            "total": number
        }},
        "permit_fees": {{
            "permits_required": ["string"],
            "inspections_required": ["string"],
            "total": number
        }},
        "timeline": {{
            "duration": "string",
            "details": "string"
        }},
        "total_estimate": number,
        "additional_notes": "string"
    }}
    
    IMPORTANT: 
    - Return ONLY the JSON object, no additional text or formatting
    - For permit_fees, list SPECIFIC Los Angeles County permits and inspections required for this job
    - Include actual current LA County permit fee amounts"""
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean up the response to ensure it's valid JSON
        # Remove any markdown code block markers and extra whitespace
        response_text = re.sub(r'```json\s*|\s*```', '', response_text)
        response_text = re.sub(r'^\s+|\s+$', '', response_text)
        
        # Try to find the JSON object if there's extra text
        json_start = response_text.find('{')
        json_end = response_text.rfind('}')
        if json_start >= 0 and json_end >= 0:
            response_text = response_text[json_start:json_end + 1]
        
        # Parse the response into a dictionary
        try:
            estimate_data = json.loads(response_text)
            return {"status": "complete", "data": estimate_data}
        except json.JSONDecodeError as je:
            st.error(f"Failed to parse response: {str(je)}\nResponse text: {response_text}")
            return {"status": "error", "message": "Invalid response format from AI model"}
            
    except Exception as e:
        return {"status": "error", "message": f"Error generating estimate: {str(e)}"}

def display_cost_estimate(estimate_data):
    """Display the cost estimate in a formatted way"""
    if estimate_data["status"] == "error":
        st.error(estimate_data["message"])
        return
        
    data = estimate_data["data"]
    
    # Labor Costs
    if "labor_costs" in data:
        st.subheader("👷 Labor Costs")
        labor = data["labor_costs"]
        st.write(f"**Description:** {labor['description']}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Hours", f"{labor['total_hours']}")
        with col2:
            st.metric("Rate", f"${labor['rate_per_hour']}/hr")
        with col3:
            st.metric("Total Labor", f"${labor['total']:,.2f}")
    
    # Materials
    if "material_costs" in data:
        st.subheader("🔧 Materials")
        materials = data["material_costs"]
        
        # Create a DataFrame for better display
        if "items" in materials:
            items_data = []
            for item in materials["items"]:
                items_data.append({
                    "Item": item["item"],
                    "Quantity": item["quantity"],
                    "Price/Unit": f"${item['price']:,.2f}",
                    "Total": f"${item['total']:,.2f}"
                })
            
            if items_data:
                df = pd.DataFrame(items_data)
                st.dataframe(df, use_container_width=True)
                st.metric("Total Materials", f"${materials['total']:,.2f}")
    
    # Permit Fees
    if "permit_fees" in data:
        st.subheader("📋 Los Angeles County Permits & Inspections")
        permit = data["permit_fees"]
        if "permits_required" in permit:
            st.write("**Required Permits:**")
            for p in permit["permits_required"]:
                st.write(f"- {p}")
        if "inspections_required" in permit:
            st.write("**Required Inspections:**")
            for i in permit["inspections_required"]:
                st.write(f"- {i}")
        st.metric("Total Permit Fees", f"${permit['total']:,.2f}")
    
    # Timeline
    if "timeline" in data:
        st.subheader("⏱️ Timeline")
        timeline = data["timeline"]
        st.write(f"**Duration:** {timeline['duration']}")
        st.write(f"**Details:** {timeline['details']}")
    
    # Total Cost
    if "total_estimate" in data:
        st.markdown("---")
        st.subheader("💰 Total Estimate")
        st.metric("Total", f"${data['total_estimate']:,.2f}", help="Including labor, materials, permits, and overhead")
    
    # Additional Notes
    if "additional_notes" in data:
        st.markdown("---")
        st.subheader("📝 Additional Notes")
        st.info(data["additional_notes"])

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
            square_footage = st.number_input(
                "Square Footage",
                min_value=0,
                help="Enter the approximate square footage of the property"
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
        "Square Footage": square_footage if square_footage > 0 else "",
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
