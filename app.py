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
        "110.26": {
            "title": "Spaces About Electrical Equipment",
            "description": "NEC Code 110.26 establishes mandatory workspace requirements for electrical equipment. For installations operating at 0-150V, a minimum working space depth of 30 inches is required, measured from the exposed live parts or enclosure front. The workspace width must be the greater of 30 inches or the width of the equipment. A minimum headroom clearance of 6.5 feet must be maintained. The workspace must remain clear and unobstructed at all times, with adequate illumination provided. For equipment rated 1200A or more, at least one entrance of sufficient area is required for personnel to escape quickly in an emergency. No storage is permitted within the designated electrical working space. These requirements ensure safe access for installation, operation, and maintenance of electrical equipment."
        },
        "210.7": {
            "title": "Branch Circuits - General Requirements",
            "description": "NEC Code 210.7 governs overcurrent protection requirements for branch circuits. All ungrounded conductors must be protected by an overcurrent device at the point where the conductor receives its supply. The overcurrent protection rating must not exceed the conductor's ampacity after adjustment for ambient temperature and number of conductors. The device must coordinate with downstream protective devices to maintain selective coordination. All overcurrent devices must be readily accessible and properly sized to handle continuous loads at 125% of the continuous load current. Temperature correction factors must be applied when ambient temperatures differ from 30°C (86°F). This code ensures proper circuit protection and helps prevent electrical fires and equipment damage."
        },
        "210.12": {
            "title": "Arc-Fault Circuit-Interrupter Protection",
            "description": "NEC Code 210.12 mandates AFCI protection requirements for dwelling units. All 120-volt, single-phase, 15- and 20-ampere branch circuits supplying outlets or devices in dwelling unit kitchens, family rooms, dining rooms, living rooms, parlors, libraries, dens, bedrooms, sunrooms, recreation rooms, closets, hallways, laundry areas, and similar rooms or areas must have AFCI protection. The protection must be provided by either a branch-circuit type AFCI, a combination-type AFCI, or a listed outlet branch-circuit type AFCI. The AFCI device must be readily accessible and meet UL 1699 testing standards. When replacing receptacles in areas requiring AFCI protection, the replacement must provide or maintain AFCI protection. This code is crucial for preventing electrical fires caused by arcing faults."
        },
        "210.24": {
            "title": "Branch Circuit Requirements",
            "description": "NEC Code 210.24 specifies comprehensive requirements for branch circuits. The circuit rating must align with standard ampere ratings (15, 20, 30, 40, and 50 amperes), and conductor sizes must be properly matched to their overcurrent protection. The maximum load must not exceed 80% of the circuit rating for continuous loads. Multi-wire branch circuits require simultaneous disconnection of all ungrounded conductors. Wire gauge selection must account for voltage drop, limited to 3% for branch circuits and 5% for the total voltage drop including feeder and branch circuit. Each AFCI-protected circuit requires a dedicated neutral conductor. This code ensures safe and efficient branch circuit installations while preventing overloading and excessive voltage drop."
        },
        "240.4": {
            "title": "Protection of Conductors",
            "description": "NEC Code 240.4 establishes requirements for conductor protection against overcurrent. Conductors must be protected at their ampacity rating after applying all adjustment factors. Small conductors (#14 through #10) have specific ampacity limitations regardless of the ampacity tables. Tap conductors have special rules for sizing and length based on the specific application. Temperature correction factors must be applied when ambient temperatures exceed 30°C (86°F). The code requires consideration of terminal temperature ratings, which may limit the maximum current. All ungrounded conductors must have overcurrent protection, with limited exceptions for control circuits and specific applications. This code is fundamental for preventing conductor overheating and maintaining system safety."
        },
        "250.122": {
            "title": "Size of Equipment Grounding Conductors",
            "description": "NEC Code 250.122 details the requirements for sizing equipment grounding conductors (EGC). The minimum size must be determined based on the rating of the overcurrent device protecting the circuit conductors, as specified in Table 250.122. When conductors are increased in size for voltage drop compensation, the EGC size must be increased proportionally. For parallel conductors, each parallel run must have an EGC sized based on the circuit overcurrent device. Proper termination methods must be maintained using listed lugs and connectors. The EGC cannot be smaller than the specified minimums in any case, and installation must account for conduit fill calculations. This code ensures proper ground fault current paths and equipment safety."
        },
        "300.5": {
            "title": "Underground Installations",
            "description": "NEC Code 300.5 provides comprehensive requirements for underground electrical installations. Minimum burial depths vary by circuit type and location: 24 inches for general circuits, 18 inches for residential branch circuits with GFCI protection, and 6 inches for low-voltage lighting. Direct-buried cables and conduits must be protected from physical damage using approved methods. Warning tape must be installed 12 inches above electrical conductors. Proper separation must be maintained from other utilities: 12 inches from telecommunications, 12 inches from gas/water, and 3 feet from fuel tanks. Raceway selection must be suitable for soil conditions and chemical exposure. Installation depth must account for the local frost line, and proper drainage must be provided to prevent water accumulation. This code ensures safe and reliable underground electrical installations."
        },
        "300.13": {
            "title": "General Installation Requirements",
            "description": "NEC Code 300.13 establishes fundamental requirements for electrical installations. Mechanical and electrical continuity of all conductors must be maintained throughout the system. No splices or terminations are permitted except within approved enclosures with proper covers. Device removal must not interrupt continuity of grounded conductors in multi-wire branch circuits. All connections must be made with approved methods and materials, properly secured and protected from physical damage. Raceways and cable assemblies must be mechanically continuous between boxes, fittings, and enclosures. This code ensures reliable electrical connections and system integrity."
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
    nec_pattern = r'NEC (\d+\.\d+)'
    
    def format_nec_code(match):
        code = match.group(1)
        nec_info = get_nec_code_info(code)
        if nec_info["found"]:
            return f"**NEC {code} {nec_info['codes'][code]['title']}**\n{nec_info['codes'][code]['description']}"
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
