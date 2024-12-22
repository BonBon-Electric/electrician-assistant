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

def initialize_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_tab" not in st.session_state:
        st.session_state.current_tab = "NEC Electrical Assistant"
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

def get_chat_response(prompt: str) -> str:
    # Include chat history in the context
    context = "\n".join([f"{msg['role']}: {msg['content']}" 
                        for msg in st.session_state.chat_history[-3:]])
    
    if is_greeting(prompt):
        # Extract name if present
        name = ""
        if "name is" in prompt.lower():
            name = prompt.lower().split("name is")[-1].strip()
            name = name.split()[0].title()  # Get first word and capitalize
        
        response = f"Hello{' ' + name if name else ''}! I'm your electrical assistant. How can I help you today?"
        return response
    
    elif is_electrical_question(prompt):
        full_prompt = f"""Previous conversation context:
{context}

Current question:
{prompt}

Respond in this format:
SUMMARY:
[Provide a brief, direct answer to the user's question in 1-2 sentences]

RELEVANT NEC CODES:
[List relevant NEC codes]

DETAILED RESPONSE:
[Provide detailed explanation]"""
    else:
        # For general conversation
        full_prompt = f"""Previous conversation context:
{context}

Current question:
{prompt}

Respond conversationally and naturally to the user's message."""

    response = get_gemini_response(full_prompt)
    update_chat_history("assistant", response)
    return response

def create_cost_estimate(description: str) -> Dict[str, Any]:
    """Generate a cost estimate based on the job description"""
    prompt = f"""Create a detailed cost estimate for the following electrical job in Los Angeles County, California:
{description}

Consider the following Los Angeles-specific factors:
- Los Angeles County permit fees and requirements
- Los Angeles area labor rates (average $85-125/hour for licensed electricians)
- Local material costs including California markups
- Los Angeles County inspection requirements
- Title 24 energy efficiency requirements
- Additional requirements specific to LA County jurisdiction

Please provide the response in the following JSON format:
{{
    "materials": [
        {{"item": "string", "quantity": number, "unit": "string", "cost_per_unit": number, "total_cost": number}}
    ],
    "labor": {{
        "hours": number,
        "rate_per_hour": number,
        "total_labor_cost": number
    }},
    "permits": {{
        "required_permits": ["string"],
        "total_permit_cost": number,
        "inspection_requirements": ["string"]
    }},
    "summary": {{
        "materials_total": number,
        "labor_total": number,
        "permits_total": number,
        "subtotal": number,
        "contingency": number,
        "total_cost": number
    }},
    "notes": ["string"],
    "compliance": {{
        "title_24": ["string"],
        "local_requirements": ["string"]
    }}
}}

Include realistic Los Angeles County costs, quantities, and a 10% contingency factor. Format all monetary values as numbers without currency symbols."""
    
    try:
        response = get_gemini_response(prompt)
        # Ensure the response is valid JSON
        estimate = json.loads(response)
        return estimate
    except json.JSONDecodeError as e:
        st.error(f"Error parsing cost estimate. Please try again.")
        return {}

def display_cost_estimate(estimate: Dict[str, Any]):
    """Display the cost estimate in a formatted way"""
    if not estimate:
        return

    st.subheader("📋 Cost Estimate Breakdown")
    
    # Materials Section
    st.write("### 🛠️ Materials")
    materials_df = pd.DataFrame(estimate['materials'])
    materials_df['total_cost'] = materials_df['total_cost'].apply(lambda x: f"${x:,.2f}")
    materials_df['cost_per_unit'] = materials_df['cost_per_unit'].apply(lambda x: f"${x:,.2f}")
    st.dataframe(materials_df, hide_index=True)

    # Labor Section
    st.write("### 👷 Labor")
    labor = estimate['labor']
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Hours", f"{labor['hours']} hrs")
    with col2:
        st.metric("Hourly Rate", f"${labor['rate_per_hour']}/hr")
    st.metric("Total Labor Cost", f"${labor['total_labor_cost']:,.2f}")

    # Permits Section
    st.write("### 📄 Permits and Inspections")
    st.write("Required Permits:")
    for permit in estimate['permits']['required_permits']:
        st.write(f"- {permit}")
    st.write("Inspection Requirements:")
    for req in estimate['permits']['inspection_requirements']:
        st.write(f"- {req}")
    st.metric("Total Permit Cost", f"${estimate['permits']['total_permit_cost']:,.2f}")

    # Compliance Section
    st.write("### ✅ Compliance Requirements")
    st.write("Title 24:")
    for req in estimate['compliance']['title_24']:
        st.write(f"- {req}")
    st.write("Local Requirements:")
    for req in estimate['compliance']['local_requirements']:
        st.write(f"- {req}")

    # Summary Section
    st.write("### 💰 Cost Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Materials Total", f"${estimate['summary']['materials_total']:,.2f}")
    with col2:
        st.metric("Labor Total", f"${estimate['summary']['labor_total']:,.2f}")
    with col3:
        st.metric("Permits Total", f"${estimate['summary']['permits_total']:,.2f}")
    
    st.metric("Subtotal", f"${estimate['summary']['subtotal']:,.2f}")
    st.metric("Contingency (10%)", f"${estimate['summary']['contingency']:,.2f}")
    st.metric("Total Cost", f"${estimate['summary']['total_cost']:,.2f}", delta="Includes contingency")

    # Notes Section
    if estimate.get('notes'):
        st.write("### 📝 Additional Notes")
        for note in estimate['notes']:
            st.write(f"- {note}")

def display_chat_history():
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def main():
    st.title("Electrician Assistant")
    initialize_session_state()
    
    # Create tabs instead of sidebar
    nec_tab, cost_tab = st.tabs(["💡 NEC Electrical Assistant", "💰 Cost Estimator"])
    
    # NEC Assistant Tab
    with nec_tab:
        st.write("## NEC Electrical Assistant")
        display_chat_history()
        
        if prompt := st.chat_input("Ask your question about electrical work:"):
            update_chat_history("user", prompt)
            with st.chat_message("user"):
                st.write(prompt)
                
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_chat_response(prompt)
                    st.write(response)
    
    # Cost Estimator Tab
    with cost_tab:
        st.write("## Cost Estimator")
        st.write("Describe the electrical work you need, and I'll provide a detailed cost estimate.")
        
        job_description = st.text_area("Job Description:", height=100,
            placeholder="Example: Install a 200 amp service panel in a residential home in Los Angeles")
        
        if st.button("Generate Estimate", type="primary"):
            if job_description:
                with st.spinner("Generating cost estimate..."):
                    estimate = create_cost_estimate(job_description)
                    display_cost_estimate(estimate)
            else:
                st.warning("Please provide a job description.")

if __name__ == "__main__":
    main()
