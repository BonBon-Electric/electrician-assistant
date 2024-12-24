import google.generativeai as genai
import chromadb
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

# Configure Google API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in .env file")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-pro')

# Initialize ChromaDB
chroma_client = chromadb.EphemeralClient()
collection = chroma_client.create_collection(
    name="electrician_docs",
    get_or_create=True
)

# Sample NEC codes for testing
nec_codes = [
    {
        "content": "NEC 210.8(A)(1) - Bathroom GFCI Requirements: All 125-volt, single-phase, 15- and 20-ampere receptacles installed in bathrooms must have GFCI protection. This includes receptacles installed within 6 feet of the outside edge of a sink, bathtub, or shower stall.",
        "metadata": {"section": "210.8", "topic": "GFCI Protection"}
    },
    {
        "content": "NEC 300.5 - Underground Installations: Direct buried cables or conduits must be installed according to the Minimum Cover Requirements table. For direct burial of 0-600V nominal circuits: 24 inches for non-residential, 18 inches for residential with GFCI protection.",
        "metadata": {"section": "300.5", "topic": "Underground Installation"}
    },
    {
        "content": "NEC 310.15 - Wire Ampacity: For a 50-amp circuit at 75°C, use: #8 copper or #6 aluminum for THHN/THWN/XHHW-2 conductors. Temperature and conduit fill corrections may apply.",
        "metadata": {"section": "310.15", "topic": "Conductor Sizing"}
    },
    {
        "content": "NEC 110.26 - Working Space: Minimum clear working space in front of electrical equipment (0-150V): depth 3 feet, width 30 inches or width of equipment, height 6.5 feet. Must maintain clear access at all times.",
        "metadata": {"section": "110.26", "topic": "Working Space"}
    },
    {
        "content": "NEC 210.52(C) - Kitchen Countertop Receptacles: Must be GFCI-protected, maximum 24 inches apart, no point along wall more than 12 inches from receptacle. Separate circuit required for countertop receptacles.",
        "metadata": {"section": "210.52", "topic": "Kitchen Receptacles"}
    }
]

# Load test data into ChromaDB
for i, code in enumerate(nec_codes):
    collection.add(
        documents=[code["content"]],
        metadatas=[code["metadata"]],
        ids=[f"code_{i}"]
    )

def get_chat_response(prompt: str) -> str:
    """Get response from Gemini model and ensure NEC codes are included"""
    
    # First get relevant NEC codes from RAG
    results = collection.query(
        query_texts=[prompt],
        n_results=3
    )
    
    rag_nec_codes = ""
    if results and results['documents'][0]:
        rag_nec_codes = "\n".join(results['documents'][0])
    
    # Get response from Gemini with NEC code requirement
    gemini_prompt = f"""You are an expert electrical contractor. Answer this electrical question with practical, accurate information. 
IMPORTANT: You MUST include all relevant NEC code references and requirements that apply to this situation.

Question: {prompt}

Here are some relevant NEC codes to consider (verify and include any additional relevant codes):
{rag_nec_codes}"""
    
    response = model.generate_content(gemini_prompt)
    
    # Add RAG NEC codes if they weren't already mentioned in the response
    final_response = response.text
    if rag_nec_codes and not any(code in response.text for code in ["NEC", "Code", "Article"]):
        final_response += "\n\n### Additional Relevant NEC Codes:\n" + rag_nec_codes
    
    return final_response

# Test questions
test_questions = [
    "What are the requirements for installing a GFCI outlet in a bathroom?",
    "How deep should electrical conduit be buried underground?",
    "What's the proper wire size for a 50 amp circuit?",
    "What are the clearance requirements for electrical panels?",
    "Can I install a regular outlet in a kitchen countertop area?",
    "What are the requirements for outdoor electrical outlets?",
    "How many outlets can I put on a 20 amp circuit?",
    "What's the minimum height for electrical outlets in a garage?",
    "Do I need GFCI protection for basement outlets?",
    "What are the requirements for electrical panel labeling?",
    "How far can a junction box be from an access point?",
    "What type of wire should I use for a hot tub installation?",
    "What are the requirements for smoke detector wiring?",
    "How many circuits do I need for a kitchen?",
    "What's the proper grounding method for a subpanel?",
    "What are the requirements for bathroom exhaust fan wiring?",
    "How do I properly wire a three-way switch?",
    "What are the requirements for arc fault protection?",
    "How should I wire a garage door opener?",
    "What are the requirements for electrical service entrance?"
]

print("Starting tests...\n")
for i, question in enumerate(test_questions, 1):
    print(f"\nTest {i}: {question}")
    print("-" * 80)
    try:
        response = get_chat_response(question)
        print("\nResponse:")
        if hasattr(response, 'text'):
            print(response.text)
            response_text = response.text
        elif hasattr(response, 'parts'):
            response_text = ''.join([part.text for part in response.parts])
            print(response_text)
        else:
            response_text = str(response)
            print(response_text)
        
        # Analysis
        print("\nAnalysis:")
        has_nec_codes = "NEC" in response_text or "Code" in response_text
        print(f"- Contains NEC references: {'Yes' if has_nec_codes else 'No'}")
        print(f"- Response length: {len(response_text)} characters")
        
    except Exception as e:
        print(f"Error testing question: {str(e)}")
    
    print("=" * 80)
    time.sleep(2)  # Avoid rate limiting

print("\nTest complete!")
