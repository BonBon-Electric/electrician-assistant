import google.generativeai as genai
import chromadb
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure Google API
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-pro')

# Initialize ChromaDB
chroma_client = chromadb.EphemeralClient()
collection = chroma_client.create_collection(
    name="electrician_docs",
    get_or_create=True
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
    "Can I install a regular outlet in a kitchen countertop area?"
]

print("Starting tests...\n")
for i, question in enumerate(test_questions, 1):
    print(f"\nTest {i}: {question}")
    print("-" * 80)
    response = get_chat_response(question)
    print(response)
    print("=" * 80)
