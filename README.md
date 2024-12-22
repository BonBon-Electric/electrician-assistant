# Electrician Assistant

An AI-powered assistant for electricians that provides expert guidance, cost estimation, and code reference features.

## Features

1. **Expert Electrician Chat**
   - Get answers to electrical questions
   - Troubleshooting guidance
   - Best practices and recommendations

2. **Cost Estimator**
   - Generate detailed cost estimates for electrical jobs
   - Includes materials, labor, and permit costs
   - JSON-formatted output for easy integration

3. **Code Reference**
   - Search and reference NEC (National Electrical Code) requirements
   - Get code interpretations and explanations
   - Stay compliant with current standards

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Unix/MacOS:
     ```bash
     source venv/bin/activate
     ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

5. Run the application:
   ```bash
   streamlit run app.py
   ```

## Project Structure

- `app.py`: Main application file containing the Streamlit interface and core functionality
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (API keys)

## Dependencies

- Python 3.10+
- Streamlit
- Google Generative AI (Gemini)
- ChromaDB
- Other dependencies listed in requirements.txt

## Usage

1. Start the application using `streamlit run app.py`
2. Select a feature from the sidebar:
   - Chat: Ask questions and get expert guidance
   - Cost Estimator: Generate detailed job cost estimates
   - Code Reference: Search and reference electrical codes

## Notes

- Ensure you have a valid Google API key for the Gemini model
- The application will prompt for the API key if not found in the .env file
- Cost estimates are AI-generated and should be verified by a professional
- Code references should be cross-checked with official NEC documentation
