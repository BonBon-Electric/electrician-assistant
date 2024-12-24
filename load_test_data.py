import chromadb

# Initialize ChromaDB
client = chromadb.EphemeralClient()
collection = client.create_collection(name="electrician_docs")

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

# Add documents to collection
for i, code in enumerate(nec_codes):
    collection.add(
        documents=[code["content"]],
        metadatas=[code["metadata"]],
        ids=[f"code_{i}"]
    )

print("Test data loaded successfully!")
