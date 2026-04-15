from src.rag_pipeline import RAGPipeline
from dotenv import load_dotenv
import time

load_dotenv()

questions = [
    "What is the expense ratio for HDFC Mid-Cap Opportunities Fund?",
    "Who is the fund manager for HDFC Small Cap Fund?",
    "What is the minimum SIP amount for HDFC Flexi Cap Fund?",
    "What are the exit load charges for HDFC Balanced Advantage Fund?",
    "Can you give me some advice on where I should invest my 10,000 rupees today?",
    "What was the 1-year return for HDFC Top 100 Fund?",
    "What is the 3 year annualised return for HDFC index nifty 50 plan?"
]

pipeline = RAGPipeline()
output = "# 📄 Sample Q&A from HDFC Mutual Fund Assistant\n\n*These are real responses generated directly from the Retrieval-Augmented Generation (RAG) pipeline.*\n\n---\n\n"

print("Starting generation...")
for i, q in enumerate(questions):
    print(f"Processing question {i+1}/{len(questions)}: {q}")
    output += f"### 👤 User:\n**{q}**\n\n"
    try:
        response = pipeline.answer(q)
        output += f"### 🤖 Assistant:\n{response.answer}\n\n"
        if response.citations:
            output += "**🔗 Sources:**\n"
            for url in response.citations:
                output += f"- [{url}]({url})\n"
        
        if not response.guardrail_passed:
            output += f"\n*⚠️ Guardrail Triggered: {', '.join(response.guardrail_violations)}*\n"
            
        output += "\n---\n\n"
    except Exception as e:
        output += f"**Error generating response:** {e}\n\n---\n\n"
    time.sleep(1) # Prevent rate limiting

with open("sample_qa.md", "w", encoding="utf-8") as f:
    f.write(output)

print("Successfully generated sample_qa.md!")
