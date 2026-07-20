import os
from datetime import date
from crewai import Crew, Process
from agents import news_researcher, news_writer
from tasks import news_research_task, news_writing_task
from dotenv import load_dotenv

load_dotenv()
os.environ["CREWAI_TRACING_ENABLED"] = os.getenv("CREWAI_TRACING_ENABLED")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

hf_embedder = dict(
    provider="huggingface",
    config=dict(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
)

# Create the Crew
crew = Crew(
    agents=[news_researcher, news_writer],
    tasks=[news_research_task, news_writing_task],
    process=Process.sequential, 
    memory=False,  
    cache=True,
    tracing=True,  
    max_rpm=30,  
    share_crew=True,  
    verbose=True,  
    max_execution_concurrency=1,
    embedder=hf_embedder
)

# Get the date in the format "Month Day, Year" (e.g., "July 20, 2026")
today = date.today().strftime("%B %d, %Y")

# Run the Crew
result = crew.kickoff(inputs={'date': today})

# Print the result
print(result)