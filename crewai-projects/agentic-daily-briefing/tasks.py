from crewai import Task
from tools import tavily_tool
from agents import news_researcher, news_writer

# News Research Task
news_research_task = Task(
    description=(
        "Execute targeted web searches using the Tavily tool to gather today's trending news. "
        "You must find news covering exactly these 5 distinct categories:\n"
        "1. Breaking News\n"
        "2. India (National/Regional news)\n"
        "3. World (International affairs)\n"
        "4. AI & Tech (Latest developments, product launches)\n"
        "5. Science (Space, health, discoveries)\n\n"
        "Ensure you extract the core headline, a brief 1-2 sentence summary, and the source URL for each story. "
        "Strictly cap your findings at a maximum of 10 distinct stories per category."
    ),
    expected_output=(
        "A raw research document categorized into the 5 specified sections, containing "
        "up to 10 verified news bullet points per section with headlines, summaries, and source URLs."
    ),
    agent=news_researcher,
    tools=[tavily_tool],
    output_file="/data/trending_news_research.md",
    async_execution = False
)

# News Formatting & Writing Task
news_writing_task = Task(
    description=(
        "Review the gathered news research and compile it into a beautifully formatted, executive daily briefing. "
        "Organize the brief cleanly using markdown headers for the 5 categories: Breaking News, India, World, AI & Tech, and Science. "
        "Ensure each category features a maximum of 10 items. Every news item must be structured as a clean bullet point "
        "containing the headline bolded, a concise executive summary, and a markdown link to the source."
    ),
    expected_output=(
        "A highly polished, comprehensive daily news briefing divided into the 5 requested categories, "
        "containing a maximum of 10 well-structured, engaging news items per section."
    ),
    agent=news_writer,
    tools=[tavily_tool],
    output_file="/data/daily_news_briefing.md",
    async_execution=False
)