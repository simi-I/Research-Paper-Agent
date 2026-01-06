import arxiv
import json
import os
import logging
from typing import List, Any
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent
from google.adk.models.google_llm import Gemini
from google.adk.tools import AgentTool
from google.genai import types 

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tools

def search_arxiv(query : str, max_results: int= 5) -> str:
    try:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        results = []
        for paper in search.results():
            result = {
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "summary": paper.summary[:300] + "..." if len(paper.summary) > 300 else paper.summary,
                "published": paper.published.strftime("%Y-%m-%d"), 
                "url": paper.entry_id
            }
            results.append(result)
            
        if not results:
            return f"No papers found for query: {query}"
        
        formatted = f"Found {len(results)} papers for '{query}':\n\n"
        for i, paper in enumerate(results, 1):
            authors = ", ".join(paper["authors"][:3])
            if len(paper["authors"]) > 3:
                authors += " et al."
            formatted += f"{i}. **{paper['title']}**\n"
            formatted += f" Authors: {authors}\n"
            formatted += f" Published: {paper['published']}\n"
            formatted += f" URL: {paper['url']}\n"
            formatted += f" Summary: {paper['summary']}\n\n"
        return formatted
    except Exception as e:
        return f"Error searching ArXiv: {str(e)}"
    
def format_citation(title: str, authors: List[str], year: str, url: str) -> str:
    """Formats a research paper citation"""
    author_str = ", ".join(authors)
    return f"{author_str} ({year}). **{title}**. Retrieved from {url}"

    
# Agents

retry_config = types.HttpRetryOptions(
    attempts=3, 
    exp_base=2, 
    initial_delay=1, 
    http_status_codes=[429, 500, 503, 504],
)

model = Gemini(model='gemini-2.5-flash-lite', retry_options=retry_config)

# Research Agent
researcher_agent = LlmAgent(
    name="researcher_agent", 
    model= model, 
    description="searches for research papers using ArXiv API",
    instruction="""
    You are a research assistant with access to ArXiv.
    When given a research topic:
    1. Use the 'search_arxiv' tool to find relevant academic papers.
    2. Focus on recent papers (2023-2025 when possible).
    3. Return the paper details including title, authors, summary, dates and URLs.
    """, 
    tools = [search_arxiv]
)

# Analyst Agent
analyst_agent = LlmAgent(
    name = "Analyst_agent", 
    model=model, 
    description= "Analyzes research data and creates visualizations.",
    instruction="""
    You are a data analyst.
    Given a list of research data or search results:
    1. Extract the publication years from the paper information.
    2. Calculate the distribution of papers by year.
    3. Create a simple ASCII bar chart showing the distribution.
    4. Return the analysis summary and the ASCII chart.
    """
)

formatter_agent = LlmAgent(
    name="formatter_agent", 
    model=model,
    description="Formats paper details into proper citations.",
    instruction="""
    You are a citation expert.
    Take the raw paper information and format it into clean academic citations.
    Use APA format: Authors (Year). Title. Retrieved from URL
    Return the final list of formatted citations as a numbered list.
    """
)

parallel_agent = ParallelAgent(
    name="Parallel_agent",
    sub_agents=[analyst_agent, formatter_agent]
)


Research_agent = SequentialAgent(
    name="Research_agents",
    sub_agents=[researcher_agent, parallel_agent]
)

root_agent = LlmAgent(
    name="root_agent", 
    model=model,
    instruction="""
    You are the Lead Research Coordinator. 
    You MUST generate a final report summarizing the results from the Research_agent, 
    The research agent is Sequential agent with a researcher agent and a parallel agent to handle analysis and formatting, 
    The final report must contain 
        - The summary of the identified papers, 
        - Show the publication year analysis chart
        - Show the "Formatted citiations"
    """, 
    tools = [
        AgentTool(Research_agent),
        
    ]
)

    
    