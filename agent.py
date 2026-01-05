import arxiv
import json
import os
import logging
from typing import List, Any
from google.adk.agents import LlmAgent
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
    
