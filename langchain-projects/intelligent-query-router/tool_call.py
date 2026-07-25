import wikipedia
import arxiv as arxiv_lib
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun

def tools_init():
    """
    Initializes the tools (Arxiv and Wikipedia).
    
    Returns:
        A list containing the initialized ArxivQueryRun and WikipediaQueryRun tools.
    """

    wikipedia.set_user_agent("IntelligentQueryRouter/1.0 (viren@example.com)")
    arxiv_lib.Client.query_url_format = "https://export.arxiv.org/api/query?{}"

    arxiv_wrapper=ArxivAPIWrapper(top_k_results=3, doc_content_chars_max=3000)
    arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

    api_wrapper=WikipediaAPIWrapper(top_k_results=3,doc_content_chars_max=3000)
    wiki=WikipediaQueryRun(api_wrapper=api_wrapper)
    
    return [wiki, arxiv]