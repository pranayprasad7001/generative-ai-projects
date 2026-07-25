from functools import partial
from typing import List
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph, START


# Graph State Definition
class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    search_query: str
    datasource: str
    generation: str
    documents: List[Document]


# Node Functions
def retrieve(state: GraphState, retriever) -> dict:
    """Retrieve documents from the vectorstore."""
    print("---RETRIEVE FROM VECTORSTORE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def wiki_search(state: GraphState, wiki) -> dict:
    """Search Wikipedia for the query."""
    print("---WIKIPEDIA SEARCH---")
    query = state.get("search_query", state["question"])
    res = wiki.invoke({"query": query})
    
    doc = Document(page_content=str(res))
    return {"documents": [doc], "question": state["question"]}


def arxiv_search(state: GraphState, arxiv) -> dict:
    """Search Arxiv for the query."""
    print("---ARXIV SEARCH---")
    query = state.get("search_query", state["question"])
    res = arxiv.invoke({"query": query})
    
    doc = Document(page_content=str(res))
    return {"documents": [doc], "question": state["question"]}


def generate(state: GraphState, llm) -> dict:
    """
    Streams response token-by-token directly to stdout.
    """
    print("\n---GENERATE ANSWER---")
    question = state["question"]
    documents = state.get("documents", [])
    
    context = "\n\n".join([doc.page_content for doc in documents])
    
    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context below.
        If the context doesn't contain the answer, say "I don't have enough context to answer."

        Context:
        {context}

        Question: {question}
        Answer:"""

    print("\nAnswer: ", end="", flush=True)
    full_response = ""
    
    # Stream token chunks in real-time
    for chunk in llm.stream(prompt):
        content = chunk.content if hasattr(chunk, "content") else str(chunk)
        print(content, end="", flush=True)
        full_response += content
        
    print("\n")
    return {"generation": full_response}


def router_node(state: GraphState, question_router) -> dict:
    print("\n---ROUTING QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})

    # Support both dictionary and Pydantic object formats safely
    if isinstance(source, dict):
        datasource = source.get("datasource")
        search_query = source.get("query", question)
    else:
        datasource = getattr(source, "datasource", None)
        search_query = getattr(source, "query", question)

    return {"datasource": datasource, "search_query": search_query}


def route_question(state: GraphState) -> str:
    datasource = state.get("datasource")
    if datasource == "wiki_search":
        print("---ROUTING TO WIKIPEDIA---")
        return "wiki_search"
    elif datasource == "arxiv_search":
        print("---ROUTING TO ARXIV---")
        return "arxiv_search"
    else:
        print("---ROUTING TO VECTORSTORE---")
        return "vectorstore"


# Graph Builder Factory Function
def build_graph(retriever, question_router, wiki_tool, arxiv_tool, llm):
    """Assembles and compiles the LangGraph workflow."""
    workflow = StateGraph(GraphState)

    # Bind dependency objects to nodes using partial
    node_router = partial(router_node, question_router=question_router)
    node_retrieve = partial(retrieve, retriever=retriever)
    node_wiki = partial(wiki_search, wiki=wiki_tool)
    node_arxiv = partial(arxiv_search, arxiv=arxiv_tool)
    node_generate = partial(generate, llm=llm)

    # Add Nodes
    workflow.add_node("router", node_router)
    workflow.add_node("retrieve", node_retrieve)
    workflow.add_node("wiki_search", node_wiki)
    workflow.add_node("arxiv_search", node_arxiv)
    workflow.add_node("generate", node_generate)

    # Add Edges
    workflow.add_edge(START, "router")
    
    # Add Conditional Edges from router
    workflow.add_conditional_edges(
        "router",
        route_question,
        {
            "wiki_search": "wiki_search",
            "arxiv_search": "arxiv_search",
            "vectorstore": "retrieve",
        },
    )

    # Route all retrieval nodes through the generate node
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("wiki_search", "generate")
    workflow.add_edge("arxiv_search", "generate")

    # Edge from generate to END
    workflow.add_edge("generate", END)

    # Initialize Memory Saver Checkpointer
    memory = InMemorySaver()

    # Compile Graph Application
    app = workflow.compile(checkpointer=memory)
    return app