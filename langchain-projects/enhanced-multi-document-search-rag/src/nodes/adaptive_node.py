"""Langgraph nodes for RAG workflow + Agent inside generate_content"""

from pydantic import BaseModel, Field
from typing import List
from state.adaptive_state import RAGState
from langchain_classic.schema import Document
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from typing import Literal
from config.mcp_config import MCPToolManager

MAX_REWRITES = 2

class _ToolUse(BaseModel):
    """Schema representing the routing decision and justification from the query analyzer."""
    tool_type: Literal["vector_search", "external", "none"] = Field(..., description="Tool type to use")
    analysis: str = Field(..., description="Analysis of the tool to use")

class RetrievalGrade(BaseModel):
    grade: Literal["yes", "no"] = Field(
        description="Whether the retrieved documents are relevant and sufficient."
    )

    reasoning: str = Field(
        description="Concise explanation for the grading decision."
    )

class QuestionRewrite(BaseModel):
    rewritten_question: str = Field(
        description="Improved retrieval-focused version of the question."
    )

    reasoning: str = Field(
        description="Brief explanation of why the rewrite should improve retrieval."
    )

class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        """
        Initialize RAG nodes with retriever and llm

        Args:
            retriever: Document retriever instance
            llm: Language model instance
        """
        self.retriever = retriever
        self.llm = llm
        self._web_search_agent = None
        self.mcp_manager = MCPToolManager()

    def query_analyzer(self, state: RAGState) -> RAGState:
        """
        Analyze the query and determine which direction to route

        Args:
            state: Current RAG state

        Returns:
            Updated RAG state with analysis and which tool to use
        """
        system_prompt = """
        You are the query analyzer for an autonomous Retrieval-Augmented Generation
        (RAG) system.

        Your responsibility is to determine whether the user's question requires
        retrieval and, if so, which source should be used.

        You must choose exactly one tool type:

            1. vector_search
               Choose this when the answer is expected to be found in the application's
               indexed documents or knowledge base.

            Examples:
            - Questions about uploaded documents
            - Questions about company policies
            - Questions asking for information contained in manuals, reports,
              research papers, or other indexed documents

            2. external
               Choose this when the answer requires general-world or external knowledge
               that is not expected to be present in the application's document store.

            Examples:
            - General factual questions
            - Wikipedia-style questions
            - Questions about historical figures, countries, events, or concepts
              not contained in the application's documents
            - Information that requires an external knowledge source

            3. none
            Choose this when retrieval is unnecessary.

            Examples:
            - Greetings
            - Farewells
            - Thanks
            - Casual conversation
            - Simple questions that can be answered without retrieving information

            Decision rules:
            - Prefer vector_search when the question clearly refers to the application's
              documents or knowledge base.
            - Use external when the question requires knowledge outside those documents.
            - Use none when no retrieval is necessary.
            - Do not choose a tool merely because a question is phrased as a question.
            - Focus on the user's intent and the likely source of the required information.

            Return a structured result containing:
            - tool_type: the selected tool type
            - analysis: a concise explanation supporting the decision
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])

        response = self.llm.with_structured_output(_ToolUse).invoke(
            prompt.format_messages(question=state.question)
        )

        state.analysis = response.analysis
        state.tool_type = response.tool_type
        return state

    def query_router(self, state: RAGState) -> str:
        """
        Routes the query to the appropriate tool based on the analysis

        Args:
            state: Current RAG state

        Returns:
            Updated RAG state with routed query
        """
        if state.tool_type == "vector_search":
            return "vector_search"
        elif state.tool_type == "external":
            return "external_search"
        else:
            return "none_search"

    async def external_search(self, state: RAGState) -> RAGState:
        """
        Perform external search to find relevant information

        Args:
            state: Current RAG state

        Returns:
            Updated RAG state with retrieved information
        """
        
        system_prompt = """
            You are an external knowledge research agent in an autonomous RAG system.

            Your task is to answer the user's question by selecting and using the most
            appropriate external search tool available to you.

            You have access to three tools:

            1. wikipedia
                Use this for general factual and encyclopedic knowledge.
                Examples:
                - "Who was Albert Einstein?"
                - "What is the history of the Eiffel Tower?"
                - "What is France?"
                - "Explain the concept of photosynthesis."

            2. tavily_search
                Use this for general web search when the question requires broader,
                current, or web-based information.
                Examples:
                - "What are the latest developments in generative AI?"
                - "What happened in the recent OpenAI announcement?"
                - "Find the latest information about NVIDIA's AI chips."
                - Questions where current or multiple web sources would be useful.

            3. arxiv_search
                Use this when the question specifically asks about scientific research,
                academic papers, machine learning research, or research findings.
                Examples:
                - "Find papers about Retrieval-Augmented Generation."
                - "What research has been done on RAG evaluation?"
                - "Find papers about Vision Transformers."
                - "What are recent approaches to autonomous agents?"

            Tool selection guidelines:
            
            - Prefer Wikipedia for stable, general encyclopedic facts.
            - Prefer Tavily for current events, recent information, broad web searches,
                or information that may not be available in Wikipedia.
            - Prefer arXiv for academic papers and scientific/technical research.
            - Use only the tool or tools necessary to answer the question.
            - Do not use a tool if the question can be answered from information already
                available in the conversation.
            - If multiple sources are genuinely useful, you may use more than one tool.
            - Do not fabricate information or search results.
            - Base the final answer on the information returned by the selected tools.

            After performing the search, provide a concise answer to the user's question
            based on the retrieved information.
        """

        tools = await self.mcp_manager.get_tools()

        if self._web_search_agent is None:
            self._web_search_agent = create_agent(
                self.llm,
                system_prompt=system_prompt,
                tools=tools
            )

        response = await self._web_search_agent.ainvoke({
            "messages": [
                ("user", state.question)
            ]
        })
        
        messages = response.get("messages", [])
        answer = messages[-1].content if messages else response.get("output", "")
        
        state.external_results = answer
        state.answer = answer
        return state

    def none_search(self, state: RAGState) -> RAGState:
        """
        Perform none search to find relevant information

        Args:
            state: Current RAG state

        Returns:
            Updated RAG state with retrieved information
        """
        system_prompt = """
            You are the direct-answering component of an autonomous RAG system.

            The user's question has already been classified as not requiring any
            retrieval from the vector store or external search tools.

            Your task is to answer the user's question directly using the information
            available in the conversation and your general knowledge.

            Guidelines:

            1. Do not perform any vector search.
            2. Do not use external search tools.
            3. Do not invent facts or sources.
            4. Answer clearly and directly.
            5. If the user is greeting, thanking, or engaging in casual conversation,
               respond naturally and conversationally.
            6. If the question is a simple factual or conceptual question that does not
               require retrieval, provide a concise and accurate explanation.
            7. If the question cannot be answered reliably without additional
               information, clearly state what information is missing instead of
               fabricating an answer.
            8. Match the level of detail to the user's question.

            Return only the final answer to the user's question.
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])

        response = self.llm.invoke(
            prompt.format_messages(
                question=state.question
            )
        )

        state.answer = response.content
        return state

    def vector_search(self, state: RAGState) -> RAGState:
        """Perform vector search to find relevant documents."""

        retrieved_documents: List[Document] = self.retriever.invoke(state.question)
        state.retrieved_docs = retrieved_documents
        return state

    def grader(self, state: RAGState) -> RAGState:
        """Grade the relevance and sufficiency of retrieved documents."""

        system_prompt = """
            You are a document relevance grader in an autonomous RAG system.

            Your task is to evaluate whether the documents retrieved from the local
            knowledge base are relevant and sufficient to answer the user's question.

            You will receive:
            1. The user's question.
            2. The documents retrieved from the local knowledge base.

            Evaluate the retrieved documents using the following criteria:

            1. Relevance:
            Determine whether the retrieved documents contain information directly
            related to the user's question.

            2. Sufficiency:
            Determine whether the retrieved documents contain enough information
            to reasonably answer the question.

            3. Grounding:
            Determine whether the documents provide factual evidence that can be
            used to support an answer.

            Decision rules:

            - Return "yes" if the retrieved documents are relevant and contain
              sufficient information to answer the question.
            - Return "no" if the documents are irrelevant, unrelated, empty, or
              insufficient to answer the question.
            - If only some documents are relevant but the relevant information is
              sufficient to answer the question, return "yes".
            - If the retrieved documents contain only partial or weakly related
              information and an accurate answer cannot reasonably be generated,
              return "no".

            Important instructions:

            - Do not answer the user's question.
            - Do not use your own general knowledge to fill missing information.
            - Judge only the retrieved documents against the user's question.
            - Focus on whether the retrieved context is good enough for the next
              generation step.

            Return your evaluation in the following structured format:

            grade: "yes" | "no"
            reasoning: "Concise explanation of why the retrieved documents are or are
                        not sufficient."
        """

        prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        (
            "human",
            """
            User Question:
            {question}

            Retrieved Documents:
            {documents}
            """
        )])

        documents = "\n\n".join(
            f"[Document {i}]\n{doc.page_content}"
            for i, doc in enumerate(state.retrieved_docs, start=1)
        )

        response = self.llm.with_structured_output(RetrievalGrade).invoke(prompt.format_messages(
                question=state.question,
                documents=documents
            )
        )

        state.grade = response.grade
        state.analysis = response.reasoning

        return state

    def grader_router(self, state: RAGState) -> str:
        """
        Route to next node based on grader output
        """
        if state.grade == "yes":
            return "generate"
        if state.rewrite_count >= MAX_REWRITES: # TODO : after max tries, fall back to external search
            return "generate"
        return "rewriter"
    
    def rewriter(self, state: RAGState) -> RAGState:
        """
        Rewrite the user's question to improve retrieval accuracy
        """
        system_prompt = """
        You are a question rewriter in an autonomous RAG system.

        Your task is to rewrite the user's question so that it is more likely to
        retrieve relevant information from the internal knowledge base.

        The previous retrieval attempt was judged insufficient or irrelevant.

        You will receive:
        1. The original user question.
        2. The current version of the question used for retrieval.

        Your goal is to improve the retrieval query while preserving the user's
        original intent.

        Rewrite guidelines:

        1. Preserve the exact meaning and intent of the user's question.
        2. Identify the key concepts, entities, topics, and relationships that are
           important for retrieval.
        3. Make vague or ambiguous wording more precise when the intended meaning
           can be inferred from the original question.
        4. Expand acronyms or abbreviations when their meaning is clear from the
           question.
        5. Replace conversational or indirect wording with clear, retrieval-friendly
           terminology.
        6. Include important keywords from the original question that may improve
           semantic or keyword matching.
        7. If the question contains multiple concepts, restructure it so the
           relationship between those concepts is explicit.
        8. Do not introduce facts, entities, assumptions, or context that are not
           supported by the original question.
        9. Do not change the scope of the question.
        10. Do not answer the question.
        11. Keep the rewritten question concise, preferably one or two sentences.
        12. If the current question is already an effective retrieval query, return
            it unchanged.

        The purpose of the rewrite is to improve document retrieval, not to make the
        question more elaborate.

        Return the result in the following structured format:

        rewritten_question: "Improved retrieval-focused version of the question"
        reasoning: "Brief explanation of what was changed and why it should improve retrieval"
        """

        prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        (
            "human",
            """
            Original Question:
            {original_question}

            Current Question:
            {current_question}
            """
        )])

        response = self.llm.with_structured_output(QuestionRewrite).invoke(
            prompt.format_messages(
                original_question=state.original_question,
                current_question=state.question,
            )
        )

        state.question = response.rewritten_question
        state.analysis = response.reasoning
        state.rewrite_count += 1

        return state