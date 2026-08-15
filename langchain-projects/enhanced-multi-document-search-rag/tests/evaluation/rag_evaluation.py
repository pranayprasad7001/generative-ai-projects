"""
RAG Evaluation Pipeline for Enhanced Multi-Document Search RAG
Dataset: data/evaluation_dataset/rag_evaluation_dataset_air_pollution.csv
Knowledge Document: data/Health_effects_of_air_pollution.pdf

Evaluates:
1. Ingestion of Health_effects_of_air_pollution.pdf into AstraDB Vector Store
2. End-to-End Workflow Execution on Golden Evaluation Questions
3. Evaluation Metrics:
   - Answer Semantic Similarity (Embedding Cosine Similarity)
   - Answer Correctness & Completeness (LLM-as-a-Judge 1-5 scale)
   - Faithfulness / Groundedness against Retrieved Chunks
   - Retrieval Relevance & Context Recall
   - System Performance: Latency (s), Cost ($ USD), Rewrite & Iteration counts
4. LangSmith Integration:
   - Dataset & Example sync (`client.create_dataset`, `client.create_examples`)
   - Automated LangSmith Experiment Run (`client.evaluate` / `evaluate`)
"""

import os
import sys
import time
import json
import asyncio
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Setup encoding and load environment
sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

# Add src directory to python path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = REPO_ROOT / "src"
DATA_DIR = REPO_ROOT / "data"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Ensure LiteLLM Base URL default if not explicitly set
if "LITELLM_BASE_URL" not in os.environ:
    os.environ["LITELLM_BASE_URL"] = "http://127.0.0.1:4000"

from config.llmgateway_config import Config
from document_ingestion.document_processor import DocumentProcessor
from document_ingestion.chunker import ChunkStrategy
from vectorstore.vectorstore import VectorStoreManager
from graph_builder.adaptive_graph_builder import GraphBuilder
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client, evaluate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("RAGEvaluation")


# =====================================================================
# Structured Output Schemas for LLM-as-a-Judge
# =====================================================================
class AnswerCorrectnessJudge(BaseModel):
    """Evaluation of generated answer against reference ground truth."""
    score: float = Field(
        ...,
        description="Rating from 1.0 (completely inaccurate/missing) to 5.0 (perfect factual alignment and completeness)."
    )
    is_factual: bool = Field(
        ...,
        description="True if key facts from ground truth are correctly stated."
    )
    missing_facts: List[str] = Field(
        default_factory=list,
        description="List of key facts present in ground truth but missing in the generated answer."
    )
    reasoning: str = Field(
        ...,
        description="Concise justification for the given score."
    )


class GroundednessJudge(BaseModel):
    """Evaluation of whether generated answer is grounded in retrieved context."""
    is_grounded: bool = Field(
        ...,
        description="True if all claims in the generated answer are supported by retrieved documents."
    )
    hallucinated_claims: List[str] = Field(
        default_factory=list,
        description="List of claims in generated answer not supported by retrieved context."
    )
    reasoning: str = Field(
        ...,
        description="Explanation of groundedness judgment."
    )


class RelevanceJudge(BaseModel):
    """Evaluation of whether generated answer directly answers the query."""
    is_relevant: bool = Field(
        ...,
        description="True if the response directly addresses the user query."
    )
    score: float = Field(
        ...,
        description="Rating from 1.0 (completely irrelevant) to 5.0 (directly and completely addresses query)."
    )
    reasoning: str = Field(
        ...,
        description="Explanation of answer relevance judgment."
    )


# =====================================================================
# Evaluation Prompts
# =====================================================================
CORRECTNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert impartial evaluator assessing RAG question-answering systems.
Compare the SYSTEM GENERATED ANSWER against the GROUND TRUTH REFERENCE ANSWER for the given USER QUESTION.

Score the SYSTEM ANSWER on a 1.0 to 5.0 scale:
5.0 = Excellent: Captures all facts from the reference, perfectly accurate, comprehensive, and relevant.
4.0 = Good: Minor missing detail or stylistic variance, but all core facts match reference.
3.0 = Fair: Partially accurate, misses some key facts, or contains minor inaccuracies.
2.0 = Poor: Major factual gaps, mostly inaccurate, or largely unrelated.
1.0 = Very Poor: Completely incorrect, hallucinated, or irrelevant.

Output structured JSON matching the schema."""),
    ("human", """User Question: {question}

Ground Truth Reference:
{ground_truth}

System Generated Answer:
{generated_answer}
""")
])

GROUNDEDNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert fact-checking evaluator.
Determine whether the GENERATED ANSWER is strictly supported by the RETRIEVED CONTEXT.
If the answer makes statements or claims that are not present in the context, flag them as hallucinated.
Output structured JSON matching the schema."""),
    ("human", """User Question: {question}

Retrieved Context:
{context}

Generated Answer:
{generated_answer}
""")
])

RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an evaluator assessing if a GENERATED ANSWER directly addresses the USER QUESTION.
Score the response from 1.0 to 5.0 on relevance and directness.
Output structured JSON matching the schema."""),
    ("human", """User Question: {question}

Generated Answer:
{generated_answer}
""")
])


# =====================================================================
# RAG Evaluator Engine
# =====================================================================
class RAGEvaluator:
    """End-to-end RAG Evaluator for Multi-Document Search System"""

    def __init__(
        self,
        pdf_path: Optional[Path] = None,
        dataset_csv_path: Optional[Path] = None,
        vector_store: Optional[VectorStoreManager] = None
    ):
        self.pdf_path = pdf_path or (DATA_DIR / "Health_effects_of_air_pollution.pdf")
        self.dataset_csv_path = dataset_csv_path or (DATA_DIR / "evaluation_dataset" / "rag_evaluation_dataset_air_pollution.csv")
        
        logger.info("Initializing models & services...")
        self.llm_generator = Config.get_llm_generator()
        self.llm_checker = Config.get_llm_checker()
        self.embeddings = Config.get_embeddings()
        
        self.vector_store = vector_store or VectorStoreManager()
        self.doc_processor = DocumentProcessor(embeddings=self.embeddings, chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
        self.graph_builder = None
        self.langsmith_client = None

        # LLM Judges
        self.correctness_judge = CORRECTNESS_PROMPT | self.llm_checker.with_structured_output(AnswerCorrectnessJudge)
        self.groundedness_judge = GROUNDEDNESS_PROMPT | self.llm_checker.with_structured_output(GroundednessJudge)
        self.relevance_judge = RELEVANCE_PROMPT | self.llm_checker.with_structured_output(RelevanceJudge)

    def get_langsmith_client(self) -> Client:
        """Initializes or retrieves the LangSmith Client."""
        if self.langsmith_client is None:
            self.langsmith_client = Client(
                api_key=Config.LANGSMITH_API_KEY or os.getenv("LANGSMITH_API_KEY")
            )
        return self.langsmith_client

    def ingest_knowledge_document(self, force: bool = False) -> int:
        """
        Loads and indexes the target PDF (Health_effects_of_air_pollution.pdf) into Astra DB.
        """
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"Knowledge document not found at: {self.pdf_path}")

        logger.info("📄 Processing and chunking knowledge document: %s", self.pdf_path)
        chunks = self.doc_processor.load_from_pdf(str(self.pdf_path), strategy=ChunkStrategy.RECURSIVE)
        logger.info("Extracted %d chunks from PDF.", len(chunks))

        logger.info("💾 Ingesting chunks into Vector Store...")
        self.vector_store.create_vectorstore(chunks)
        logger.info("✅ Document ingestion completed successfully.")
        return len(chunks)

    def initialize_graph(self):
        """Initializes and builds the Adaptive LangGraph RAG workflow."""
        retriever = self.vector_store.get_retriever(k=Config.COHERE_RERANKER_TOP_N, search_type="similarity")
        self.graph_builder = GraphBuilder(
            retriever=retriever,
            llm_generator=self.llm_generator,
            llm_checker=self.llm_checker
        )
        self.graph_builder.build_graph()
        logger.info("Adaptive Graph initialized.")

    def load_dataset(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Loads the evaluation CSV dataset."""
        if not self.dataset_csv_path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found at: {self.dataset_csv_path}")

        logger.info("Loading evaluation dataset: %s", self.dataset_csv_path)
        df = pd.read_csv(self.dataset_csv_path)
        
        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]
        if "question" not in df.columns:
            raise ValueError("CSV must contain a 'question' column.")
        
        # Determine ground truth column
        if "true output" in df.columns:
            df["ground_truth"] = df["true output"]
        elif "ground_truth" not in df.columns and "answer" in df.columns:
            df["ground_truth"] = df["answer"]
        elif "ground_truth" not in df.columns:
            df["ground_truth"] = df.iloc[:, 1]  # fallback to second column

        if limit and limit > 0:
            df = df.head(limit)
            logger.info("Evaluating on top %d samples.", len(df))
        else:
            logger.info("Evaluating on all %d samples.", len(df))

        return df

    def compute_cosine_similarity(self, text_a: str, text_b: str) -> float:
        """Calculates cosine similarity between two text embeddings."""
        if not text_a or not text_b:
            return 0.0
        try:
            vecs = self.embeddings.embed_documents([text_a, text_b])
            vec_a, vec_b = np.array(vecs[0]), np.array(vecs[1])
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
        except Exception as e:
            logger.warning("Error computing embedding cosine similarity: %s", e)
            return 0.0

    async def evaluate_single_sample(self, question: str, ground_truth: str) -> Dict[str, Any]:
        """Runs a single test case through the RAG pipeline and performs evaluation."""
        start_time = time.time()
        
        # 1. Run Graph Workflow
        run_res = await self.graph_builder.run(question)
        latency = round(time.time() - start_time, 3)

        generated_answer = run_res.get("answer", "")
        retrieved_docs = run_res.get("retrieved_docs", [])
        tool_type = run_res.get("tool_type", "unknown")
        retrieval_grade = run_res.get("retrieval_grade", "unknown")
        hallucination_grade = run_res.get("hallucination_grade", "unknown")
        answer_relevance_grade = run_res.get("answer_relevance_grade", "unknown")
        rewrite_count = run_res.get("rewrite_count", 0)
        total_cost = run_res.get("total_cost", 0.0)

        context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No context retrieved."

        # 2. Semantic Embedding Similarity
        semantic_sim = self.compute_cosine_similarity(generated_answer, ground_truth)

        # 3. LLM-as-a-Judge: Correctness vs Ground Truth
        try:
            correctness_eval: AnswerCorrectnessJudge = await self.correctness_judge.ainvoke({
                "question": question,
                "ground_truth": ground_truth,
                "generated_answer": generated_answer
            })
            correctness_score = correctness_eval.score
            is_factual = correctness_eval.is_factual
            correctness_reasoning = correctness_eval.reasoning
        except Exception as e:
            logger.error("Error running correctness judge on '%s': %s", question[:30], e)
            correctness_score = 0.0
            is_factual = False
            correctness_reasoning = f"Evaluation failed: {e}"

        # 4. LLM-as-a-Judge: Faithfulness / Groundedness against Retrieved Context
        try:
            groundedness_eval: GroundednessJudge = await self.groundedness_judge.ainvoke({
                "question": question,
                "context": context_text,
                "generated_answer": generated_answer
            })
            is_grounded = groundedness_eval.is_grounded
            groundedness_reasoning = groundedness_eval.reasoning
        except Exception as e:
            logger.error("Error running groundedness judge on '%s': %s", question[:30], e)
            is_grounded = False
            groundedness_reasoning = f"Evaluation failed: {e}"

        # 5. LLM-as-a-Judge: Answer Relevance
        try:
            relevance_eval: RelevanceJudge = await self.relevance_judge.ainvoke({
                "question": question,
                "generated_answer": generated_answer
            })
            relevance_score = relevance_eval.score
            relevance_reasoning = relevance_eval.reasoning
        except Exception as e:
            relevance_score = 0.0
            relevance_reasoning = f"Evaluation failed: {e}"

        return {
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "semantic_similarity": round(semantic_sim, 4),
            "correctness_score_1_to_5": correctness_score,
            "is_factual": is_factual,
            "is_grounded": is_grounded,
            "relevance_score_1_to_5": relevance_score,
            "internal_retrieval_grade": retrieval_grade,
            "internal_hallucination_grade": hallucination_grade,
            "internal_relevance_grade": answer_relevance_grade,
            "tool_type": tool_type,
            "num_retrieved_chunks": len(retrieved_docs),
            "rewrite_count": rewrite_count,
            "latency_sec": latency,
            "total_cost_usd": total_cost,
            "correctness_reasoning": correctness_reasoning,
            "groundedness_reasoning": groundedness_reasoning,
            "relevance_reasoning": relevance_reasoning
        }

    # =====================================================================
    # LangSmith Dataset Creation & Evaluation Methods
    # =====================================================================
    def create_or_sync_langsmith_dataset(
        self,
        dataset_name: str = "Enhanced RAG Test Evaluation",
        limit: Optional[int] = None
    ) -> Any:
        """
        Creates or retrieves the LangSmith dataset and populates examples from the CSV.
        """
        client = self.get_langsmith_client()
        df = self.load_dataset(limit=limit)

        # Check if dataset already exists
        if client.has_dataset(dataset_name=dataset_name):
            logger.info("Found existing LangSmith dataset: '%s'", dataset_name)
            dataset = client.read_dataset(dataset_name=dataset_name)
        else:
            logger.info("Creating new LangSmith dataset: '%s'", dataset_name)
            dataset = client.create_dataset(
                dataset_name=dataset_name,
                description="Golden evaluation dataset for Air Pollution RAG benchmark"
            )

        # Format examples
        examples = [
            {
                "inputs": {"question": str(row["question"]).strip()},
                "outputs": {"ground_truth": str(row["ground_truth"]).strip()},
                "metadata": {
                    "source": "rag_evaluation_dataset_air_pollution.csv",
                    "sample_index": int(idx)
                }
            }
            for idx, row in df.iterrows()
        ]

        # Check existing examples count before creating
        existing_examples = list(client.list_examples(dataset_id=dataset.id))
        if len(existing_examples) < len(examples):
            logger.info("Uploading %d examples to LangSmith dataset '%s'...", len(examples), dataset_name)
            client.create_examples(dataset_id=dataset.id, examples=examples)
            logger.info("✅ Uploaded %d examples to LangSmith.", len(examples))
        else:
            logger.info("Dataset already contains %d examples. Skipping re-upload.", len(existing_examples))

        return dataset

    async def run_langsmith_evaluation(
        self,
        dataset_name: str = "Enhanced RAG Test Evaluation",
        experiment_prefix: str = "rag-doc-relevance",
        metadata: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None
    ) -> Any:
        """
        Runs LangSmith evaluate() on the dataset with custom LLM-as-a-judge evaluators.
        """
        if self.graph_builder is None:
            self.initialize_graph()

        client = self.get_langsmith_client()
        # Ensure dataset exists in LangSmith
        self.create_or_sync_langsmith_dataset(dataset_name=dataset_name, limit=limit)

        # 1. Define Target function
        async def target(inputs: dict) -> dict:
            question = inputs.get("question") or inputs.get("input", "")
            res = await self.graph_builder.run(question)
            retrieved_docs = res.get("retrieved_docs", [])
            context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else ""
            return {
                "answer": res.get("answer", ""),
                "context": context_text,
                "retrieved_chunks_count": len(retrieved_docs),
                "tool_type": res.get("tool_type", "unknown"),
                "total_cost": res.get("total_cost", 0.0),
                "retrieval_grade": res.get("retrieval_grade"),
                "hallucination_grade": res.get("hallucination_grade"),
                "answer_relevance_grade": res.get("answer_relevance_grade"),
                "rewrite_count": res.get("rewrite_count", 0)
            }

        # 2. Define Evaluators
        async def correctness_evaluator(run, example) -> dict:
            question = example.inputs.get("question", "")
            ground_truth = example.outputs.get("ground_truth", "")
            generated_answer = run.outputs.get("answer", "")
            try:
                eval_res = await self.correctness_judge.ainvoke({
                    "question": question,
                    "ground_truth": ground_truth,
                    "generated_answer": generated_answer
                })
                return {
                    "key": "correctness",
                    "score": eval_res.score / 5.0,  # Normalized 0.0 - 1.0
                    "comment": f"Score: {eval_res.score}/5.0 | {eval_res.reasoning}"
                }
            except Exception as e:
                return {"key": "correctness", "score": 0.0, "comment": f"Error: {e}"}

        async def groundedness_evaluator(run, example) -> dict:
            question = example.inputs.get("question", "")
            context = run.outputs.get("context", "")
            generated_answer = run.outputs.get("answer", "")
            try:
                eval_res = await self.groundedness_judge.ainvoke({
                    "question": question,
                    "context": context,
                    "generated_answer": generated_answer
                })
                return {
                    "key": "groundedness",
                    "score": 1.0 if eval_res.is_grounded else 0.0,
                    "comment": eval_res.reasoning
                }
            except Exception as e:
                return {"key": "groundedness", "score": 0.0, "comment": f"Error: {e}"}

        async def relevance_evaluator(run, example) -> dict:
            question = example.inputs.get("question", "")
            generated_answer = run.outputs.get("answer", "")
            try:
                eval_res = await self.relevance_judge.ainvoke({
                    "question": question,
                    "generated_answer": generated_answer
                })
                return {
                    "key": "relevance",
                    "score": eval_res.score / 5.0,
                    "comment": eval_res.reasoning
                }
            except Exception as e:
                return {"key": "relevance", "score": 0.0, "comment": f"Error: {e}"}

        async def retrieval_relevance_evaluator(run, example) -> dict:
            retrieval_grade = run.outputs.get("retrieval_grade")
            num_chunks = run.outputs.get("retrieved_chunks_count", 0)
            score = 1.0 if retrieval_grade == "yes" and num_chunks > 0 else 0.0
            return {
                "key": "retrieval_relevance",
                "score": score,
                "comment": f"Internal retrieval grade: {retrieval_grade}, Chunks: {num_chunks}"
            }

        eval_metadata = metadata or {
            "version": "Adaptive RAG Graph",
            "generator_model": Config.LLM_MODEL_GENERATOR,
            "checker_model": Config.LLM_MODEL_CHECKER,
            "embedding_model": Config.EMBEDDING_MODEL,
            "document": "Health_effects_of_air_pollution.pdf"
        }

        print("\n" + "="*70)
        print(f"🚀 Running LangSmith Evaluation on dataset: '{dataset_name}'")
        print(f"   Experiment Prefix: '{experiment_prefix}'")
        print("="*70)

        experiment_results = evaluate(
            target,
            data=dataset_name,
            evaluators=[
                correctness_evaluator,
                groundedness_evaluator,
                relevance_evaluator,
                retrieval_relevance_evaluator
            ],
            experiment_prefix=experiment_prefix,
            metadata=eval_metadata,
            client=client,
            max_concurrency=1
        )

        print("\n✅ LangSmith Evaluation completed successfully!")
        return experiment_results

    # =====================================================================
    # Local Evaluation Runner
    # =====================================================================
    async def run_evaluation(
        self,
        limit: Optional[int] = None,
        output_csv_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """Executes full evaluation dataset and generates local metric summary."""
        df = self.load_dataset(limit=limit)
        results = []

        print("\n" + "="*70)
        print(f"🚀 Starting Local RAG Evaluation on {len(df)} questions...")
        print("="*70)

        for idx, row in df.iterrows():
            q = str(row["question"]).strip()
            gt = str(row["ground_truth"]).strip()
            
            print(f"\n[{idx + 1}/{len(df)}] Evaluating Question: {q}")
            res = await self.evaluate_single_sample(q, gt)
            results.append(res)
            
            print(f"  ➜ Score (1-5): {res['correctness_score_1_to_5']} | Semantic Sim: {res['semantic_similarity']:.3f} | Grounded: {res['is_grounded']} | Latency: {res['latency_sec']}s")

        results_df = pd.DataFrame(results)

        # Output file paths
        out_csv = output_csv_path or (REPO_ROOT / "tests" / "evaluation" / f"eval_results_air_pollution_{int(time.time())}.csv")
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"\n💾 Saved full itemized evaluation results to:\n   {out_csv}")

        self.print_summary_report(results_df)
        return results_df

    def print_summary_report(self, df: pd.DataFrame):
        """Prints a structured summary report of the evaluation metrics."""
        avg_score = df["correctness_score_1_to_5"].mean()
        avg_sim = df["semantic_similarity"].mean()
        factual_rate = (df["is_factual"].sum() / len(df)) * 100
        grounded_rate = (df["is_grounded"].sum() / len(df)) * 100
        avg_latency = df["latency_sec"].mean()
        total_cost = df["total_cost_usd"].sum()
        avg_chunks = df["num_retrieved_chunks"].mean()

        print("\n" + "="*70)
        print("📊 RAG EVALUATION BENCHMARK SUMMARY REPORT")
        print("="*70)
        print(f" Total Samples Evaluated    : {len(df)}")
        print(f" Mean Correctness Score     : {avg_score:.2f} / 5.00  ({(avg_score/5.0)*100:.1f}%)")
        print(f" Mean Semantic Similarity   : {avg_sim:.4f}")
        print(f" Factual Accuracy Rate      : {factual_rate:.1f}%")
        print(f" Faithfulness / Groundedness: {grounded_rate:.1f}%")
        print(f" Avg Retrieved Chunks       : {avg_chunks:.1f}")
        print(f" Avg Latency per Query      : {avg_latency:.2f} seconds")
        print(f" Total Evaluation Cost      : ${total_cost:.5f} USD")
        print("="*70)


# =====================================================================
# Main Execution CLI
# =====================================================================
async def main():
    parser = argparse.ArgumentParser(description="Run RAG Evaluation for Air Pollution Dataset")
    parser.add_argument("--limit", type=int, default=None, help="Number of questions to evaluate (default: all)")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip document ingestion step if already indexed")
    parser.add_argument("--output", type=str, default=None, help="Custom output CSV path for local eval")
    parser.add_argument("--langsmith", action="store_true", help="Run evaluation through LangSmith and log experiment")
    parser.add_argument("--dataset-name", type=str, default="Enhanced RAG Test Evaluation", help="LangSmith dataset name")
    parser.add_argument("--experiment-prefix", type=str, default="rag-doc-relevance", help="LangSmith experiment prefix")
    args = parser.parse_args()

    evaluator = RAGEvaluator()

    # Step 1: Ingest Document
    if not args.skip_ingest:
        print("\n[Step 1/3] Ingesting Knowledge Document: Health_effects_of_air_pollution.pdf...")
        evaluator.ingest_knowledge_document()
    else:
        print("\n[Step 1/3] Skipping document ingestion (--skip-ingest specified)...")

    # Step 2: Initialize RAG Graph
    print("\n[Step 2/3] Initializing Graph Builder and Retrievers...")
    evaluator.initialize_graph()

    # Step 3: Run Evaluation (LangSmith or Local)
    if args.langsmith:
        print(f"\n[Step 3/3] Running LangSmith Evaluation on '{args.dataset_name}'...")
        await evaluator.run_langsmith_evaluation(
            dataset_name=args.dataset_name,
            experiment_prefix=args.experiment_prefix,
            limit=args.limit
        )
    else:
        print("\n[Step 3/3] Running Local Evaluation on dataset...")
        output_path = Path(args.output) if args.output else None
        await evaluator.run_evaluation(limit=args.limit, output_csv_path=output_path)


if __name__ == "__main__":
    asyncio.run(main())
