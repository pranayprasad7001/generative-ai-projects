import os
import bs4
import hashlib
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ["USER_AGENT"] = "MyRAGApp/1.0"

def load_docs_from_urls(urls):
    """
    Loads documents from URLs and splits them into chunks
    """
    if not urls:
        return []

    loader = WebBaseLoader(
        web_paths=urls,
        header_template={"User-Agent": "MyRAGApp/1.0"}, 
        bs_kwargs={
            "parse_only": bs4.SoupStrainer(
            ["article", "main", "div"], 
            class_=["content", "post-text", "body", "post-content"] 
            )
        },
        bs_get_text_kwargs={
            "separator": " ", 
            "strip": True
        }
    )
    
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,   
        chunk_overlap=100, 
        separators=["\n\n", "\n", " ", ""]
    )

    splitted_docs = text_splitter.split_documents(docs)
    
    print(f"Loaded {len(docs)} source documents. Created {len(splitted_docs)} total chunks.")
    return splitted_docs


def embed_documents(splitted_docs, vector_store):
    """
    Embeds documents and prevents duplicates using deterministic hashing IDs.
    """
    if not splitted_docs:
        return vector_store

    # Generate unique deterministic IDs based on content and source
    ids = []
    for doc in splitted_docs:
        source = doc.metadata.get("source", "")
        content = doc.page_content
        # Combine source URL and content chunk to form a unique hash key
        unique_string = f"{source}_{content}"
        doc_id = hashlib.md5(unique_string.encode("utf-8")).hexdigest()
        ids.append(doc_id)

    # Add documents using explicit IDs (Upserts existing records)
    vector_store.add_documents(splitted_docs, ids=ids)
    
    print(f"Processed {len(splitted_docs)} chunks into vectorstore safely!")
    return vector_store


def vectorstore_to_retriever(vector_store):
    """
    Converts vectorstore to retriever
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    print(f"Vectorstore converted to retriever successfully!")
    return retriever