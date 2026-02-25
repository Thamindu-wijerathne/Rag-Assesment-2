import os
import uuid
import tempfile
from typing import Dict, Any
import json

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from groq import AsyncGroq

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from helper import _post_callback, _safe_json_extract
from llm_calls import generate_answer_from_context, guardrail_classify, RETRIEVE_TOOL


app = FastAPI(title="RAG Agent API - Phase 1")

groq_client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

@app.get("/")
async def health():
    return {"status": "ok"}

# --- In-memory "temporary DB" ---
# doc_id -> {"vectorstore": FAISS, "meta": {...}}
DOC_STORE: Dict[str, Dict[str, Any]] = {}

class UploadResponse(BaseModel):
    document_id: str
    pages_loaded: int

class QueryRequest(BaseModel):
    document_id: str
    question: str
    callback_url: HttpUrl
    top_k: int = 4
    
class AckResponse(BaseModel):
    status: str
    job_id: str

_embeddings = None


def retrieve_context(vs, query: str, k: int = 4):
    docs = vs.similarity_search(query, k=k)
    context = "\n\n".join(
        f"[source={d.metadata.get('source','doc')} page={d.metadata.get('page','?')}]\n{d.page_content}"
        for d in docs
    )
    sources = [{"source": d.metadata.get("source"), "page": d.metadata.get("page")} for d in docs]
    return context, sources


async def agentic_rag_answer(vs, question: str, top_k: int = 4):
    print("agent called")

    system = (
        "You are a RAG agent. You MAY call the tool `retrieve_context` to fetch relevant text. "
        "If you need more info, call it again (possibly with a rewritten query). "
        "When you have enough context, answer clearly and cite sources if possible."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]

    sources_accum = []

    # force at least one retrieval for true RAG:
    tool_choice = "required" 

    # only 4 loops for now 
    for _ in range(4): 
        resp = await groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=[RETRIEVE_TOOL],
            tool_choice=tool_choice, 
            temperature=0,
            max_tokens=800,
        )

        msg = resp.choices[0].message
        print("msg in loop :", msg)
        print("\n")
        tool_calls = getattr(msg, "tool_calls", None)

        if tool_calls:
            # Execute each tool call locally
            for call in tool_calls:
                name = call.function.name
                args = json.loads(call.function.arguments or "{}")

                if name == "retrieve_context":
                    q = args.get("query", question)
                    k = int(args.get("k", top_k))
                    ctx, src = retrieve_context(vs, q, k=k)
                    sources_accum.extend(src)

                    # Add tool result back to the conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": name,
                        "content": ctx
                    })

            # after first forced retrieval, let it decide
            tool_choice = "auto"
            continue

        # No tool calls => final answer
        final_text = msg.content or ""
        print("\nfinal text :", final_text)
        return final_text, sources_accum

    return "I'm sorry, I couldn't complete the request in time.", sources_accum

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings

def _split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return splitter.split_documents(docs)

async def _run_rag_and_callback(job_id: str, document_id: str, question: str, callback_url: str, top_k: int):
    print("_run_rag_and_callback", job_id, document_id, question, callback_url, top_k)
    try:
        # checking is it valid quection
        guard = await guardrail_classify(question) 

        if guard["status"] == "INVALID":
            payload= {
                "job_id": job_id,
                "document_id": document_id,
                "question": question,
                "answer": guard["reply"],
                "status": guard["status"],
                }
            await _post_callback(str(callback_url), payload)
            return

        if guard["status"] == "GREETING":
            payload= {
                "job_id": job_id,
                "document_id": document_id,
                "question": question,
                "answer": guard["reply"],
                "status": "Greeting",
                }
            await _post_callback(str(callback_url), payload)
            return

        sanitized_q = guard.get("sanitized_request") or question
        print("sanitized_q :", sanitized_q)

        if document_id not in DOC_STORE:
            raise ValueError("Unknown document_id. Upload first.")

        vs: FAISS = DOC_STORE[document_id]["vectorstore"]

        answer, sources = await agentic_rag_answer(vs, sanitized_q, top_k=top_k)

        payload = {
            "job_id": job_id,
            "document_id": document_id,
            "question": question,
            "answer": answer,
            "sources": sources,
            "status": "completed",
        }
        await _post_callback(str(callback_url), payload)

    except Exception as e:
        payload = {
            "job_id": job_id,
            "document_id": document_id,
            "question": question,
            "error": str(e),
            "status": "failed",
        }
        # best-effort callback even on failure
        try:
            await _post_callback(str(callback_url), payload)
        except Exception:
            pass


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    filename = (file.filename or "").lower()

    if not (filename.endswith(".pdf") or filename.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Only PDF or TXT files are supported.")

    doc_id = str(uuid.uuid4())

    # Save to temp file
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, file.filename or f"{doc_id}.pdf")
        content = await file.read()
        with open(path, "wb") as f:
            f.write(content)

        if filename.endswith(".txt"):
            text = content.decode("utf-8", errors="ignore")
            # print(text)
            # Wrap as a single "Document-like" structure via LangChain docs format
            docs = [Document(page_content=text, metadata={"source": file.filename, "page": 1})]
        else:
            loader = PyMuPDFLoader(path)
            docs = loader.load()

        chunks = _split_docs(docs)

        # Build FAISS index in-memory
        vs = FAISS.from_documents(chunks, embedding=get_embeddings())

    DOC_STORE[doc_id] = {
        "vectorstore": vs,
        "meta": {"filename": file.filename, "chunks": len(vs.index_to_docstore_id)},
    }

    print("ntotal:", vs.index.ntotal)
    print("dimension:", vs.index.d)
    ids = list(vs.index_to_docstore_id.values())
    print("first 5 ids:", ids[:5])


    pages_loaded = len(docs)
    return UploadResponse(document_id=doc_id, pages_loaded=pages_loaded)


@app.post("/query", response_model=AckResponse)
async def query(req: QueryRequest, background_tasks: BackgroundTasks):
    if req.document_id not in DOC_STORE:
        raise HTTPException(status_code=404, detail="document_id not found. Upload first.")
    
    job_id = str(uuid.uuid4())

    # Immediate ACK, do work in background
    background_tasks.add_task(
        _run_rag_and_callback,
        job_id,
        req.document_id,
        req.question,
        str(req.callback_url),
        req.top_k,
    )

    return AckResponse(status="accepted", job_id=job_id)