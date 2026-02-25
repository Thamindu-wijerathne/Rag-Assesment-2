
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

_embeddings = None


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


# this make remember innerfunction about db
def make_retrieve_fn(vs):
    def _retrieve(query: str, k: int = 4):
        docs = vs.similarity_search(query, k=k)
        context = "\n\n".join(
            f"[source={d.metadata.get('source','doc')} page={d.metadata.get('page','?')}]\n{d.page_content}"
            for d in docs
        )
        sources = [{"source": d.metadata.get("source"), "page": d.metadata.get("page")} for d in docs]
        return context, sources
    return _retrieve