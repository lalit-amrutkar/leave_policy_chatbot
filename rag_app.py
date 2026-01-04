from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# Load embeddings (local, free)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS index
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Local LLM via Ollama (NO internet, NO API key)
llm = Ollama(
    model="mistral",
    temperature=0
)

# RAG chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 3})
)

while True:
    q = input("\nAsk question (exit to quit): ")
    if q.lower() == "exit":
        break
    print("\nAnswer:", qa.run(q))
