# rag_chain.py — ScientificAnimals
# This module implements a Retrieval-Augmented Generation (RAG) pipeline
# that retrieves information from Wikipedia and arXiv and uses a Hugging Face LLM to answer user questions.

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.document_loaders import WikipediaLoader, ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Step 1: Load documents from external sources
# Wikipedia provides general knowledge and arXiv provides scientific literature.
def load_documents(animal):
    wiki_loader = WikipediaLoader(query=animal, lang="en")
    wiki_docs = wiki_loader.load()

    arxiv_loader = ArxivLoader(query=animal, load_max_docs=2)  # Limit to top 2 results for performance
    arxiv_docs = arxiv_loader.load()

    return wiki_docs + arxiv_docs

# Step 2: Split the text into chunks
# This is essential for creating meaningful embeddings for retrieval.
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    return splitter.split_documents(documents)

# Step 3: Create a vector store from document chunks
# FAISS is used here for efficient similarity search.
# We use sentence-transformers for semantic embedding.
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

# Step 4: Set up the RAG pipeline and return the answer
# This function will be called from app.py to get the scientific name.
def get_scientific_name(question):
    # Extract the animal name from the question string
    animal = question.split("animal")[-1].strip(" ?")

    # Load and process knowledge sources
    documents = load_documents(animal)
    chunks = split_documents(documents)
    vectorstore = create_vectorstore(chunks)

    # Use a pre-trained model from Hugging Face to perform question answering
    # flan-t5-base is a general-purpose, instruction-following model
    llm = HuggingFaceHub(repo_id="google/flan-t5-base", model_kwargs={"temperature": 0.2, "max_length": 256})

    # Combine the retriever and generator into a RAG chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    # Run the chain to generate an answer
    result = qa_chain.run(question)
    return result
