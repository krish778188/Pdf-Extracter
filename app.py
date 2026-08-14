import os
import chainlit as cl
import asyncio
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

load_dotenv()

# Define the instruction globally
system_instruction = (
    "You are a helpful AI assistant. You must answer the user's question "
    "using ONLY the provided context below. "
    "If the answer is not contained within the context, you must strictly say: "
    "'I cannot find the answer in the document.' Do not make up an answer.\n\n"
    "Context:\n{context}"
)

@cl.on_chat_start
async def on_chat_start():
    # 1. Prompt the user to upload a PDF file
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin chatting!",
            accept=["application/pdf"],
            max_size_mb=20, # Prevents users from uploading massive files
            timeout=180,
        ).send()

    file = files[0]
    
    # 2. Show a loading message in the UI
    msg = cl.Message(content=f"Processing `{file.name}`... Please wait.")
    await msg.send()

    # 3. Process the uploaded PDF (Extract & Split)
    # Chainlit automatically saves the uploaded file to a temporary path
    loader = PyPDFLoader(file.path)
    docs = loader.load()
    docs = [doc for doc in docs if doc.page_content]
    
    if os.path.exists(file.path):
        os.remove(file.path)
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    chunks = text_splitter.split_documents(docs)

    # 4. Initialize Models
    llm = ChatMistralAI(model="mistral-large-latest", temperature=0)
    
    # USE LOCAL EMBEDDINGS (Instant, no rate limits!)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    msg.content = f"Building database for `{file.name}` locally... This will be fast!"
    await msg.update()

    # 5. Create the IN-MEMORY Vector Store Instantly
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    retriever = vector_store.as_retriever(
    search_type="mmr",  # Changed from "similarity"
    search_kwargs={
        "k": 15,        # The final number of diverse chunks sent to Mistral
        "fetch_k": 50   # The initial pool of chunks to fetch in the background
    }
)
    
    # 6. Build the Chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("human", "{input}")
    ])
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # 7. Store the chain in this specific user's session
    cl.user_session.set("retrieval_chain", retrieval_chain)
    
    # 8. Update the UI to let them know it's ready
    msg.content = f"Successfully processed `{file.name}`! What would you like to know about it?"
    await msg.update()

@cl.on_message
async def on_message(message: cl.Message):
    # NEW: Check if the user wants to start a new file
    if message.content.strip().lower() in ["new", "reset", "restart", "clear"]:
        # Wipe the current chain from memory
        cl.user_session.set("retrieval_chain", None)
        await cl.Message(content="🧹 Context cleared! Restarting session...").send()
        # Trigger the start function again to ask for a new PDF
        await on_chat_start()
        return

    # Normal Chat Logic
    chain = cl.user_session.get("retrieval_chain")
    
    # If the user somehow bypassed the upload, prevent an error
    if chain is None:
        await cl.Message(content="Please upload a PDF first!").send()
        return
    
    msg = cl.Message(content="")
    await msg.send()
    
    res = await chain.ainvoke({"input": message.content})
    
    msg.content = res["answer"]
    await msg.update()