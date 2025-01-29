from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import os
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import shutil
from uuid import uuid4
from io import BytesIO
import time  # Add this import
from PyPDF2 import PdfReader  # Add this import
from langchain_community.document_loaders.base import BaseBlobParser
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec  # Update import
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from dotenv import load_dotenv

load_dotenv()

# Define constants
index_name = 'docchat-index'

# Initialize Pinecone
pc = Pinecone(
    api_key=os.environ["PINECONE_API_KEY"]
)

# Create or get index
try:
    # Check if index exists
    if index_name not in pc.list_indexes().names():
        # Create index if it doesn't exist
        pc.create_index(
            name=index_name,
            dimension=384,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        print(f"Created new index: {index_name}")
    else:
        print(f"Found existing index: {index_name}")

    # Get the index
    index = pc.Index(index_name)
    print(f"Successfully connected to index: {index_name}")
except Exception as e:
    print(f"Error with Pinecone setup: {str(e)}")
    raise e

# Initialize global variables
sessions: Dict[str, ChatMessageHistory] = {}
vectorstores: Dict[str, LangchainPinecone] = {}
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://doc-chat-ai-xbde.vercel.app/",  
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

@app.post("/start-session")
async def start_session():
    session_id = str(uuid4())
    sessions[session_id] = ChatMessageHistory()
    return {"session_id": session_id}

# Add this custom loader class
class BytesIOPDFLoader:
    def __init__(self, file_content):
        self.file_content = file_content

    def load(self):
        pdf_reader = PdfReader(self.file_content)
        documents = []
        
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text.strip():  # Only create document if there's text content
                doc = Document(
                    page_content=text,
                    metadata={
                        "page_number": page_num + 1,
                        "total_pages": len(pdf_reader.pages),
                    }
                )
                documents.append(doc)
                
        return documents

# Update the upload endpoint
@app.post("/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    session_id: str = Form(...),
):
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    
    try:
        documents = []
        
        for uploaded_file in files:
            try:
                content = await uploaded_file.read()
                pdf_file = BytesIO(content)
                
                loader = BytesIOPDFLoader(pdf_file)
                docs = loader.load()
                documents.extend(docs)
                
            except Exception as file_error:
                print(f"Error processing file {uploaded_file.filename}: {str(file_error)}")
                raise HTTPException(status_code=400, detail=f"Error processing file: {str(file_error)}")
        
        if not documents:
            raise HTTPException(status_code=400, detail="No valid documents were processed")
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        
        vectorstores[session_id] = LangchainPinecone.from_documents(
            documents=splits,
            embedding=embeddings,
            index_name=index_name,
            namespace=session_id
        )
        
        return {"message": "Files processed successfully"}
        
    except Exception as e:
        print(f"Unexpected error in upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in sessions:
        sessions[session_id] = ChatMessageHistory()
    return sessions[session_id]

@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    
    try:
        if request.session_id not in vectorstores:
            vectorstores[request.session_id] = LangchainPinecone.from_existing_index(
                index_name=index_name,
                embedding=embeddings,
                namespace=request.session_id
            )
        
        retriever = vectorstores[request.session_id].as_retriever()
        
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        response = conversational_rag_chain.invoke(
            {"input": request.message},
            config={"configurable": {"session_id": request.session_id}},
        )
        
        return {"answer": response['answer']}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add cleanup endpoint
@app.delete("/session/{session_id}")
async def cleanup_session(session_id: str):
    # Only cleanup session history
    if session_id in sessions: 
        del sessions[session_id]
    if session_id in vectorstores:
        del vectorstores[session_id]
    
    return {"message": "Session cleaned up successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
