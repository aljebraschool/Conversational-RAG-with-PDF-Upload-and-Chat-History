
#load the needed libraries
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv

load_dotenv()

#load huggingface api key
hf_token = os.getenv("HF_TOKEN")
#use the huggingface api key to get embedding
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2') 

## set up the streamlit
st.title("Conversational RAG with PDF upload and chat history")
st.write("Upload PDF to chat with")


#input your Groq Api key
groq_api_key = st.text_input("Enter your Groq API key ", type = 'password')

#validate the api key
if groq_api_key:
    model = ChatGroq(groq_api_key = groq_api_key, model_name = 'gemma2-9b-it' )

    #Chat interface
    #to keep track of chat create a session id
    session_id = st.text_input("Session ID", value = "default")

    #manage your chat history with the help of session state

    #create a variable called store to manage the session state
    if "store" not in st.session_state:
        #initialize the variable "store" to empty dictionary to keep track of the state
        st.session_state.store = {}

    #create your upload button on streamlit
    uploaded_files = st.file_uploader("Choose the PDF file to upload", type = "pdf", accept_multiple_files = True)

    #process the uploaded files
    if uploaded_files:
        #create an empty list of documents to collect the pages in the pdf
        documents = []
        
        #go through each file in the pdf file
        for uploaded_file in uploaded_files:
            #create a temporary folder to store the uploded files
            temp_pdf_folder = f"./temp.pdf"

            #open the folder to write the uploded pdf file into it
            with open(temp_pdf_folder, "wb") as file:
                #now write each of the uploaded files in the folder
                file.write(uploaded_file.getvalue())
                #also get name of each file
                file_name = uploaded_file.name

            #load the pdf folder using PyPDFLoader
            loader = PyPDFLoader(temp_pdf_folder)

            #convert the loader file to document
            docs = loader.load()

            #put each loader docs file in the document list above
            documents.extend(docs)

        #split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
        #use the RecursiveCharacterTextspliter object above on the document
        split_document = text_splitter.split_documents(documents)
        #now pass this split to vector store as embedding
        vector_store = FAISS.from_documents(split_document, embeddings)
        # vector_store = Chroma.from_documents(documents = split_document, embedding = embeddings)
        #get the vectore store using retrieval
        retriever = vector_store.as_retriever()


        #create context prompt for the model
        contextualized_q_system_prompt = (
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
        #Now create a prompt template for the model
        contextualized_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ('system', contextualized_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ('human', "{input}")
                ]
            )
        history_aware_retriever = create_history_aware_retriever(
                    model, retriever, contextualized_q_prompt
                )
        
        #Now a question-answer prompt that is based on the context
        system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}"
            )
        
        qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ('system', system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}")
                ]
            )
        question_answer_chain = create_stuff_documents_chain(
            model, qa_prompt
        )

        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
                if session_id not in st.session_state.store:
                    st.session_state.store[session_id] = ChatMessageHistory()
                return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
        user_input = st.text_input("Your question: ")

        if user_input:
                # Create session history to keep track of chat history
                session_history = get_session_history(session_id)
                # Get response from your chain
                response = conversational_rag_chain.invoke(
                    {'input': user_input},
                        config={'configurable': {'session_id': session_id}}
                    )

                st.write(st.session_state.store)
                st.success(response['answer'])
                st.write("chat_history", session_history.messages)
else:
    st.warning("Please enter a valid GROQ API key ")

    
