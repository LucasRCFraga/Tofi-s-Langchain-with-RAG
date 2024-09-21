import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# Load PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text 

# Get text chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Embed
def get_embed_text(splits):
    vectorstore = Chroma.from_texts(texts=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    return retriever

# Conversation memory
def get_conversation_memory(retriever):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
        retriever=retriever,
        memory=memory
    )
    return conversation_chain

def main():
    load_dotenv()
    st.title("Chatbot utilizing RAG")
    st.caption("Talk to Tofi, your friendly chatbot")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    user_input = st.text_input("Enter your question:")

    pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)
    button = st.button("Submit")

    if button:
        if pdf_docs is None or user_input is None:
            st.warning("Please upload a PDF file to continue.")
            st.stop()

        if pdf_docs is not None and user_input is not None:
            with st.spinner("Processing..."):
                # Get the pdf text
                raw_pdf_text = get_pdf_text(pdf_docs)

                # Get the text chunks
                text_chunks = get_text_chunks(raw_pdf_text)
                #st.write(text_chunks)
                
                # Embed the text chunks
                vectorstore = get_embed_text(text_chunks)
                
                # Creating the conversation Chain
                st.session_state.conversation = get_conversation_memory(vectorstore)
                
                # Displaying the conversation
                st.write(st.session_state.conversation.invoke({"question": user_input}))

if __name__ == "__main__":
    main()
