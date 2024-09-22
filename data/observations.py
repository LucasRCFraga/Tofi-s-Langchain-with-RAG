"""
    #Cursor provided code for one function that was not used, but worth keeping in mind.
    if uploaded_pdf is not None:
        loader = PyPDFLoader(uploaded_pdf)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever()

    Observation temperature 0 | 0.5 | 1 and premade versus custom prompt, these answers are based on the RAG without applying RAG-Fusion to it.
        Additionally, the answers are based on a personal book I wrote when I was younger and never published, so it's the best way
        I found to test the RAG-Fusion at the time.
    
    (These answers were based on the rlm/rag-prompt)
        rlm/rag-prompt: You are an assistant for question-answering tasks. Use the following pieces 
        of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        
    User Prompt: Can you make me a comprehensive list of the main topics discussed in this document?
    
        Answer with 0 temp: The main topics discussed in this document include leadership changes 
            within a guild, a significant announcement made during a meeting, and the initiation of 
            training sessions with four masters of different combat styles.
        
        Answer with 0.5 temp: The main topics discussed in the document include training with different weapons 
            and martial arts styles, the presentation of important members of the continent, the Grã-Mestre being a 
            strong female warrior, and a violent confrontation at the citadel.
            
        Answer with 1 temp: The main topics discussed in the document are the training session with four masters 
            of the continent, the choice between staying for the test or leaving the colosseum, and the upcoming presentation 
            where important members of the continent will speak. The document also mentions the presence of powerful 
            individuals like the Grã-Mestre, Nara, and the distribution of food, drink, and first aid instructions before 
            a farewell message. 
"""