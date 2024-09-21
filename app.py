from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()

# Load PDF
loader = PyPDFLoader(
    "./example_data/a2.pdf",
)
docs = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Embed
vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever()
    #Default value for search_kwargs is {"k": 4}

#### RETRIEVAL and GENERATION ####

#Custom Prompt
template_prompt = """ Answer the question based only on the following context:
{context}

Question: {question}
"""

custom_prompt = ChatPromptTemplate.from_template(template_prompt)

# Prompt
#prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = ChatOpenAI(temperature=0.5)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_prompt
    | llm
    | StrOutputParser()
)

# Question
rag_chain.invoke("Can you make me a comprehensive list of the main topics discussed in this document?")


"""
    Observation temperature 0 | 0.5 | 1 and premade versus custom prompt
    
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
