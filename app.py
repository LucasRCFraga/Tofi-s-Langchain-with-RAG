from langchain import hub
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.load import dumps, loads
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

#Question for the RAG-Fusion
question = "Give me a detailed description of the characteristics of the main characters in this book"

#### RETRIEVAL and GENERATION ####

#Custom Prompt
template_prompt = """ Answer the question based only on the following context:
{context}

Question: {question}
"""
custom_prompt = ChatPromptTemplate.from_template(template_prompt)

#--------------------------------------------------------------------------------------------------#

#Rag-Fusion Prompt
template = """
You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):
"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)

generate_queries = (
    prompt_rag_fusion 
    | ChatOpenAI(temperature=0)
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)

def reciprocal_rank_fusion(results: list[list], k=60):
    """ Reciprocal_rank_fusion that takes multiple lists of ranked documents 
        and an optional parameter k used in the RRF formula """
    
    # Initialize a dictionary to hold fused scores for each unique document
    fused_scores = {}

    # Iterate through each list of ranked documents
    for docs in results:
        # Iterate through each document in the list, with its rank (position in the list)
        for rank, doc in enumerate(docs):
            # Convert the document to a string format to use as a key (assumes documents can be serialized to JSON)
            doc_str = dumps(doc)
            # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            # Retrieve the current score of the document, if any
            previous_score = fused_scores[doc_str]
            # Update the score of the document using the RRF formula: 1 / (rank + k)
            fused_scores[doc_str] += 1 / (rank + k)

    # Sort the documents based on their fused scores in descending order to get the final reranked results
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    # Return the reranked results as a list of tuples, each containing the document and its fused score
    return reranked_results



#--------------------------------------------------------------------------------------------------#

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
#rag_chain.invoke("Create a comprehensive list of the main topics in a way that makes a potential user want to read the document, the list should be 5 topics long")


retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
docs = retrieval_chain_rag_fusion.invoke({"question": question})

# RAG
template = """Answer the following question based on this context:

{context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

final_rag_chain = (
    {"context": retrieval_chain_rag_fusion, 
     "question": itemgetter("question")} 
    | prompt
    | llm
    | StrOutputParser()
)

final_rag_chain.invoke({"question":question})

"""
#Cursor provided code
if uploaded_pdf is not None:
    loader = PyPDFLoader(uploaded_pdf)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
"""


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
