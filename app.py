# pip install python-dotenv // for reading .env file
# streamlit for GUI
import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
#from langchain.chat_models import ChatOpenAI
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, user_template, bot_template


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text
    
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    #embeddings = OpenAIEmbeddings()
    # ALternate for OPENAPIEmbedding Srv which is paid, use huggingface embedding
    #https://huggingface.co/spaces/mteb/leaderboard
    #https://huggingface.co/hkunlp/instructor-xl
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore =FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    #llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory= ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm,
                                                               retriever=vectorstore.as_retriever(),
                                                               memory=memory)
    return conversation_chain 

def handle_user_question(user_question):
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)    


def main():
    print('Hare Krshna app.py')
    load_dotenv()
    print("this comes from .env >>", os.getenv("OPENAI_API_KEY"))
    st.set_page_config(page_title="Ask Your Questions About TS Payments", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    st.header("Ask Your Questions About TS Payments 💬")
    user_question = st.text_input("Ask a question about TS Payments APIs:")

    if user_question:
        handle_user_question(user_question)

    # st.write(user_template.replace("{{MSG}}","hello Robot"), unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}","hello HUman"), unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    with st.sidebar:
        #st.subheader("TS Payments API Documents")
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload the Payments API Documents and click 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                
                # Read the content of pdfs
                raw_text = get_pdf_text(pdf_docs)
                #st.write(raw_text)
                # break the pdf contents into chunks of specific size and overlap
                chunks = get_text_chunks(raw_text)
                #st.write(chunks)
                # persist the chunks as vector(number 1010100 etc) into a vector DB like FASSI
                #instructor-embeddings.github.io
                #huggingface.co/spaces/mteb/leaderboard
                # OPENAPI 
                vectorstore = get_vectorstore(chunks)
                print("HB")
                print("RR")
                # create Conversation Chain, to avoid reloading of conversation when Streamlit reload for anyreason keep it in session
                st.session_state.conversation= get_conversation_chain(vectorstore)





if __name__=='__main__':
    main()