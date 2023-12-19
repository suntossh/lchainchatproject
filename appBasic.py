# pip install python-dotenv // for reading .env file
# streamlit for GUI
import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


def main():
    load_dotenv()
    # print(os.getenv('HELLO'))
    print('Hare Krshna')
    print("Hare Krshna \n this comes from .env >>", os.getenv("OPENAI_API_KEY"))
    st.set_page_config(page_title="Ask Your Questions About TS Payments")
    st.header("Ask Your Questions About TS Payments 💬")

    #uploading file
    pdf = st.file_uploader("Upload Your PDF", type="pdf")

    #extract the text from PDF
    if pdf is not None:
        pdf_Reader = PdfReader(pdf)
        text = ""
        for page in pdf_Reader.pages:
            text += page.extract_text()
        
        st.write(text)
        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        
        chunks = text_splitter.split_text(text)
        #st.write(chunks)
        #create Embeddings
        #embeddings = OpenAIEmbeddings()
        #knowledge_base = FAISS.from_texts(chunks, embeddings)





if __name__=='__main__':
    main()