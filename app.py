import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models.openai import ChatOpenAI
from htmlTemplates import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    text = ""
    for pdf_doc in pdf_docs:
        reader = PdfReader(pdf_doc)
        for page in reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator  = '\n',
                                          chunk_size = 1000,
                                          chunk_overlap=200,
                                          length_function=len)
   
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore

def get_conversation(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(input_key='question',return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({"question":user_question})
    st.write(response)


def main():
    load_dotenv()
    st.set_page_config(page_title='Research chat', page_icon='🏠')
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header('Research chat: Documents')
    user_question=st.text_input('Ask questions directly to the research paper')
    if user_question:
        handle_user_input(user_question)


    st.write(user_template.replace("{{MSG}}","Hello fellow"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}","Hello there"), unsafe_allow_html=True)
   
    with st.sidebar:
        st.subheader('Research Documents: Pose estimation papers')
        pdf_docs=st.file_uploader(
            "Upload your documents here and click on 'Process'", accept_multiple_files=True)
        if st.button('Process'):
            with st.spinner("Processing"):
                #upload pdf text
                raw_text = get_pdf_text(pdf_docs)
                st.write(raw_text)


                #get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)


                #create a vector store with embeddings
                vectorstore = get_vectorstore(text_chunks)


                #create conversation chain
                st.session_state.conversation = get_conversation(vectorstore)

    

if __name__ == '__main__':
    main()