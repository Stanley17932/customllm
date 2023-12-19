from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import ctransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from langchain.llms.ctransformers import CTransformers

DB_FAISS_PATH = "vectorstores/"

custom_prompt_template = """ Use the information processed to answer the users questions.
If you dont know the answer say you dont know do not make up one

Context: {}
Question: {}

Only return helpful answer nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for the chatbot for the created vector stores
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_llm():
    llm= CTransformers(model=r"E:\customllm\llama\llama\llama-2-7b-chat\consolidated.00.pth", model_type="llama",
                       max_new_tokens=512, temperature=0.5)

    return llm

def retrieval_qa_chain(prompt,db, llm):
    """
    Chain for the chatbot
    """
    qa_chain = RetrievalQA.from_chain_type(llm=llm, 
                                        chain_type="stuff", 
                                        retriever=db.as_retriever(search_kwargs={'k':2}),
                                        return_source_documents = True,
                                        chain_type_kwargs = {'prompt': prompt})
   
    return qa_chain

def qa_bot():
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device":"cpu"})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(qa_prompt, db, llm)
    return qa

def final_result(query):
    qa = qa_bot()
    response = qa({"question":query})
    return response


### Chainlit code#############
@cl.on_chat_start
async def start():
    chain = qa_bot()
    if chain:
        msg = cl.Message(content="Hi, I'm a chatbot. How can I help you?")
        await msg.send()
        msg.content = "Hello welcome, what's your query today?"
        await msg.update()
        cl.user_session.set("chain", chain)
    else:
        print("Error: Chain is None.")


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer= True,
        answer_prefix_tokens= ["FINAL","ANSWER"]
    )
    cb.answer_reached=True
    res = await chain.acall(message, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" +str(sources)
    else:
        answer += f"\nNo sources found."
    
    await cl.Message(content=answer).send()