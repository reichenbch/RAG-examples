import sys
import subprocess

subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt', '--quiet'])

import os
import openai
import fasttext
import gradio as gr

from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from huggingface_hub import hf_hub_download
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage, HumanMessage
from langchain.embeddings.openai import OpenAIEmbeddings

model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
model = fasttext.load_model(model_path)

os.environ['OPENAI_API_KEY'] = <openai-api-key>

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
vectorStore = FAISS.load_local("faiss_doc_idx", embeddings)

def predict(message, history):
    history_langchain_format = []
        
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))
    
    language = model.predict(message)[0][0].split('__')[-1]
    template = """I want you to act as a question answering bot which uses the context mentioned and answer in a concise manner and doesn't make stuff up.
            You will answer question based on the context - {context}.
            You will create content in""" + str(language) + """language.
            Question: {question}
            Answer:
            """
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorStore.as_retriever(), chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    result = qa_chain({"query": message})
    
    history_langchain_format.append(HumanMessage(content=message))
    history_langchain_format.append(AIMessage(content=result['result']))
    
    return result['result']

gr.ChatInterface(predict,
    chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Ask me a question related to PAN Services", container=False, scale=7),
    title="DocumentQABot",
    theme="soft",
    examples=["What is the cost/fees of a PAN card?", "How long does it usually take to receive the PAN card after applying?"],
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",).launch(share=True) 