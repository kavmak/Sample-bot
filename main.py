import os
import streamlit as st
import pickle
import time
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import UnstructuredImageLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import TextLoader
import json
import re
from urllib.request import urlopen
from audio_recorder_streamlit import audio_recorder
import datetime
import openai
import requests
from dotenv import load_dotenv
load_dotenv()

def load_txt_data(uploaded_file):
    with open('messages.txt', 'w') as f:
        f.write(uploaded_file.getvalue().decode())
    return uploaded_file.getvalue().decode()

def translate_text(text):
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=f"Translate the text to English with proper meaning preserved:\n{text}",
        temperature=0.5,
        max_tokens=60
    )
    translated_text = response.choices[0].text.strip()
    return translated_text


def save_audio_file(audio_bytes, file_extension):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"my_file.mp3"
    with open(file_name, "wb") as f:
        f.write(audio_bytes)
    return file_name


def my_audio_recorder():
    audio_bytes = audio_recorder(
        text="",
        icon_name="microphone",
        icon_size="2x",
    )
    if audio_bytes:
        save_audio_file(audio_bytes, "mp3")
        return transcribe_audio("my_file.mp3")


def transcribe(audio_file):
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript


def transcribe_audio(file_path):
    """
    Transcribe the audio file at the specified path.
    :param file_path: The path of the audio file to transcribe
    :return: The transcribed text
    """
    with open(file_path, "rb") as audio_file:
        transcript = transcribe(audio_file)
    original_text = transcript["text"]
    st.write(original_text)
    english_text = translate_text(original_text)
    st.write(english_text)
    return english_text

st.title("Smart Read")
st.subheader(" - Team Hi-Fi")
st.sidebar.title("News Article URLs")
col1, col2 = st.columns(2)
st.subheader(" ")
with col1:
    process_clicked = st.button("Process", key="process_button")

with col2:
    post_on_slack = st.button("Post on slack", key="slack_button")
st.subheader(" ")

urls = []
urls = urls[:0]

for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
ans_placeholder = st.empty()
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.9, max_tokens=500)

input_documents_list = []
uploaded_pdfs = st.sidebar.file_uploader("Upload pdf", accept_multiple_files=True)
uploaded_ppts = st.sidebar.file_uploader("Upload ppt", accept_multiple_files=True, type=['ppt', 'pptx'])
uploaded_imgs = st.sidebar.file_uploader("Upload images", accept_multiple_files=True, type=['png', 'jpeg', 'jpg'])

if process_clicked:
    with open("messages.txt", "r") as f:
        for line in f:
            urls_in_line = re.findall(r'<(.*?)>', line)
            urls += urls_in_line


    # load data
    url_loader = UnstructuredURLLoader(urls=urls)
    ans_placeholder.text("Data Loading...Started...✅✅✅")
    input_documents_list += url_loader.load()

    for uploaded_ppt in uploaded_ppts:
        ppt_loader = UnstructuredPowerPointLoader(uploaded_ppt.name)
        input_documents_list += (ppt_loader.load())

    for uploaded_img in uploaded_imgs:
        img_loader = UnstructuredImageLoader(uploaded_img.name)
        input_documents_list += (img_loader.load())

    for uploaded_pdf in uploaded_pdfs:
        pdf_loader = UnstructuredPDFLoader(uploaded_pdf.name)
        input_documents_list += (pdf_loader.load())


    url = "https://app.coupler.io/export/w/84f9c7c2-3797-11ef-9ac8-eb93d593d5a7.json?access_token=4c48db06afa0fe5fb599bf5515b2c74ba860d8efd12ae96e63de06844a56"
    response = urlopen(url)
    data_json = json.loads(response.read())
    with open('messages.txt', 'w') as file:
        for message in data_json:
            text = message['text']
            file.write(text + '\n')

    loader = TextLoader("messages.txt")
    input_documents_list += (loader.load())


    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ',', ' '],
        chunk_size=500
    )
    ans_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(input_documents_list)
    #create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()

    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    ans_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

query = main_placeholder.text_input("Question: ")
query2 = my_audio_recorder()
if query2 != None:
    st.text_area(label="Recorded question: ", value=query2)

if query == "":
    query = query2

if query:
    ans_placeholder.text("Fetching the answer...✅✅✅")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f) #TODO
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever()) #TODO
            result = chain({"question": query}, return_only_outputs=True)
            st.header("Answer")
            st.write(result["answer"])

            sources = result.get("sources", "")
            fl = 0
            if sources:
                sources_list = sources.split("\n")
                for source in sources_list:
                    if source == "messages.txt":
                        fl = 1
            if sources and fl == 0:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)

            if post_on_slack: #TODO
                ans_placeholder.text("Posting on Slack...✅✅✅")
                url = ''
                if fl == 0:
                    payload = {"text": f"*Question:*\n{query}\n*Answer:*\n{result['answer']}\n*Sources:*\n{result['sources']}"}
                else:
                    payload = {"text": f"*Question:*\n{query}\n*Answer:*\n{result['answer']}"}
                r = requests.post(url, data=json.dumps(payload))
                query = ""
                query2 = None
                urls = urls[:0]


    query = ""
    query2 = None
    urls = urls[:0]
