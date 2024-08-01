import os
import slack
import ast
from langchain.llms import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from config import OPENAI_API_KEY, slack_channel, slack_bot_token
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
embeddings = OpenAIEmbeddings()


def extract_text_from_pdf(path):
    loader = PyPDFLoader(path)
    docs = loader.load()
    return docs


def split_text(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    return docs


def save_to_vectorstore(chunks):
    db = FAISS.from_documents(chunks, embeddings)
    return db


def query_llm(db, queries):
    answers = []
    retriever = db.as_retriever()
    for query in queries:
        docs = retriever.invoke(query)
        text_chunks = [docs[idx].page_content for idx in range(len(docs))]
        context = " ".join(text_chunks)
        request = {"query": query, "context": context}
        llm = OpenAI()
        response = llm.predict(f"Your role is to answer the question based on context. The input format is a dictonary"
                               f"which has keys query and context and the value of dictonary is the original query and extracted"
                               f"context from PDF. Send the answer in the format of a dictonary, keys will be query, answer, confidence score in percent."
                               f"If confidence score is less than 30%, you can answer Data is not available. The input is {request}")

        try:
            response_dict = ast.literal_eval(response.strip())
        except (ValueError, SyntaxError):
            response_dict = {"query": query, "answer": "Data is not available", "confidence score": "0%"}
        formatted_response = {'query': response_dict.get('query', query),
                              'answer': response_dict.get('answer', "Data is not available"),
                              'score': response_dict.get('confidence score', "0%")}

        answers.append(formatted_response)
    return answers


def send_to_slack(text):
    slack_client = slack.WebClient(token=slack_bot_token)
    slack_client.chat_postMessage(channel=slack_channel, text=text)
