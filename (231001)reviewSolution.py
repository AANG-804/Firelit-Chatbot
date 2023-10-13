import streamlit as st
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

# 1. Vectorise the sales response csv data

loader = CSVLoader(file_path=r"C:\Users\P\Desktop\Langchain\reviewSolution\(231001)reviewpair.csv")
documents = loader.load()
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)

# 2. Function for similarity search

def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    # print(page_contents_array)

    return page_contents_array

# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")


template = """
You are a very friendly, sociable representative of a delivery restaurant. 
You must be very good at dealing with customer complaints about their orders. 
You are responsible for the status of customer reviews on the delivery platform. 
You must be straightforward and responsive to customers, and avoid using ambiguous language.

I will share a prospect's message with you and you will give me the best answer that.
I will share delivery restaurant`s request that you should reflects after you select the best practice.
I should send to this prospect based on past best practies, delivery restaurant`s request.
and you will follow ALL of the rules below:

1/ Response should be very similar or even identical to the past best practies, 
in terms of length, ton of voice, logical arguments and other details

2/ If the best practice are irrelevant, then try to mimic the style of the best practice to prospect's message

3/ Once you've built your response, make sure that it naturally reflects delivery restaurant`s request.

Below is a message I received from the prospect:
{message}

Here is a list of best practies of how we normally respond to prospect in similar scenarios:
{best_practice}

Here is delivery restaurant`s request:
{request}

Please write the best response that I should send to this prospect:
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice","request"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)



# 4. Retrieval augmented generation
def generate_response(message,request):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice, request=request)
    return response

message = """
소고기 타다끼를 시켰는데 너무 익혀서 나왔어요. 다시는 안먹을거 같아요.
"""

request = """
다시는 시키지 말라는 말을 반영해줘.
"""
response =  generate_response(message,request)

print('--------소비자 리뷰 예시--------')
print(message)
print('--------사장의 반영 요청 내용--------')
print(request)
print('--------생성 답변--------')
print(response)