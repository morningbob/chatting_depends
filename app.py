import json
import openai
import boto3

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.docstore.document import Document


def lambda_handler(event, context):
    
    openai.api_key = get_api_key()
    print("I passed api key line")
    
    model = ChatOpenAI(openai_api_key=openai.api_key, temperature=0)
    print("I passed create model line")
    
    doc =  Document(page_content="Jessie is a programmer.  She is specialized in writing Android apps.", metadata={"source": "local"})
    print("I created doc")
    db = Chroma.from_documents([doc], OpenAIEmbeddings(openai_api_key=openai.api_key))
    print("I created chroma db")
    result = db.similarity_search(
        "who is Jessie",
        k = 1
    )
    print("I searched")
    print(result)
    print("finished")
    '''
    sample_text = [
    #"Jessie is a programmer.  She is specialized in writing Android apps.",
    "Jessie had released many apps in Google Play Store.  The apps include using bluetooth to chat, send messages using asymmetric encryption, identify emotions using AI models, let people share dog walk routes with the others.",
    #"Jessie also built her website with React.  The website's address is www.jessbitcom.pro.",
    #"Jessie also wrote some sample iOS apps with Swift and SwiftUI.  This includes the bluetooth or wifi to chat with nearby devices, the iOS version."
    #"Jessie has a dog who's name is Python.  She likes dogs very much.  Python is a male red shiba.  He is 4 years old.",
    #"Jessie also like to read fictions and watch TV."
    ]
    print("I passed setup text line")
    vectorstore = DocArrayInMemorySearch.from_texts(
        sample_text,
        embedding=OpenAIEmbeddings(openai_api_key=openai.api_key)
    )
    print("I passed create vectorstore line")
    retriever = vectorstore.as_retriever()
    print("I passed retriever setup line")

    memory = ConversationBufferMemory(return_messages=True,memory_key="chat_history")
    qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0, openai_api_key=openai.api_key), vectorstore.as_retriever(), memory=memory)
    print("before invoke")
    response = qa.invoke({"question": "What is Jessie's job?"})
    '''
    #print(response)
    
    #text_response = response['answer']
    return {
        'statusCode':200,
        'body': {
            'response' : result[0].page_content
        }
    }
    
def get_api_key():
    lambda_client = boto3.client('lambda')
    response = lambda_client.invoke(
            FunctionName = 'arn:aws:lambda:us-east-1:611916090982:function:openai_get_api_key',
            InvocationType = 'RequestResponse'
        )

    openai_api_key = json.load(response['Payload'])['body']['api_key']
    return openai_api_key
    
'''

def lambda_handler(event, context):
    
    model_to_use = "text-davinci-003"
    input_prompt="Write an email to Elon Musk asking him why he bought Twitter for such a huge amount"
    
    openai.api_key = get_api_key()
    
    response = openai.Completion.create(
      model=model_to_use,
      prompt=input_prompt,
      temperature=0,
      max_tokens=100,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0
    )
    
    response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages= [
                    {'role': 'user', 'content': 'Translate the following English text to French: '}
                ]
            )
    #print(response)
    
    text_response = response['choices'][0]['text'].strip()
    return {
        'statusCode':200,
        'body': {
            'response' : text_response
        }
    }
    
def get_api_key():
    lambda_client = boto3.client('lambda')
    response = lambda_client.invoke(
            FunctionName = 'arn:aws:lambda:us-east-1:611916090982:function:openai_get_api_key',
            InvocationType = 'RequestResponse'
        )

    openai_api_key = json.load(response['Payload'])['body']['api_key']
    return openai_api_key
'''