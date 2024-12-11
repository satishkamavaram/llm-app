"""from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
import os
import chromadb
from langchain_core.documents import Document
from uuid import uuid4

os.environ["OPENAI_API_KEY"] = "_Jk9pvU28ToZqT3BlbkFJI9y6BNaUjF76Yn73FoVFFlmh6gQdX47-p8JQL4oSOPaD9zVr9B0terNnm9g7K_f2-f7n7qscYA"

pdf_url = "/Users/satish/Downloads/digital-leader.pdf"
loader = PyPDFLoader(pdf_url)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
texts = text_splitter.split_documents(documents)


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

#print(texts)
#https://github.com/chroma-core/chroma/blob/c665838b0d143e2c2ceb82c4ade7404dc98124ff/chromadb/config.py#L83
# Define Chroma server settings for client-server architecture
chroma_settings = Settings(
 #   chroma_api_impl="rest",  # Use REST API for Chroma
    chroma_server_host="localhost",  # Chroma server host
    chroma_server_http_port="8000"  # Chroma server port
)

texts_str = [text.page_content for text in texts]
ids = [text[:5] for text in texts_str]

collection_name = "digital_leader"
chroma_client = chromadb.HttpClient(host='localhost', port=8000)
collection = chroma_client.get_or_create_collection(name=collection_name) 
#collection.add(documents=texts_str,ids=ids)

#https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.chroma.Chroma.html
# https://python.langchain.com/docs/integrations/vectorstores/chroma/
# Initialize Chroma vector store using ChromaDB embeddings
# No external embedding function is required here
vectordb = Chroma(
    collection_name=collection_name,  # Unique collection name
   # client_settings = chroma_settings,
    embedding_function=embeddings,
    client= chroma_client
)

docs = []
count = 0
for text in texts_str:
    count = count+1
    doc = Document(
    page_content=text,
    metadata={"source": "tweet"},
    id=count,)
    docs.append(doc)

uuids = [str(uuid4()) for _ in range(len(docs))]

vectordb.add_documents(documents=docs, ids=uuids)
#ids = vectordb.add_documents(documents=texts)

# Example: Query the database
query = "What is new societal expectations"
results = vectordb.similarity_search(query, k=3)

#results = await vector_store.asimilarity_search("When was Nike incorporated?")
#print(results[0])
print("???????????????????????????????????????????????????????????????????????????????????????????????")
# Display results
for result in results:
    print("************************************************************************************")
    print(result.page_content)

"""
########above code - load pdf, chunk it, openai embedding, connect to chroma db , generate embedding , store to chroma db, query similary search##############


########below code -  openai embedding, connect to chroma db , get embedding for user query with query similary search, pass the returned results to chatgpt model for summary details##############

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from chromadb.config import Settings
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
import os
import chromadb
from langchain_core.documents import Document
from uuid import uuid4

os.environ["OPENAI_API_KEY"] = "skMeKJmUbYsFU86E14kHT3NITkbzn73FoVFFlmh6gQdX47-p8JQL4oSOPaD9zVr9B0terNnm9g7K_f2-f7n7qscYA"

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

collection_name = "digital_leader"
chroma_client = chromadb.HttpClient(host='localhost', port=8000)
collection = chroma_client.get_or_create_collection(name=collection_name) 
#collection.add(documents=texts_str,ids=ids)

#https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.chroma.Chroma.html
# https://python.langchain.com/docs/integrations/vectorstores/chroma/
# Initialize Chroma vector store using ChromaDB embeddings
# No external embedding function is required here
vectordb = Chroma(
    collection_name=collection_name,  # Unique collection name
    embedding_function=embeddings,
    client= chroma_client
)

# Example: Query the database
query = "What is new societal expectations"
# using similarity search - fetches similar text data
results = vectordb.similarity_search(query, k=3)

#results = await vector_store.asimilarity_search(query,k=3)
#print(results[0])
print("??????????????????????????????????abc?????????????????????????????????????????????????????????????")
# Display results
passage = ""
for result in results:
    print("********************************abc****************************************************")
    print(result.page_content)
    passage = passage + result.page_content

print("==================================abc==============================================")
print(passage)

### using retrievers - fetches most relevant data
#https://python.langchain.com/api_reference/chroma/vectorstores/langchain_chroma.vectorstores.Chroma.html#langchain_chroma.vectorstores.Chroma.as_retriever
retriever = vectordb.as_retriever(
    search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5}
)

results = retriever.invoke(query)

print("retriever??????????????????????????????????abc?????????????????????????????????????????????????????????????")
# Display results
passage = ""
for result in results:
    print("retriever********************************abc****************************************************")
    print(result.page_content)
    passage = passage + result.page_content

print("retriever==================================abc==============================================")
print(passage)



from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# https://python.langchain.com/docs/tutorials/classification/
class Classification(BaseModel):
    sentiment: str = Field(..., enum=["happy", "neutral", "sad"])
    summary: str = Field(
        ...,
        description="summary information",
    )
    

model = ChatOpenAI(model="gpt-4o-mini",temperature=1).with_structured_output(
    Classification
)


tagging_prompt = ChatPromptTemplate.from_template(
    """
Extract the desired summary information from the following passage for a given user input.

Only extract the properties mentioned in the 'Classification' function.


Passage:
{passage}

user_input:
{input}
"""
)

prompt = tagging_prompt.invoke({"passage": passage, "input": query})

response = model.invoke(prompt)
print("==========================summary response========abc==============================================")
print(response.dict())
