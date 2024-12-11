import chromadb
chroma_client = chromadb.HttpClient(host='localhost', port=8000)


collection = chroma_client.create_collection(name="my_collection")
collection.add(
    documents=[
        "Get device details",
        "Get all devices"
    ],
    #metadatas= [{},{}]
    ids=["1", "2"]
)

results = collection.query(
    query_texts=["I want to know device information of samsung"], # Chroma will embed this for you
    n_results=1 # how many results to return
)
print(results)
