from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from data import resumes, jobs
import openai
import time

# Constants
openai.api_key = <"OPENAI_API_KEY">

def collection_create(collection_name, dimension):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, descrition='Ids', is_primary=True, auto_id=True),
        FieldSchema(name='name', dtype=DataType.VARCHAR, descrition='name', max_length=200),
        FieldSchema(name='description', dtype=DataType.VARCHAR, description='description', max_length=20000),
        FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, description='CV embedding', dim=dimension)
    ]
    schema = CollectionSchema(fields=fields, description='Datasets descrption')
    collection = Collection(name=collection_name, schema=schema)

    index_params = {
        'index_type': 'IVF_FLAT',
        'metric_type': 'L2',
        'params': {'nlist': 1024}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"Collection '{collection_name}' has been created.")

def openai_embedding(text, model="text-embedding-ada-002"):
   return openai.Embedding.create(input = text, model=model)['data'][0]['embedding']

def data_insert(collection_name, data):
    collection = Collection(name=collection_name)
    for row in data:
        ins=[[row['name']], [row['description']], [openai_embedding(row['description'])]]
        collection.insert(ins)
        time.sleep(1)  # Free OpenAI account limited to RPM
    print(f"Inserted {len(data)} rows to the {collection_name} collection...")

def _search(collection, text, limit):
    # Search parameters for the index
    search_params={
        "metric_type": "L2"
    }

    results=collection.search(
        data=[openai_embedding(text)],  # Embeded search value
        anns_field="embedding",  # Search across embeddings
        param=search_params,
        limit=limit,  # Limit to five results per search
        output_fields= ['name', 'description']
    )
    return results

def search(collection_name, text, limit):
    collection = Collection(name=collection_name)
    collection.load()

    res = []
    hits = _search(collection, text, limit)
    
    # Iterate over the hits
    for hit in hits:
        for item in hit:
            # Extract name and description from each item and append to the results
            res.append({
                "name": item.entity.get('name'),
                "description": item.entity.get('description'),
                "distance": item.distance  # This field represents the relevance of the result
            })
    
    collection.release()
    return res


if __name__ == "__main__":
    # Connection to the Milvus DB
    connections.connect(host='localhost', port='19530')
    print("Connected to Milvus DB...")
    
    # Create collections for resumes and jobs
    collection_create('resumes', 1536)  # The dimension of your embeddings should match the output of the model used. Here it is assumed to be 768.
    collection_create('jobs', 1536)

    # Insert the data into the respective collections
    data_insert('resumes', resumes)
    data_insert('jobs', jobs)
    
    # Demonstrate a search
    cv = resumes[0]['description']  # Let's say we're searching for jobs that match the first resume
    name = resumes[0]['name']
    print(f"\nLet's find related jobs for {name}")
    print(f"CV: {cv}")
    print(f"Search results:")
    print(search('jobs', cv, 3))  # The 3 most relevant jobs will be printed


    
    
