
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, VectorStoreIndex, Settings, Document
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.schema import TextNode
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from sentence_transformers import SentenceTransformer

# from langchain.chains.combine_documents.stuff import StuffDocumentsChain 
# from langchain.chains.llm import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain.chains.summarize import load_summarize_chain

import psycopg2

import pandas as pd
import textwrap
import os

import psycopg2 as pg

import streamlit as st

import requests
from PIL import Image
import traceback
from urllib.parse import quote_plus
import warnings


import json
from datetime import datetime , timedelta
 
# from llama_index.vector_stores.postgres import PGVectorStoreQuery as MetadataFilter

 
import time
import re

db = {
    'db_host': os.environ.get('db_host'), 
    'db_port': os.environ.get('db_port'), 
    'db_name': os.environ.get('db_name'), 
    'db_user': os.environ.get('db_user'), 
    'db_password': os.environ.get('POSTGRES_PASSWORD'), 
}
warnings.filterwarnings("ignore", category=FutureWarning)


def get_db_connection(_db):
    return psycopg2.connect(
        database=_db['db_name'],
        user=_db['db_user'], 
        password=_db['db_password'],
        host=_db['db_host'], 
        port=_db['db_port'],
    )

 
class pgDb:

    # @st.cache_resource(ttl=300)
    def get_db_cursor(self,_db):
        # Connect to the database
        conn = pg.connect(
            database=_db['db_name'],
            user=_db['db_user'], 
            password=_db['db_password'],
            host=_db['db_host'], 
            port=_db['db_port'],
        )

        # # Create a cursor object
        # cursor = conn.cursor()
        # return (conn, cursor)
        return coon



def ask_question(llm, question, interesting_phenomenon):
    messages = [
        ChatMessage(role="system", content=interesting_phenomenon),
        ChatMessage(role="user", content=question),
    ]
    stream_response = llm.stream_chat(messages)
    for r in stream_response:
        yield r.delta + " "

def context_engine(em,  db, ollama_host='ollama_container', llm_model='gemma2:2b'):
    embed_model_name = em['embed_model_name']
    embed_dimension = em['embed_dimension']
    # table_name = re.sub("[-.:#@\\/\\\]","_",embed_model_name)
    table_name = 'train1_review_embeddings'
    
    try:
        embed_model = HuggingFaceEmbedding(
            model_name=f"./models/{embed_model_name}", 
            trust_remote_code=True,
        )
        print(f"\n  Using {llm_model} language model with {embed_model_name} embedding model.\n")
        print("-"*100)
    except:
        download_model = SentenceTransformer(
            embed_model_name, 
            trust_remote_code=True,
        )
        download_model.save(f'models/{embed_model_name}')
        embed_model = HuggingFaceEmbedding(
            model_name=f"./models/{embed_model_name}", 
            trust_remote_code=True,
        )

    llm = Ollama(model=llm_model, request_timeout=900.0, base_url=f"http://{ollama_host}:11434")
    Settings.llm = llm
    Settings.embed_model = embed_model
    vector_store = PGVectorStore.from_params(
        database=db['db_name'],
        host=db['db_host'],
        password=quote_plus(db['db_password']),
        port=db['db_port'],
        user=db['db_user'],
        table_name=table_name,
        embed_dim=embed_dimension,
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store, 
        storage_context=storage_context,
    )
    query_engine = index.as_query_engine()
    

    return storage_context, query_engine, vector_store, index # vivian 

def review_embeddings(review, review_date, product, product_id, vector_store):
    review_node = [
                    TextNode(text=review,
                        metadata={
                            "review_date": review_date,
                            "product_name": product,
                            "product_id": product_id,

                        },
                    )
                ]
    vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    vector_index.insert_nodes(review_node)
    return review_node

class ChatOllama:
    def __init__(self, base_url="http://ollama_container:11434", model="ollama"):

        self.base_url = base_url
        # self.model = model 
        self.model = "gemma2:2b"
 
def review_summarize_langchain(db, _llm,  product_id, question):
     
    conn = get_db_connection(db)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT text
    FROM data_train1_review_embeddings
    WHERE metadata_->>'product_id' = %s 
    LIMIT 3 ;
    """, str(product_id) )
    results = cursor.fetchall()
    text_list = [row[0] for row in results] 
    result_string = '. '.join(text_list) 
    cursor.close()
     

   
    # Define prompt
    prompt_template = """Write a concise summary in a maximum of 10 bullet points of the following text enclosed within three backticks.:
    ```{text}```
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)
    llm_chain  = load_summarize_chain(_llm, chain_type="map_reduce")

    # llm_chain = LLMChain(llm=_llm, prompt=prompt)
     

    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text") 
    res = stuff_chain.invoke(result_string)
    return res["output_text"]



def review_summarize2( db, _llm,  product_id, question):
    # Step 1: Generate an embedding for the question using the existing HuggingFaceEmbedding model
    embed_model = Settings.embed_model 
    query_vector  = embed_model.get_text_embedding(question )   
    # Convert the query vector into the format expected by PostgreSQL's vector type
    query_vector_str = '[' + ','.join(map(str, query_vector)) + ']'

    conn = get_db_connection(db)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT text
    FROM data_train1_review_embeddings
    WHERE metadata_->>'product_id' = %s
    ORDER BY embedding <-> %s::vector
    LIMIT 3 ;
    """, (str(product_id), query_vector_str))
     

    results = cursor.fetchall()
    
    # reviews = [json.loads(metadata)['review_text'] for metadata, embedding in results]
    reviews = [text for text in results]
    
    cursor.close()

    summary_prompt = (
    "Summarize the review in 25 words."
    )
     
     
    info_response = _llm.complete(f"{summary_prompt}\n\n{reviews}")
    return json.loads(info_response.json())['text'] 
     


def review_summarize(db, vector_index, product_id):


    # Calculate date range: start_date is 1 year ago, end_date is today
    start_date = datetime.strftime(datetime.today() - timedelta(365), "%Y-%m-%d")
    end_date = datetime.strftime(datetime.today(), "%Y-%m-%d")

    # Perform the retrieval without predefined filters
    retriever = vector_index.as_retriever(similarity_top_k=10)
    retrieved_nodes = retriever.retrieve("What do customers think about price and parts included in the set?")

    # Manually filter results based on metadata
    filtered_nodes = []
    for node in retrieved_nodes:
        metadata = node.node.metadata
        review_date = metadata.get('review_date')
        product = metadata.get('product_id')

        # Apply the date range filter and product name filter manually
        if start_date <= review_date <= end_date and product == product_id:
            filtered_nodes.append(node)

    # Output the results using Streamlit's st.write
    for node in filtered_nodes:
        st.write(f"Score: {round(node.score * 100, 1)}")
        st.write(f"Metadata: {node.node.metadata}")
        st.write(f"Review Text: {node.node.to_dict()['text']}")
        st.write("---")  # Optional: Add a separator between reviews for better readability
     


# Generate review' vector
# conn = get_db_connection(db)
# embeddingAll(conn, vector_store)
def embeddingAll(conn, vector_store): 
    
    try: 
        cursor = conn.cursor() 
        # SQL query to fetch product_id and product_review
        query = "SELECT p.id  product_id , p.name  product_name , product_review   FROM products p join reviews r on p.id=r.product_id "
        cursor.execute(query)
        rows = cursor.fetchall()

        # Loop through the results and process each review
        for row in rows:
            product_id,   product_name, product_review = row
            # Call review_embeddings function for each review
            review_embeddings(
                product_review, 
                datetime.strftime(datetime.today(), "%Y-%m-%d"), 
                product_name, 
                product_id, 
                vector_store
            )

        # Commit changes if needed
        conn.commit()
        st.write("All reviews have been embedded successfully!")

    except Exception as e:
        st.write("An error occurred while embedding reviews:")
        st.write(str(e))
        traceback.print_exc()
    finally:
        # Always close the cursor and connection to avoid resource leaks
        if cursor:
            cursor.close()
        if conn:
            conn.close() 
  
embed_models = [{"embed_model_name": 'Alibaba-NLP/gte-large-en-v1.5', "embed_dimension": 1024}]

st.set_page_config(
    page_title="TrainStore",
    page_icon=":train:",
    initial_sidebar_state="expanded",
    layout='wide'    
    )




@st.cache_data(persist="disk")
def text_generation_with_gemma(_llm, about):
    interesting_info_response = _llm.complete(about)
    interesting_info = json.loads(interesting_info_response.json())['text']
    return interesting_info

@st.dialog(f"Write a review :")
def write_review(product_name, product_id, vector_store):
    review = st.text_area(product_name) 
    if st.button("Submit Review"):
        st.session_state.review = {"product": product_name, "review": review}
        review_embeddings(
            review, 
            datetime.strftime(datetime.today(), "%Y-%m-%d"), 
            product_name, 
            product_id, 
            vector_store
        )
        st.rerun()



## Show the models downloaded to ollama
with st.sidebar.container(border=True):
    # st.dataframe(pd.DataFrame(models_available())['name'])
    if st.button("Tell me more!", on_click=text_generation_with_gemma.clear):
        text_generation_with_gemma.clear()

def main():
    ollama_address = 'ollama_container'
    slm = 'gemma2:2b'
    text_llm = Ollama(model=slm, request_timeout=300.0, base_url=f"http://{ollama_address}:11434", additional_kwargs={"low_vram": True}) 
    storage_context, query_engine, vector_store, index  = context_engine(embed_models[0], db, ollama_host=ollama_address, llm_model=slm) # vivian 
   
    with open('./products.json', 'r') as pf:
        products = json.load(pf)
    product = products[0]
    st.subheader(product['Name'])
    col01, col02, col03 = st.columns([4,3,3])

    with col02:
        with st.container(height=50, border=False):
            st.empty()
        with st.container(height=1100, border=True):
            col02_1, col02_2 = st.columns([1,1])
            with col02_1:
                st.metric(label="Price", value=f"${product['Price']}")
            with col02_2:
                if st.button(label="BuyNow"):
                    st.balloons()
            product_table = {
                "Manufacturer": product['Manufacturer'],
                "PartNumber": product['PartNumber'],
                "Scale": product['Scale'],
                "Type": product['Type'],
                "Weight": product['Weight'],
            }
            st.dataframe(product_table, use_container_width=True)
            st.markdown("---")
            st.write(product['ProductDetails'])


    with col03:
        with st.container(height=50, border=False):
            st.empty()
        with st.container(height=300, border=True):
            messages = st.container(height=200)
            if prompt := st.chat_input("How can I help you?"):
                messages.chat_message("user").write(prompt)
                messages.chat_message("assistant").write(f"Echo: {prompt}")
        with st.container(height=800, border=True): 
            st.html("<h4><b>Customer reviews</b></h4>")
            if "review" not in st.session_state:
                if st.button("Write a review"):
                    write_review(product['Name'], product['ProductId'], vector_store)
            else:
                f"Thank you for your review!"
                # st.write(st.session_state.review)

    with col01:
        tab1, tab2, tab3, tab4 = st.tabs(["Train", "Assembled", "Parts", "Box"])
        with tab1:
            placeholder_tab1 = st.empty()
            with placeholder_tab1.container(height=800, border=True):
                w1, h1 = Image.open("./images/product0/image1.png").size
                image1 = Image.open("./images/product0/image1.png").resize((int(w1*300/h1), 300))
                st.image(image1, use_column_width=True)

        with tab2:
            placeholder_tab2 = st.empty()
            with placeholder_tab2.container(height=800, border=True):
                w2, h2 = Image.open("./images/product0/image2.png").size
                image2 = Image.open("./images/product0/image2.png").resize((w2*300//h2, 300))
                st.image(image2, use_column_width=True)

        with tab3:
            placeholder_tab3 = st.empty()
            with placeholder_tab3.container(height=800, border=True):
                w3, h3 = Image.open("./images/product0/image3.png").size
                image3 = Image.open("./images/product0/image3.png").resize((int(w3*300/h3), 300))
                st.image(image3, use_column_width=True)


        with tab4:
            placeholder_tab4 = st.empty()
            with placeholder_tab4.container(height=800, border=True):
                w4, h4 = Image.open("./images/product0/image4.png").size
                image4 = Image.open("./images/product0/image4.png").resize((int(w4*300/h4), 300))
                st.image(image4, use_column_width=True)

        with placeholder_tab1.container(border=True):
            w1, h1 = Image.open("./images/product0/image1.png").size
            image1 = Image.open("./images/product0/image1.png").resize((int(w1*300/h1), 300))
            st.image(image1, use_column_width=True)
            product_size = product['Scale']
            product_cost = product['Price']
            product_type = product['Type']
            locomotive_types = product['LocomotiveType']
            with st.spinner(f"Toy Trains"): 
                ollama_address = 'ollama_container'
                ollama_llm = ChatOllama()

                #text_llm = Ollama(model='gemma2:2b', request_timeout=300.0, base_url=f"http://{ollama_address}:11434", additional_kwargs={"low_vram": True})
                reviews_chain1  = review_summarize_langchain(db,   ollama_llm , 1, "Describe how this train set fits into different living spaces, and offer guidance on choosing the right size for various room sizes." )
                st.write(f'chain1. size = {reviews_chain1}')
                reviews1 = review_summarize2(db,   text_llm, 1, "Describe how this train set fits into different living spaces, and offer guidance on choosing the right size for various room sizes."  ) # "vivian" 
                st.write(f'1. size = {reviews1}')
                reviews2 = review_summarize2(db,   text_llm, 1, "What do customers think about the value for money?"  ) # "vivian" 
                st.write(f'2. cost = {reviews2}')
                reviews3= review_summarize2(db,   text_llm, 1, "Assess the suitability of the train set as a starter set for beginners. Highlight its ease of assembly, user-friendly features, and how it balances simplicity with engaging play to make it ideal for those new to model trains."  ) # "vivian" 
                st.write(f'3. Good starter set = {reviews3}')
                reviews4= review_summarize2(db,   text_llm, 1, "Provide a detailed overview of the train set's functionality. Highlight any special functions or capabilities that make it stand out. Including its operational features, ease of use."  ) # "vivian" 
                st.write(f'4. Functionality = {reviews4}')
                reviews5 = review_summarize2(db,   text_llm, 1, "Elaborate on the upgradeability options available for this train set. How can customers enhance their collection over time to keep their experience fresh and engaging?"  ) # "vivian" 
                st.write(f'5. Upgradeability = {reviews5}')

                prompt1=f"""
                    You are helpful toy trains shopping assistant. 
                    Help this online train store visitors improve their experience with helpful information about:
                        1. Size: {reviews1}
                        2. Cost: {reviews2}
                        3. Good starter set: {reviews3}
                        4. Functionality: {reviews4}
                        5. upgradeability {reviews5}
                        and other important factors.
                    Limit your reponse to 30 words.
                """
                activity_description = text_generation_with_gemma(text_llm, prompt1)
                st.info(f'Things to consider when shopping for Toy train set.', icon="ℹ️")

               



                st.success(f'Text generated with {slm} and cached with @st.cache_data decorator. Click the button to generate new text.')
                if st.button("Tell me more!", key="AboutTrain", help="This button triggers a new text to be generated by the Language Model used in this application.", type="primary"):
                    text_generation_with_gemma.clear(text_llm, prompt1)
                st.write(activity_description)
               

        with placeholder_tab2.container(border=True):
            w2, h2 = Image.open("./images/product0/image2.png").size
            image2 = Image.open("./images/product0/image2.png").resize((w2*300//h2, 300))
            accessories_included = product['AccessoriesIncluded']
            st.image(image2, use_column_width=True)
            with st.spinner(f"Assembling toy train sets"):
                prompt2=f"""
                    You are helpful toy trains shopping assistant. 
                    Write a brief recommendations on how to assemble a toy train set. 
                    Describe how these accessories included in this product {accessories_included} will help customers.
                    Limit your reponse to 300 words.
                """
                activity_description = text_generation_with_gemma(text_llm, prompt2)
                st.info('Read more about assembling toy trains!', icon="ℹ️")
                if st.button("Tell me more!", key="AssembledTrain", help="This button triggers a new text to be generated by the Language Model used in this application.", type="primary"):
                    text_generation_with_gemma.clear(text_llm, prompt2)
                st.write(activity_description)

        with placeholder_tab3.container(border=True):
            w3, h3 = Image.open("./images/product0/image3.png").size
            image3 = Image.open("./images/product0/image3.png").resize((int(w3*300/h3), 300))
            st.image(image3, use_column_width=True)
            with st.spinner(f"Toy train parts"):
                prompt3=f"""
                    You are a train enthusiast shopping assistant.
                    Review the following parts included in this train set and how customers can find long term value with these parts:
                        - {accessories_included}
                        - {locomotive_types}
                    Include the value customers can find for this {product_size} for this price {product_cost}.
                    Limit your reponse to 300 words.
                """
                activity_description = text_generation_with_gemma(text_llm, prompt3)
                st.info('More details about parts and accessories included in this train set.', icon="ℹ️")
                if st.button("Tell me more!", key="TrainAccessories", help="This button triggers a new text to be generated by the Language Model used in this application.", type="primary"):
                    text_generation_with_gemma.clear(text_llm, prompt3)
                st.write(activity_description)

if __name__ == "__main__":
    main()

