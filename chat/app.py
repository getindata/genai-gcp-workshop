# To authenticate locally, run:
# gcloud auth application-default login
# gcloud config set project datamass-2023-genai
# gcloud auth application-default set-quota-project datamass-2023-genai

from app_utils.app_utils import (
    download_utils,
    rate_limit,
    CustomVertexAIEmbeddings
)
download_utils("utils")

import json
import textwrap
import uuid
import numpy as np
import vertexai
from google.cloud import aiplatform
import langchain
from langchain.chains import RetrievalQA
from langchain.document_loaders import GCSDirectoryLoader
from langchain.llms import VertexAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.matching_engine import MatchingEngine
from utils.matching_engine_utils import MatchingEngineUtils
import streamlit as st
from streamlit_chat import message


PROJECT_ID = "datamass-2023-genai"  # @param {type:"string"}
REGION = "us-central1"  # @param {type:"string"}
EMBEDDING_QPM = 100
EMBEDDING_NUM_BATCH = 5
ME_REGION = "us-central1"
ME_INDEX_NAME = f"{PROJECT_ID}-me-index"  # @param {type:"string"}
ME_EMBEDDING_DIR = f"{PROJECT_ID}-me-bucket"  # @param {type:"string"}
ME_DIMENSIONS = 768  # when using Vertex PaLM Embedding
NUMBER_OF_RESULTS = 10
SEARCH_DISTANCE_THRESHOLD = 0.6

vertexai.init(project=PROJECT_ID, location=REGION)

llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=1024,
    temperature=0.2,
    top_p=0.8,
    top_k=40,
    verbose=True,
)

embeddings = CustomVertexAIEmbeddings(
    requests_per_minute=EMBEDDING_QPM,
    num_instances_per_batch=EMBEDDING_NUM_BATCH,
)

mengine = MatchingEngineUtils(PROJECT_ID, ME_REGION, ME_INDEX_NAME)

ME_INDEX_ID, ME_INDEX_ENDPOINT_ID = mengine.get_index_and_endpoint()

me = MatchingEngine.from_components(
    project_id=PROJECT_ID,
    region=ME_REGION,
    gcs_bucket_name=f"gs://{ME_EMBEDDING_DIR}".split("/")[2],
    embedding=embeddings,
    index_id=ME_INDEX_ID,
    endpoint_id=ME_INDEX_ENDPOINT_ID,
)

retriever = me.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": NUMBER_OF_RESULTS,
        "search_distance": SEARCH_DISTANCE_THRESHOLD,
    },
)

template = """SYSTEM: You are an intelligent assistant helping the users with their questions on research papers.

Question: {question}

Strictly Use ONLY the following pieces of context to answer the question at the end. Think step-by-step and then answer.

Do not try to make up an answer:
 - If the answer to the question cannot be determined from the context alone, say "I cannot determine the answer to that."
 - If the context is empty, just say "I do not know the answer to that."

=============
{context}
=============

Question: {question}
Helpful Answer:"""

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    verbose=True,
    chain_type_kwargs={
        "prompt": PromptTemplate(
            template=template,
            input_variables=["context", "question"],
        ),
    },
)

qa.combine_documents_chain.verbose = True
qa.combine_documents_chain.llm_chain.verbose = True
qa.combine_documents_chain.llm_chain.llm.verbose = True


def get_response(query, qa=qa, k=NUMBER_OF_RESULTS, search_distance=SEARCH_DISTANCE_THRESHOLD):
    qa.retriever.search_kwargs["search_distance"] = search_distance
    qa.retriever.search_kwargs["k"] = k
    result = qa({"query": query})

    return result["result"]

### Streamlit Chat

# App title
st.title("LLM Interface to indexed documents")

# Generate empty lists for generated and past.
# `generated` stores generated responses
if "generated" not in st.session_state:
    st.session_state["generated"] = ["Write your query and hit [Enter]!"]
# `past`` stores user's questions
if "past" not in st.session_state:
    st.session_state["past"] = [
        "This is a conversational interface to indexed documents."
    ]

# Layout of input/response containers
response_container = st.container()
input_container = st.container()

# User input
with input_container:
    query = st.text_input("You: ", "", key="input")

# Conditional display of generated responses as a function of user provided prompts
with response_container:
    if query:
        response = get_response(query)
        st.session_state.past.append(query)
        st.session_state.generated.append(response)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
            message(st.session_state["generated"][i], key=str(i))
