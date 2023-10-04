"""Chat interface to indexed documents."""

# 0. Imports

from importlib import import_module

import streamlit as st
import vertexai
from langchain.chains import RetrievalQA
from langchain.llms import VertexAI
from langchain.prompts import PromptTemplate
from streamlit_chat import message

from app_utils.app_utils import (
    CustomVertexAIEmbeddings,
    download_utils,
    get_response,
    template,
)

download_utils("utils")
MatchingEngine = getattr(import_module("utils.matching_engine"), "MatchingEngine")
MatchingEngineUtils = getattr(
    import_module("utils.matching_engine_utils"), "MatchingEngineUtils"
)

# 1. Parameters

# Fixed
PROJECT_ID = "datamass-2023-genai"
REGION = "us-central1"
ME_REGION = "us-central1"
ME_INDEX_NAME = f"{PROJECT_ID}-me-index"
ME_EMBEDDING_DIR = f"{PROJECT_ID}-me-bucket"

# Controlled with Streamlit app
with st.sidebar:
    st.write("**Vector search parameters:**")
    EMBEDDING_QPM = st.number_input(
        "Requests per minute (`EMBEDDING_QPM`)", 10, 1000, 100, 10
    )
    EMBEDDING_NUM_BATCH = st.number_input(
        "Number of instances per batch (`EMBEDDING_NUM_BATCH`)", 1, 50, 5, 1
    )
    NUMBER_OF_RESULTS = st.number_input(
        "Number of returned results (`NUMBER_OF_RESULTS`)", 1, 30, 10, 1
    )
    SEARCH_DISTANCE_THRESHOLD = st.number_input(
        "Search distance threshold (`SEARCH_DISTANCE_THRESHOLD`)", 0.01, 1.0, 0.6, 0.01
    )
    st.write("**LLM parameters:**")
    TEMPERATURE = st.number_input("Temperature (`TEMPERATURE`)", 0.0, 1.0, 0.2, 0.01)
    TOP_P = st.number_input("Top p (`TOP_P`)", 0.01, 1.0, 0.8, 0.01)
    TOP_K = st.number_input("Top k (`TOP_K`)", 1, 40, 40, 1)

# 2. Object initialization

vertexai.init(project=PROJECT_ID, location=REGION)

llm = VertexAI(
    model_name="text-bison@001",
    max_output_tokens=1024,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
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

# 3. Streamlit Chat

# App title
st.title("LLM interface to indexed documents")

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
    query = st.chat_input("", key="input")

# Conditional display of generated responses as a function of user provided prompts
with response_container:
    if query:
        response = get_response(
            query, qa=qa, k=NUMBER_OF_RESULTS, search_distance=SEARCH_DISTANCE_THRESHOLD
        )
        st.session_state.past.append(query)
        st.session_state.generated.append(response)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
            message(st.session_state["generated"][i], key=str(i))
