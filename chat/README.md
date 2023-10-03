# Chat Interface

Prerequisites:
- Python 3.10
- [Poetry](https://python-poetry.org/)
- [gcloud CLI utility](https://cloud.google.com/sdk/docs/install) 

To install dependencies:
```
poetry env use <your_python_installation>
poetry install
```

To authenticate locally:
```
gcloud auth application-default login
gcloud config set project datamass-2023-genai
gcloud auth application-default set-quota-project datamass-2023-genai
```

To run app:
```
streamlit run app.py
```