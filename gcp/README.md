## Prepare blog posts
```bash
pip install docx2pdf
docx2pdf iac/resources/blog_posts
```

## setup GCP project
```bash
cd gcp/
gcloud auth application-default login
gcloud config set project datamass-2023-genai
terraform init
terraform apply

```