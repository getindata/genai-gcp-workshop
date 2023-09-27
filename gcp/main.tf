resource "google_storage_bucket" "input-bucket" {
  project = var.project_id
  name          = local.input-bucket
  location      = var.location
  force_destroy = true

  public_access_prevention = "enforced"
}

resource "google_storage_bucket_iam_binding" "input_bucket_binding" {
  bucket = google_storage_bucket.input-bucket.name
  role = "roles/storage.admin"
  members = [
    "group:${var.admin-group}",
  ]
}


resource "google_storage_bucket_object" "blog-posts" {
  for_each = fileset(path.module, "resources/blog_posts/*.pdf")
  name   = "blog_posts/${reverse(split("/", each.value))[0]}"
  source = each.value
  bucket = google_storage_bucket.input-bucket.name
}

resource "google_compute_network" "vpc_network" {
  project = var.project_id
  name = "default"
  auto_create_subnetworks = true
  routing_mode = "GLOBAL"
}
resource "google_project_service" "api" {
  project = var.project_id
  for_each = toset(["servicenetworking.googleapis.com", "aiplatform.googleapis.com"])
  service = each.value
}