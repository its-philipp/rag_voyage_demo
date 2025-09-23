output "repo_id" {
  value       = databricks_repo.repo.id
  description = "Databricks repo id"
}

output "index_job_id" {
  value       = databricks_job.index_job.id
  description = "Job id for FAISS index build"
}

output "bm25_job_id" {
  value       = databricks_job.bm25_job.id
  description = "Job id for BM25 index build"
}
