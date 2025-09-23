variable "databricks_host" {
  description = "Databricks workspace host, e.g., https://adb-XXXX.XX.azuredatabricks.net"
  type        = string
}

variable "databricks_token" {
  description = "Databricks PAT token"
  type        = string
  sensitive   = true
}

variable "repo_url" {
  description = "Git URL of this repository (HTTPS)"
  type        = string
}

variable "repo_branch" {
  description = "Branch to use for Databricks repo"
  type        = string
  default     = "main"
}

variable "job_cluster_node_type_id" {
  description = "Override node type for job clusters"
  type        = string
  default     = null
}

variable "job_cluster_spark_version" {
  description = "Override spark version for job clusters"
  type        = string
  default     = null
}

variable "warehouse_id" {
  description = "Optional SQL warehouse id if needed for tasks"
  type        = string
  default     = null
}
