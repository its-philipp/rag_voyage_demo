terraform {
  required_version = ">= 1.5.0"
  required_providers {
    databricks = {
      source  = "databricks/databricks"
      version = ">= 1.39.0"
    }
  }
}

provider "databricks" {
  host  = var.databricks_host
  token = var.databricks_token
}

# Optionally discover latest LTS Spark version and a small node type for jobs
data "databricks_spark_version" "latest_lts" {
  long_term_support = true
}

data "databricks_node_type" "smallest" {
  local_disk = true
}

# Clone this repo into the workspace for running jobs
resource "databricks_repo" "repo" {
  url    = var.repo_url
  branch = var.repo_branch
}

locals {
  node_type   = coalesce(var.job_cluster_node_type_id, data.databricks_node_type.smallest.id)
  spark_ver   = coalesce(var.job_cluster_spark_version, data.databricks_spark_version.latest_lts.id)
  repo_path   = databricks_repo.repo.path
}

# Optional: Secret scope for API keys
resource "databricks_secret_scope" "app" {
  name = "rag-voyage-demo"
}

resource "databricks_secret" "voyage" {
  count         = var.voyage_api_key == null ? 0 : 1
  scope         = databricks_secret_scope.app.name
  key           = "VOYAGE_API_KEY"
  string_value  = var.voyage_api_key
}

# Job: Build FAISS index (runs apps/cli/build_index.py)
resource "databricks_job" "index_job" {
  name = "rag-voyage-demo - Build FAISS Index"

  job_cluster {
    job_cluster_key = "jc"
    new_cluster {
      node_type_id  = local.node_type
      spark_version = local.spark_ver
      num_workers   = 1
    }
  }

  task {
    task_key        = "build_index"
    job_cluster_key = "jc"
    python_wheel_task {
      package_name = "rag-voyage-demo"
      entry_point  = "none" # not using entry points; we'll use python task below
    }
    # For simplicity, run python directly
    spark_python_task {
      python_file   = "${local.repo_path}/apps/cli/build_index.py"
      parameters    = []
    }
    environment_key = "env"
    environment {
      variables = {
        VOYAGE_API_KEY = "{{secrets/${databricks_secret_scope.app.name}/VOYAGE_API_KEY}}"
      }
    }
  }
}

# Job: Build BM25 index (runs scripts/build_bm25_index.py)
resource "databricks_job" "bm25_job" {
  name = "rag-voyage-demo - Build BM25 Index"

  job_cluster {
    job_cluster_key = "jc"
    new_cluster {
      node_type_id  = local.node_type
      spark_version = local.spark_ver
      num_workers   = 1
    }
  }

  task {
    task_key        = "build_bm25"
    job_cluster_key = "jc"
    spark_python_task {
      python_file   = "${local.repo_path}/scripts/build_bm25_index.py"
      parameters    = []
    }
    environment_key = "env"
    environment {
      variables = {
        VOYAGE_API_KEY = "{{secrets/${databricks_secret_scope.app.name}/VOYAGE_API_KEY}}"
      }
    }
  }
}
