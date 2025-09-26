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
      spark_env_vars = {
        VOYAGE_API_KEY = "{{secrets/${databricks_secret_scope.app.name}/VOYAGE_API_KEY}}"
      }
      # init_scripts removed: DBFS init scripts EOL. We'll rely on ML runtime and library.
    }
  }

  task {
    task_key        = "build_index"
    job_cluster_key = "jc"

    library {
      pypi {
        package = "faiss-cpu==1.8.0"
      }
    }
    library {
      pypi {
        package = "voyageai>=0.2.1"
      }
    }
    library {
      pypi {
        package = "rank-bm25>=0.2.2"
      }
    }
    library {
      pypi {
        package = "transformers==4.36.0"
      }
    }
    library {
      pypi {
        package = "sentence-transformers>=2.2.2"
      }
    }
    library {
      pypi {
        package = "torch==2.2.2"
      }
    }
    library {
      pypi {
        package = "tqdm>=4.66.2"
      }
    }
    library {
      pypi {
        package = "python-dotenv>=1.0.1"
      }
    }
    library {
      pypi {
        package = "pyyaml>=6.0.1"
      }
    }
    library {
      pypi {
        package = "colbert-ai>=0.2.19"
      }
    }
    library {
      pypi {
        package = "flask>=3.0.0"
      }
    }

    spark_python_task {
      python_file   = "${local.repo_path}/apps/cli/build_index.py"
      parameters    = []
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
      spark_env_vars = {
        VOYAGE_API_KEY = "{{secrets/${databricks_secret_scope.app.name}/VOYAGE_API_KEY}}"
      }
      # init_scripts removed: DBFS init scripts EOL. We'll rely on ML runtime and library.
    }
  }

  task {
    task_key        = "build_bm25"
    job_cluster_key = "jc"

    library {
      pypi {
        package = "faiss-cpu==1.8.0"
      }
    }
    library {
      pypi {
        package = "voyageai>=0.2.1"
      }
    }
    library {
      pypi {
        package = "rank-bm25>=0.2.2"
      }
    }
    library {
      pypi {
        package = "transformers==4.36.0"
      }
    }
    library {
      pypi {
        package = "sentence-transformers>=2.2.2"
      }
    }
    library {
      pypi {
        package = "torch==2.2.2"
      }
    }
    library {
      pypi {
        package = "tqdm>=4.66.2"
      }
    }
    library {
      pypi {
        package = "python-dotenv>=1.0.1"
      }
    }
    library {
      pypi {
        package = "pyyaml>=6.0.1"
      }
    }
    library {
      pypi {
        package = "colbert-ai>=0.2.19"
      }
    }
    library {
      pypi {
        package = "flask>=3.0.0"
      }
    }

    spark_python_task {
      python_file   = "${local.repo_path}/scripts/build_bm25_index.py"
      parameters    = []
    }
  }
}
