terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Optional: use S3 remote backend for state (uncomment and set values)
  # backend "s3" {
  #   bucket  = "your-terraform-state-bucket-name"
  #   key     = "dem-project/terraform.tfstate"
  #   region  = "us-east-1"
  #   encrypt = true
  # }
}

provider "aws" {
  region = var.aws_region
}

# Resources are defined in separate *.tf files in this module.



