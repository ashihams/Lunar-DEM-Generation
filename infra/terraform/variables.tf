variable "project_name" {
  description = "Name prefix for resources"
  type        = string
  default     = "lunar-dem"
}

variable "aws_region" {
  description = "AWS region to deploy to"
  type        = string
  default     = "ap-south-1"
}

variable "input_bucket_name" {
  description = "S3 bucket name for input/raw data"
  type        = string
}

variable "output_bucket_name" {
  description = "S3 bucket name for output/processed data"
  type        = string
}

variable "vpc_cidr_block" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.20.0.0/16"
}

variable "public_subnet_cidr" {
  description = "CIDR block for public subnet"
  type        = string
  default     = "10.20.1.0/24"
}

variable "availability_zone" {
  description = "Availability zone for subnet"
  type        = string
  default     = "ap-south-1a"
}

variable "instance_type" {
  description = "EC2 instance type for compute"
  type        = string
  default     = "c6i.2xlarge"
}

variable "ami_id" {
  description = "AMI ID for compute instances"
  type        = string
}

variable "key_name" {
  description = "EC2 key pair name for SSH access"
  type        = string
  default     = null
}

variable "pipeline_docker_image" {
  description = "Docker image name:tag to run pipeline (from Dockerfile)"
  type        = string
  default     = "lunar-pipeline:cuda118"
}


