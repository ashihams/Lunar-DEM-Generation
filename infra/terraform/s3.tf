resource "aws_s3_bucket" "input" {
  bucket = var.input_bucket_name
  tags = {
    Project = var.project_name
    Purpose = "dem-input"
  }
}

resource "aws_s3_bucket" "output" {
  bucket = var.output_bucket_name
  tags = {
    Project = var.project_name
    Purpose = "dem-output"
  }
}


