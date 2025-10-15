output "input_bucket_name" {
  value       = var.input_bucket_name
  description = "Input S3 bucket"
}

output "output_bucket_name" {
  value       = var.output_bucket_name
  description = "Output S3 bucket"
}

output "vpc_id" {
  value       = local.vpc_id
  description = "VPC ID"
}

output "public_subnet_id" {
  value       = aws_subnet.public.id
  description = "Public subnet ID"
}

output "security_group_id" {
  value       = aws_security_group.instance_sg.id
  description = "Instance security group ID"
}

output "instance_public_ip" {
  value       = aws_instance.compute.public_ip
  description = "Public IP of the compute instance"
}
