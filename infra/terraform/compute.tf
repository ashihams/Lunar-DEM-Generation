data "aws_ami" "linux" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["al2023-ami-*-x86_64"]
  }
}

locals {
  ami_id_effective = coalesce(var.ami_id, data.aws_ami.linux.id)
}

resource "aws_instance" "compute" {
  ami                         = local.ami_id_effective
  instance_type               = var.instance_type
  subnet_id                   = aws_subnet.public.id
  vpc_security_group_ids      = [aws_security_group.instance_sg.id]
  associate_public_ip_address = true
  key_name                    = var.key_name
  user_data_base64            = base64encode(templatefile("${path.module}/user_data.sh.tftpl", {
    region           = var.aws_region
    s3_input_bucket  = var.input_bucket_name
    s3_output_bucket = var.output_bucket_name
    docker_image     = var.pipeline_docker_image
  }))

  tags = {
    Name    = "${var.project_name}-compute"
    Project = var.project_name
  }
}

output "public_ip" { value = aws_instance.compute.public_ip }


