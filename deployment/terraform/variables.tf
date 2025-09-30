# Input variables for SVG-AI infrastructure

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "svg-ai"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "prod"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-west-2"
}

variable "cluster_version" {
  description = "Kubernetes cluster version"
  type        = string
  default     = "1.27"
}

variable "instance_types" {
  description = "EC2 instance types for EKS node groups"
  type        = map(list(string))
  default = {
    api    = ["t3.medium", "t3.large"]
    worker = ["c5.large", "c5.xlarge"]
    system = ["t3.small", "t3.medium"]
  }
}

variable "min_size" {
  description = "Minimum number of nodes in each node group"
  type        = map(number)
  default = {
    api    = 2
    worker = 1
    system = 1
  }
}

variable "max_size" {
  description = "Maximum number of nodes in each node group"
  type        = map(number)
  default = {
    api    = 10
    worker = 8
    system = 3
  }
}

variable "desired_size" {
  description = "Desired number of nodes in each node group"
  type        = map(number)
  default = {
    api    = 3
    worker = 2
    system = 2
  }
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.medium"
}

variable "db_allocated_storage" {
  description = "RDS allocated storage in GB"
  type        = number
  default     = 100
}

variable "redis_node_type" {
  description = "ElastiCache Redis node type"
  type        = string
  default     = "cache.t3.medium"
}

variable "redis_num_cache_nodes" {
  description = "Number of Redis cache nodes"
  type        = number
  default     = 2
}

variable "enable_deletion_protection" {
  description = "Enable deletion protection for critical resources"
  type        = bool
  default     = true
}

variable "backup_retention_period" {
  description = "RDS backup retention period in days"
  type        = number
  default     = 7
}

variable "monitoring_retention_days" {
  description = "CloudWatch logs retention period in days"
  type        = number
  default     = 14
}

variable "ssl_certificate_arn" {
  description = "SSL certificate ARN for load balancer"
  type        = string
  default     = ""
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access the infrastructure"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Restrict this in production
}

variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = "svg-ai.com"
}

variable "route53_zone_id" {
  description = "Route53 hosted zone ID"
  type        = string
  default     = ""
}

variable "enable_nat_gateway" {
  description = "Enable NAT gateway for private subnets"
  type        = bool
  default     = true
}

variable "enable_dns_hostnames" {
  description = "Enable DNS hostnames in VPC"
  type        = bool
  default     = true
}

variable "enable_dns_support" {
  description = "Enable DNS support in VPC"
  type        = bool
  default     = true
}

variable "cost_optimization" {
  description = "Enable cost optimization features"
  type        = bool
  default     = true
}