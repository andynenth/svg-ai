# EKS Cluster Configuration

module "eks" {
  source = "terraform-aws-modules/eks/aws"
  version = "~> 19.15"

  cluster_name    = local.cluster_name
  cluster_version = var.cluster_version

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # Cluster endpoint configuration
  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true
  cluster_endpoint_public_access_cidrs = var.allowed_cidr_blocks

  # Cluster logging
  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  cloudwatch_log_group_retention_in_days = var.monitoring_retention_days

  # Cluster security group
  cluster_additional_security_group_ids = [aws_security_group.additional.id]

  # EKS Managed Node Groups
  eks_managed_node_groups = {
    # API nodes - optimized for API workloads
    api = {
      name = "${local.cluster_name}-api"

      instance_types = var.instance_types.api
      capacity_type  = "ON_DEMAND"

      min_size     = var.min_size.api
      max_size     = var.max_size.api
      desired_size = var.desired_size.api

      ami_type = "AL2_x86_64"
      platform = "linux"

      subnet_ids = module.vpc.private_subnets

      # Launch template configuration
      create_launch_template = true
      launch_template_name   = "${local.cluster_name}-api"

      # Node group configuration
      labels = {
        NodeType = "api"
        Environment = var.environment
      }

      taints = []

      # Instance configuration
      disk_size = 50
      disk_type = "gp3"

      # Auto Scaling Group tags
      tags = merge(local.common_tags, {
        "k8s.io/cluster-autoscaler/enabled" = "true"
        "k8s.io/cluster-autoscaler/${local.cluster_name}" = "owned"
        "NodeType" = "api"
      })
    }

    # Worker nodes - optimized for compute workloads
    worker = {
      name = "${local.cluster_name}-worker"

      instance_types = var.instance_types.worker
      capacity_type  = "SPOT"  # Cost optimization for workers

      min_size     = var.min_size.worker
      max_size     = var.max_size.worker
      desired_size = var.desired_size.worker

      ami_type = "AL2_x86_64"
      platform = "linux"

      subnet_ids = module.vpc.private_subnets

      create_launch_template = true
      launch_template_name   = "${local.cluster_name}-worker"

      labels = {
        NodeType = "worker"
        Environment = var.environment
      }

      taints = [
        {
          key    = "worker"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]

      disk_size = 100
      disk_type = "gp3"

      tags = merge(local.common_tags, {
        "k8s.io/cluster-autoscaler/enabled" = "true"
        "k8s.io/cluster-autoscaler/${local.cluster_name}" = "owned"
        "NodeType" = "worker"
      })
    }

    # System nodes - for system components
    system = {
      name = "${local.cluster_name}-system"

      instance_types = var.instance_types.system
      capacity_type  = "ON_DEMAND"

      min_size     = var.min_size.system
      max_size     = var.max_size.system
      desired_size = var.desired_size.system

      ami_type = "AL2_x86_64"
      platform = "linux"

      subnet_ids = module.vpc.private_subnets

      create_launch_template = true
      launch_template_name   = "${local.cluster_name}-system"

      labels = {
        NodeType = "system"
        Environment = var.environment
      }

      taints = [
        {
          key    = "system"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]

      disk_size = 30
      disk_type = "gp3"

      tags = merge(local.common_tags, {
        "k8s.io/cluster-autoscaler/enabled" = "true"
        "k8s.io/cluster-autoscaler/${local.cluster_name}" = "owned"
        "NodeType" = "system"
      })
    }
  }

  # EKS Addons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }

  # IAM role for service accounts
  enable_irsa = true

  # Access management
  manage_aws_auth_configmap = true
  create_aws_auth_configmap = true

  aws_auth_users = [
    {
      userarn  = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
      username = "admin"
      groups   = ["system:masters"]
    }
  ]

  tags = local.common_tags
}

# IRSA for AWS Load Balancer Controller
module "load_balancer_controller_irsa_role" {
  source = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.20"

  role_name = "${local.cluster_name}-aws-load-balancer-controller"

  attach_load_balancer_controller_policy = true

  oidc_providers = {
    ex = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:aws-load-balancer-controller"]
    }
  }

  tags = local.common_tags
}

# IRSA for Cluster Autoscaler
module "cluster_autoscaler_irsa_role" {
  source = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.20"

  role_name = "${local.cluster_name}-cluster-autoscaler"

  attach_cluster_autoscaler_policy = true
  cluster_autoscaler_cluster_names = [local.cluster_name]

  oidc_providers = {
    ex = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:cluster-autoscaler"]
    }
  }

  tags = local.common_tags
}