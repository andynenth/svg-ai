#!/bin/bash
# Restore script for cleanup from 20250930_200223
# This script will restore all files removed during cleanup

set -e  # Exit on any error

BACKUP_DIR="cleanup_backup/20250930_200223"

if [ ! -d "$BACKUP_DIR" ]; then
    echo "Error: Backup directory not found: $BACKUP_DIR"
    exit 1
fi

echo "🔄 Restoring 175 files from cleanup 20250930_200223..."
echo "Backup directory: $BACKUP_DIR"
echo

RESTORED=0
FAILED=0


# Restore: /Users/nrw/python/svg-ai/backend/__init__.py
if [ -f "cleanup_backup/20250930_200223/backend/__init__.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/__init__.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/__init__.py")"
    cp "cleanup_backup/20250930_200223/backend/__init__.py" "/Users/nrw/python/svg-ai/backend/__init__.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/__init__.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/__init__.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/__init__.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/__init__.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/__init__.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/__init__.py" "/Users/nrw/python/svg-ai/backend/ai_modules/__init__.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/__init__.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/base_ai_converter.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/base_ai_converter.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/base_ai_converter.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/base_ai_converter.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/base_ai_converter.py" "/Users/nrw/python/svg-ai/backend/ai_modules/base_ai_converter.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/base_ai_converter.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/classification/__init__.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/classification/__init__.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/classification/__init__.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/classification/__init__.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/classification/__init__.py" "/Users/nrw/python/svg-ai/backend/ai_modules/classification/__init__.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/classification/__init__.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/classification/efficientnet_classifier_fixed.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/classification/efficientnet_classifier_fixed.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/classification/efficientnet_classifier_fixed.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/classification/efficientnet_classifier_fixed.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/classification/efficientnet_classifier_fixed.py" "/Users/nrw/python/svg-ai/backend/ai_modules/classification/efficientnet_classifier_fixed.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/classification/efficientnet_classifier_fixed.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/classification/logo_classifier.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/classification/logo_classifier.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/classification/logo_classifier.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/classification/logo_classifier.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/classification/logo_classifier.py" "/Users/nrw/python/svg-ai/backend/ai_modules/classification/logo_classifier.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/classification/logo_classifier.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/feature_pipeline.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/feature_pipeline.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/feature_pipeline.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/feature_pipeline.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/feature_pipeline.py" "/Users/nrw/python/svg-ai/backend/ai_modules/feature_pipeline.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/feature_pipeline.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/management/memory_monitor.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/management/memory_monitor.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/management/memory_monitor.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/management/memory_monitor.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/management/memory_monitor.py" "/Users/nrw/python/svg-ai/backend/ai_modules/management/memory_monitor.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/management/memory_monitor.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/management/model_cache.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/management/model_cache.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/management/model_cache.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/management/model_cache.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/management/model_cache.py" "/Users/nrw/python/svg-ai/backend/ai_modules/management/model_cache.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/management/model_cache.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/__init__.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/__init__.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/__init__.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/__init__.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/__init__.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/__init__.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/__init__.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/action_mapping.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/action_mapping.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/action_mapping.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/action_mapping.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/action_mapping.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/action_mapping.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/action_mapping.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/correlation_formulas_old.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/correlation_formulas_old.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/correlation_formulas_old.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/correlation_formulas_old.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/correlation_formulas_old.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/correlation_formulas_old.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/correlation_formulas_old.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/correlation_rollout.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/correlation_rollout.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/correlation_rollout.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/correlation_rollout.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/correlation_rollout.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/correlation_rollout.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/correlation_rollout.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/correlation_visualizer.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/correlation_visualizer.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/correlation_visualizer.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/correlation_visualizer.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/correlation_visualizer.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/correlation_visualizer.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/correlation_visualizer.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/day12_gpu_training_executor.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/day12_gpu_training_executor.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/day12_gpu_training_executor.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/day12_gpu_training_executor.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/day12_gpu_training_executor.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/day12_gpu_training_executor.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/day12_gpu_training_executor.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/day12_hyperparameter_optimizer.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/day12_hyperparameter_optimizer.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/day12_hyperparameter_optimizer.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/day12_hyperparameter_optimizer.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/day12_hyperparameter_optimizer.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/day12_hyperparameter_optimizer.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/day12_hyperparameter_optimizer.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/day12_model_export_manager.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/day12_model_export_manager.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/day12_model_export_manager.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/day12_model_export_manager.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/day12_model_export_manager.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/day12_model_export_manager.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/day12_model_export_manager.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/day12_model_validator.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/day12_model_validator.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/day12_model_validator.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/day12_model_validator.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/day12_model_validator.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/day12_model_validator.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/day12_model_validator.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/deployment_readiness_validator.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/deployment_readiness_validator.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/deployment_readiness_validator.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/deployment_readiness_validator.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/deployment_readiness_validator.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/deployment_readiness_validator.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/deployment_readiness_validator.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/feedback_integrator.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/feedback_integrator.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/feedback_integrator.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/feedback_integrator.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/feedback_integrator.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/feedback_integrator.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/feedback_integrator.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/learned_optimizer.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/learned_optimizer.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/learned_optimizer.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/learned_optimizer.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/learned_optimizer.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/learned_optimizer.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/learned_optimizer.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/online_learner.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/online_learner.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/online_learner.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/online_learner.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/online_learner.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/online_learner.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/online_learner.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/optimization_logger.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/optimization_logger.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/optimization_logger.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/optimization_logger.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/optimization_logger.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/optimization_logger.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/optimization_logger.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/parameter_tuner.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/parameter_tuner.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/parameter_tuner.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/parameter_tuner.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/parameter_tuner.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/parameter_tuner.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/parameter_tuner.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/quality_metrics.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/quality_metrics.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/quality_metrics.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/quality_metrics.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/quality_metrics.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/quality_metrics.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/quality_metrics.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/quality_validator.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/quality_validator.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/quality_validator.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/quality_validator.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/quality_validator.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/quality_validator.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/quality_validator.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/rl_optimizer.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/rl_optimizer.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/rl_optimizer.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/rl_optimizer.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/rl_optimizer.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/rl_optimizer.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/rl_optimizer.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/stage1_monitoring_dashboard.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/stage1_monitoring_dashboard.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/stage1_monitoring_dashboard.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/stage1_monitoring_dashboard.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/stage1_monitoring_dashboard.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/stage1_monitoring_dashboard.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/stage1_monitoring_dashboard.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/stage1_training_executor.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/stage1_training_executor.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/stage1_training_executor.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/stage1_training_executor.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/stage1_training_executor.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/stage1_training_executor.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/stage1_training_executor.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/task_12_2_master_pipeline.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/task_12_2_master_pipeline.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/task_12_2_master_pipeline.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/task_12_2_master_pipeline.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/task_12_2_master_pipeline.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/task_12_2_master_pipeline.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/task_12_2_master_pipeline.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/test_enhanced_router_system.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/test_enhanced_router_system.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/test_enhanced_router_system.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/test_enhanced_router_system.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/test_enhanced_router_system.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/test_enhanced_router_system.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/test_enhanced_router_system.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/tests/optimization/test_feature_mapping_optimizer.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/tests/optimization/test_feature_mapping_optimizer.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/tests/optimization/test_feature_mapping_optimizer.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/tests/optimization/test_feature_mapping_optimizer.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/tests/optimization/test_feature_mapping_optimizer.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/tests/optimization/test_feature_mapping_optimizer.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/tests/optimization/test_feature_mapping_optimizer.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/training_execution_engine.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/training_execution_engine.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/training_execution_engine.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/training_execution_engine.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/training_execution_engine.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/training_execution_engine.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/training_execution_engine.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/training_monitor.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/training_monitor.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/training_monitor.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/training_monitor.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/training_monitor.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/training_monitor.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/training_monitor.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/training_orchestrator.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/training_orchestrator.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/training_orchestrator.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/training_orchestrator.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/training_orchestrator.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/training_orchestrator.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/training_orchestrator.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/training_pipeline.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/training_pipeline.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/training_pipeline.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/training_pipeline.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/training_pipeline.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/training_pipeline.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/training_pipeline.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/training_visualizer.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/training_visualizer.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/training_visualizer.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/training_visualizer.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/training_visualizer.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/training_visualizer.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/training_visualizer.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/validation_framework.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/validation_framework.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/validation_framework.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/validation_framework.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/validation_framework.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/validation_framework.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/validation_framework.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/validator.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/validator.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/validator.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/validator.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/validator.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/validator.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/validator.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/pipeline/__init__.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/pipeline/__init__.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/pipeline/__init__.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/pipeline/__init__.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/pipeline/__init__.py" "/Users/nrw/python/svg-ai/backend/ai_modules/pipeline/__init__.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/pipeline/__init__.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/pipeline/component_interfaces.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/pipeline/component_interfaces.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/pipeline/component_interfaces.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/pipeline/component_interfaces.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/pipeline/component_interfaces.py" "/Users/nrw/python/svg-ai/backend/ai_modules/pipeline/component_interfaces.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/pipeline/component_interfaces.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/pipeline/pipeline_config.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/pipeline/pipeline_config.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/pipeline/pipeline_config.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/pipeline/pipeline_config.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/pipeline/pipeline_config.py" "/Users/nrw/python/svg-ai/backend/ai_modules/pipeline/pipeline_config.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/pipeline/pipeline_config.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/pipeline/pipeline_monitor.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/pipeline/pipeline_monitor.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/pipeline/pipeline_monitor.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/pipeline/pipeline_monitor.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/pipeline/pipeline_monitor.py" "/Users/nrw/python/svg-ai/backend/ai_modules/pipeline/pipeline_monitor.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/pipeline/pipeline_monitor.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/prediction/__init__.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/prediction/__init__.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/prediction/__init__.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/prediction/__init__.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/prediction/__init__.py" "/Users/nrw/python/svg-ai/backend/ai_modules/prediction/__init__.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/prediction/__init__.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/prediction/model_utils.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/prediction/model_utils.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/prediction/model_utils.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/prediction/model_utils.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/prediction/model_utils.py" "/Users/nrw/python/svg-ai/backend/ai_modules/prediction/model_utils.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/prediction/model_utils.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/prediction/quality_predictor.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/prediction/quality_predictor.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/prediction/quality_predictor.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/prediction/quality_predictor.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/prediction/quality_predictor.py" "/Users/nrw/python/svg-ai/backend/ai_modules/prediction/quality_predictor.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/prediction/quality_predictor.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/production_readiness.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/production_readiness.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/production_readiness.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/production_readiness.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/production_readiness.py" "/Users/nrw/python/svg-ai/backend/ai_modules/production_readiness.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/production_readiness.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/quality/ab_testing.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/quality/ab_testing.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/quality/ab_testing.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/quality/ab_testing.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/quality/ab_testing.py" "/Users/nrw/python/svg-ai/backend/ai_modules/quality/ab_testing.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/quality/ab_testing.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/quality/realtime_monitor.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/quality/realtime_monitor.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/quality/realtime_monitor.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/quality/realtime_monitor.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/quality/realtime_monitor.py" "/Users/nrw/python/svg-ai/backend/ai_modules/quality/realtime_monitor.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/quality/realtime_monitor.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/routing/__init__.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/routing/__init__.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/routing/__init__.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/routing/__init__.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/routing/__init__.py" "/Users/nrw/python/svg-ai/backend/ai_modules/routing/__init__.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/routing/__init__.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/routing/adaptive_router.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/routing/adaptive_router.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/routing/adaptive_router.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/routing/adaptive_router.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/routing/adaptive_router.py" "/Users/nrw/python/svg-ai/backend/ai_modules/routing/adaptive_router.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/routing/adaptive_router.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/routing/complexity_analyzer.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/routing/complexity_analyzer.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/routing/complexity_analyzer.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/routing/complexity_analyzer.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/routing/complexity_analyzer.py" "/Users/nrw/python/svg-ai/backend/ai_modules/routing/complexity_analyzer.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/routing/complexity_analyzer.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/routing/intelligent_tier_selector.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/routing/intelligent_tier_selector.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/routing/intelligent_tier_selector.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/routing/intelligent_tier_selector.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/routing/intelligent_tier_selector.py" "/Users/nrw/python/svg-ai/backend/ai_modules/routing/intelligent_tier_selector.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/routing/intelligent_tier_selector.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/routing/routing_analytics.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/routing/routing_analytics.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/routing/routing_analytics.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/routing/routing_analytics.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/routing/routing_analytics.py" "/Users/nrw/python/svg-ai/backend/ai_modules/routing/routing_analytics.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/routing/routing_analytics.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/routing/routing_config.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/routing/routing_config.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/routing/routing_config.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/routing/routing_config.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/routing/routing_config.py" "/Users/nrw/python/svg-ai/backend/ai_modules/routing/routing_config.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/routing/routing_config.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/rule_based_classifier_refactored.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/rule_based_classifier_refactored.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/rule_based_classifier_refactored.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/rule_based_classifier_refactored.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/rule_based_classifier_refactored.py" "/Users/nrw/python/svg-ai/backend/ai_modules/rule_based_classifier_refactored.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/rule_based_classifier_refactored.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/testing/__init__.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/testing/__init__.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/testing/__init__.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/testing/__init__.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/testing/__init__.py" "/Users/nrw/python/svg-ai/backend/ai_modules/testing/__init__.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/testing/__init__.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/testing/report_generator.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/testing/report_generator.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/testing/report_generator.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/testing/report_generator.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/testing/report_generator.py" "/Users/nrw/python/svg-ai/backend/ai_modules/testing/report_generator.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/testing/report_generator.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/testing/test_orchestrator.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/testing/test_orchestrator.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/testing/test_orchestrator.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/testing/test_orchestrator.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/testing/test_orchestrator.py" "/Users/nrw/python/svg-ai/backend/ai_modules/testing/test_orchestrator.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/testing/test_orchestrator.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/training/__init__.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/training/__init__.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/training/__init__.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/training/__init__.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/training/__init__.py" "/Users/nrw/python/svg-ai/backend/ai_modules/training/__init__.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/training/__init__.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/training/logo_dataset.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/training/logo_dataset.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/training/logo_dataset.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/training/logo_dataset.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/training/logo_dataset.py" "/Users/nrw/python/svg-ai/backend/ai_modules/training/logo_dataset.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/training/logo_dataset.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/utils/__init__.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/utils/__init__.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/utils/__init__.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/utils/__init__.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/utils/__init__.py" "/Users/nrw/python/svg-ai/backend/ai_modules/utils/__init__.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/utils/__init__.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/utils/cache_manager.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/utils/cache_manager.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/utils/cache_manager.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/utils/cache_manager.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/utils/cache_manager.py" "/Users/nrw/python/svg-ai/backend/ai_modules/utils/cache_manager.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/utils/cache_manager.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/utils/lazy_loader.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/utils/lazy_loader.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/utils/lazy_loader.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/utils/lazy_loader.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/utils/lazy_loader.py" "/Users/nrw/python/svg-ai/backend/ai_modules/utils/lazy_loader.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/utils/lazy_loader.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/utils/parallel_processor.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/utils/parallel_processor.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/utils/parallel_processor.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/utils/parallel_processor.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/utils/parallel_processor.py" "/Users/nrw/python/svg-ai/backend/ai_modules/utils/parallel_processor.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/utils/parallel_processor.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/utils/performance_monitor.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/utils/performance_monitor.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/utils/performance_monitor.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/utils/performance_monitor.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/utils/performance_monitor.py" "/Users/nrw/python/svg-ai/backend/ai_modules/utils/performance_monitor.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/utils/performance_monitor.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/utils/profiler.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/utils/profiler.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/utils/profiler.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/utils/profiler.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/utils/profiler.py" "/Users/nrw/python/svg-ai/backend/ai_modules/utils/profiler.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/utils/profiler.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/api/monitoring_api.py
if [ -f "cleanup_backup/20250930_200223/backend/api/monitoring_api.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/api/monitoring_api.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/api/monitoring_api.py")"
    cp "cleanup_backup/20250930_200223/backend/api/monitoring_api.py" "/Users/nrw/python/svg-ai/backend/api/monitoring_api.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/api/monitoring_api.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/api/optimization_api.py
if [ -f "cleanup_backup/20250930_200223/backend/api/optimization_api.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/api/optimization_api.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/api/optimization_api.py")"
    cp "cleanup_backup/20250930_200223/backend/api/optimization_api.py" "/Users/nrw/python/svg-ai/backend/api/optimization_api.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/api/optimization_api.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/api/unified_optimization_api.py
if [ -f "cleanup_backup/20250930_200223/backend/api/unified_optimization_api.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/api/unified_optimization_api.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/api/unified_optimization_api.py")"
    cp "cleanup_backup/20250930_200223/backend/api/unified_optimization_api.py" "/Users/nrw/python/svg-ai/backend/api/unified_optimization_api.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/api/unified_optimization_api.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/config.py
if [ -f "cleanup_backup/20250930_200223/backend/config.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/config.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/config.py")"
    cp "cleanup_backup/20250930_200223/backend/config.py" "/Users/nrw/python/svg-ai/backend/config.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/config.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/converters/__init__.py
if [ -f "cleanup_backup/20250930_200223/backend/converters/__init__.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/converters/__init__.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/converters/__init__.py")"
    cp "cleanup_backup/20250930_200223/backend/converters/__init__.py" "/Users/nrw/python/svg-ai/backend/converters/__init__.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/converters/__init__.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/converters/intelligent_converter.py
if [ -f "cleanup_backup/20250930_200223/backend/converters/intelligent_converter.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/converters/intelligent_converter.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/converters/intelligent_converter.py")"
    cp "cleanup_backup/20250930_200223/backend/converters/intelligent_converter.py" "/Users/nrw/python/svg-ai/backend/converters/intelligent_converter.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/converters/intelligent_converter.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/converters/mock_converter.py
if [ -f "cleanup_backup/20250930_200223/backend/converters/mock_converter.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/converters/mock_converter.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/converters/mock_converter.py")"
    cp "cleanup_backup/20250930_200223/backend/converters/mock_converter.py" "/Users/nrw/python/svg-ai/backend/converters/mock_converter.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/converters/mock_converter.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/integration/tier4_pipeline_integration.py
if [ -f "cleanup_backup/20250930_200223/backend/integration/tier4_pipeline_integration.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/integration/tier4_pipeline_integration.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/integration/tier4_pipeline_integration.py")"
    cp "cleanup_backup/20250930_200223/backend/integration/tier4_pipeline_integration.py" "/Users/nrw/python/svg-ai/backend/integration/tier4_pipeline_integration.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/integration/tier4_pipeline_integration.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/utils/__init__.py
if [ -f "cleanup_backup/20250930_200223/backend/utils/__init__.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/utils/__init__.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/utils/__init__.py")"
    cp "cleanup_backup/20250930_200223/backend/utils/__init__.py" "/Users/nrw/python/svg-ai/backend/utils/__init__.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/utils/__init__.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/utils/cache.py
if [ -f "cleanup_backup/20250930_200223/backend/utils/cache.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/utils/cache.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/utils/cache.py")"
    cp "cleanup_backup/20250930_200223/backend/utils/cache.py" "/Users/nrw/python/svg-ai/backend/utils/cache.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/utils/cache.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/utils/error_messages.py
if [ -f "cleanup_backup/20250930_200223/backend/utils/error_messages.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/utils/error_messages.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/utils/error_messages.py")"
    cp "cleanup_backup/20250930_200223/backend/utils/error_messages.py" "/Users/nrw/python/svg-ai/backend/utils/error_messages.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/utils/error_messages.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/utils/image_loader.py
if [ -f "cleanup_backup/20250930_200223/backend/utils/image_loader.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/utils/image_loader.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/utils/image_loader.py")"
    cp "cleanup_backup/20250930_200223/backend/utils/image_loader.py" "/Users/nrw/python/svg-ai/backend/utils/image_loader.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/utils/image_loader.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/utils/metrics.py
if [ -f "cleanup_backup/20250930_200223/backend/utils/metrics.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/utils/metrics.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/utils/metrics.py")"
    cp "cleanup_backup/20250930_200223/backend/utils/metrics.py" "/Users/nrw/python/svg-ai/backend/utils/metrics.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/utils/metrics.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/utils/parameter_cache.py
if [ -f "cleanup_backup/20250930_200223/backend/utils/parameter_cache.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/utils/parameter_cache.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/utils/parameter_cache.py")"
    cp "cleanup_backup/20250930_200223/backend/utils/parameter_cache.py" "/Users/nrw/python/svg-ai/backend/utils/parameter_cache.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/utils/parameter_cache.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/utils/preprocessor.py
if [ -f "cleanup_backup/20250930_200223/backend/utils/preprocessor.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/utils/preprocessor.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/utils/preprocessor.py")"
    cp "cleanup_backup/20250930_200223/backend/utils/preprocessor.py" "/Users/nrw/python/svg-ai/backend/utils/preprocessor.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/utils/preprocessor.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/utils/svg_optimizer.py
if [ -f "cleanup_backup/20250930_200223/backend/utils/svg_optimizer.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/utils/svg_optimizer.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/utils/svg_optimizer.py")"
    cp "cleanup_backup/20250930_200223/backend/utils/svg_optimizer.py" "/Users/nrw/python/svg-ai/backend/utils/svg_optimizer.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/utils/svg_optimizer.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/utils/svg_post_processor.py
if [ -f "cleanup_backup/20250930_200223/backend/utils/svg_post_processor.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/utils/svg_post_processor.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/utils/svg_post_processor.py")"
    cp "cleanup_backup/20250930_200223/backend/utils/svg_post_processor.py" "/Users/nrw/python/svg-ai/backend/utils/svg_post_processor.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/utils/svg_post_processor.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/utils/visual_compare.py
if [ -f "cleanup_backup/20250930_200223/backend/utils/visual_compare.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/utils/visual_compare.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/utils/visual_compare.py")"
    cp "cleanup_backup/20250930_200223/backend/utils/visual_compare.py" "/Users/nrw/python/svg-ai/backend/utils/visual_compare.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/utils/visual_compare.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/advanced_cache.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/advanced_cache.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/advanced_cache.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/advanced_cache.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/advanced_cache.py" "/Users/nrw/python/svg-ai/backend/ai_modules/advanced_cache.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/advanced_cache.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/analytics_dashboard.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/analytics_dashboard.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/analytics_dashboard.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/analytics_dashboard.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/analytics_dashboard.py" "/Users/nrw/python/svg-ai/backend/ai_modules/analytics_dashboard.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/analytics_dashboard.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/cache_monitor.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/cache_monitor.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/cache_monitor.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/cache_monitor.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/cache_monitor.py" "/Users/nrw/python/svg-ai/backend/ai_modules/cache_monitor.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/cache_monitor.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/cached_components.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/cached_components.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/cached_components.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/cached_components.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/cached_components.py" "/Users/nrw/python/svg-ai/backend/ai_modules/cached_components.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/cached_components.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/classification/base_feature_extractor.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/classification/base_feature_extractor.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/classification/base_feature_extractor.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/classification/base_feature_extractor.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/classification/base_feature_extractor.py" "/Users/nrw/python/svg-ai/backend/ai_modules/classification/base_feature_extractor.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/classification/base_feature_extractor.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/classification/efficientnet_classifier.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/classification/efficientnet_classifier.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/classification/efficientnet_classifier.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/classification/efficientnet_classifier.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/classification/efficientnet_classifier.py" "/Users/nrw/python/svg-ai/backend/ai_modules/classification/efficientnet_classifier.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/classification/efficientnet_classifier.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/classification/feature_extractor.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/classification/feature_extractor.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/classification/feature_extractor.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/classification/feature_extractor.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/classification/feature_extractor.py" "/Users/nrw/python/svg-ai/backend/ai_modules/classification/feature_extractor.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/classification/feature_extractor.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/classification/hybrid_classifier.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/classification/hybrid_classifier.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/classification/hybrid_classifier.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/classification/hybrid_classifier.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/classification/hybrid_classifier.py" "/Users/nrw/python/svg-ai/backend/ai_modules/classification/hybrid_classifier.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/classification/hybrid_classifier.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/classification/rule_based_classifier.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/classification/rule_based_classifier.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/classification/rule_based_classifier.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/classification/rule_based_classifier.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/classification/rule_based_classifier.py" "/Users/nrw/python/svg-ai/backend/ai_modules/classification/rule_based_classifier.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/classification/rule_based_classifier.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/classification/statistical_classifier.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/classification/statistical_classifier.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/classification/statistical_classifier.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/classification/statistical_classifier.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/classification/statistical_classifier.py" "/Users/nrw/python/svg-ai/backend/ai_modules/classification/statistical_classifier.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/classification/statistical_classifier.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/database_cache.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/database_cache.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/database_cache.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/database_cache.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/database_cache.py" "/Users/nrw/python/svg-ai/backend/ai_modules/database_cache.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/database_cache.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/feature_extraction.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/feature_extraction.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/feature_extraction.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/feature_extraction.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/feature_extraction.py" "/Users/nrw/python/svg-ai/backend/ai_modules/feature_extraction.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/feature_extraction.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/adaptive_optimizer.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/adaptive_optimizer.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/adaptive_optimizer.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/adaptive_optimizer.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/adaptive_optimizer.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/adaptive_optimizer.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/adaptive_optimizer.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/agent_interface.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/agent_interface.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/agent_interface.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/agent_interface.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/agent_interface.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/agent_interface.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/agent_interface.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/base_optimizer.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/base_optimizer.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/base_optimizer.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/base_optimizer.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/base_optimizer.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/base_optimizer.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/base_optimizer.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/checkpoint_manager.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/checkpoint_manager.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/checkpoint_manager.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/checkpoint_manager.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/checkpoint_manager.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/checkpoint_manager.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/checkpoint_manager.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/colab_persistence_manager.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/colab_persistence_manager.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/colab_persistence_manager.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/colab_persistence_manager.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/colab_persistence_manager.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/colab_persistence_manager.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/colab_persistence_manager.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/colab_training_visualization.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/colab_training_visualization.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/colab_training_visualization.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/colab_training_visualization.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/colab_training_visualization.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/colab_training_visualization.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/colab_training_visualization.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/correlation_formulas_backup.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/correlation_formulas_backup.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/correlation_formulas_backup.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/correlation_formulas_backup.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/correlation_formulas_backup.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/correlation_formulas_backup.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/correlation_formulas_backup.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/cpu_performance_optimizer.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/cpu_performance_optimizer.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/cpu_performance_optimizer.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/cpu_performance_optimizer.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/cpu_performance_optimizer.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/cpu_performance_optimizer.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/cpu_performance_optimizer.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/day12_training_visualization.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/day12_training_visualization.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/day12_training_visualization.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/day12_training_visualization.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/day12_training_visualization.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/day12_training_visualization.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/day12_training_visualization.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/day13_deployment_packager.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/day13_deployment_packager.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/day13_deployment_packager.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/day13_deployment_packager.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/day13_deployment_packager.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/day13_deployment_packager.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/day13_deployment_packager.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/day13_export_optimizer.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/day13_export_optimizer.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/day13_export_optimizer.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/day13_export_optimizer.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/day13_export_optimizer.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/day13_export_optimizer.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/day13_export_optimizer.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/day13_integration_tester.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/day13_integration_tester.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/day13_integration_tester.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/day13_integration_tester.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/day13_integration_tester.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/day13_integration_tester.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/day13_integration_tester.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/day13_performance_optimizer.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/day13_performance_optimizer.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/day13_performance_optimizer.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/day13_performance_optimizer.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/day13_performance_optimizer.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/day13_performance_optimizer.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/day13_performance_optimizer.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/end_to_end_validation.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/end_to_end_validation.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/end_to_end_validation.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/end_to_end_validation.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/end_to_end_validation.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/end_to_end_validation.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/end_to_end_validation.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/enhanced_intelligent_router.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/enhanced_intelligent_router.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/enhanced_intelligent_router.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/enhanced_intelligent_router.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/enhanced_intelligent_router.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/enhanced_intelligent_router.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/enhanced_intelligent_router.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/enhanced_performance_monitor.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/enhanced_performance_monitor.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/enhanced_performance_monitor.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/enhanced_performance_monitor.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/enhanced_performance_monitor.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/enhanced_performance_monitor.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/enhanced_performance_monitor.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/enhanced_router_integration.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/enhanced_router_integration.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/enhanced_router_integration.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/enhanced_router_integration.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/enhanced_router_integration.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/enhanced_router_integration.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/enhanced_router_integration.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/error_handler.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/error_handler.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/error_handler.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/error_handler.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/error_handler.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/error_handler.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/error_handler.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/export_validation_framework.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/export_validation_framework.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/export_validation_framework.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/export_validation_framework.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/export_validation_framework.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/export_validation_framework.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/export_validation_framework.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/feature_mapping.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/feature_mapping.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/feature_mapping.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/feature_mapping.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/feature_mapping.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/feature_mapping.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/feature_mapping.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/feature_mapping_optimizer.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/feature_mapping_optimizer.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/feature_mapping_optimizer.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/feature_mapping_optimizer.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/feature_mapping_optimizer.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/feature_mapping_optimizer.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/feature_mapping_optimizer.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/feature_mapping_optimizer_v2.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/feature_mapping_optimizer_v2.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/feature_mapping_optimizer_v2.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/feature_mapping_optimizer_v2.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/feature_mapping_optimizer_v2.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/feature_mapping_optimizer_v2.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/feature_mapping_optimizer_v2.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/gpu_model_architecture.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/gpu_model_architecture.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/gpu_model_architecture.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/gpu_model_architecture.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/gpu_model_architecture.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/gpu_model_architecture.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/gpu_model_architecture.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/gpu_training_pipeline.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/gpu_training_pipeline.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/gpu_training_pipeline.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/gpu_training_pipeline.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/gpu_training_pipeline.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/gpu_training_pipeline.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/gpu_training_pipeline.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/intelligent_router.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/intelligent_router.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/intelligent_router.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/intelligent_router.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/intelligent_router.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/intelligent_router.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/intelligent_router.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/learned_correlations.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/learned_correlations.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/learned_correlations.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/learned_correlations.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/learned_correlations.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/learned_correlations.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/learned_correlations.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/local_inference_optimizer.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/local_inference_optimizer.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/local_inference_optimizer.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/local_inference_optimizer.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/local_inference_optimizer.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/local_inference_optimizer.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/local_inference_optimizer.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/local_integration_tester.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/local_integration_tester.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/local_integration_tester.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/local_integration_tester.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/local_integration_tester.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/local_integration_tester.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/local_integration_tester.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/model_export_pipeline.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/model_export_pipeline.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/model_export_pipeline.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/model_export_pipeline.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/model_export_pipeline.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/model_export_pipeline.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/model_export_pipeline.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/parameter_router.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/parameter_router.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/parameter_router.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/parameter_router.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/parameter_router.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/parameter_router.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/parameter_router.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/performance_optimizer.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/performance_optimizer.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/performance_optimizer.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/performance_optimizer.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/performance_optimizer.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/performance_optimizer.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/performance_optimizer.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/performance_testing_framework.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/performance_testing_framework.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/performance_testing_framework.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/performance_testing_framework.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/performance_testing_framework.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/performance_testing_framework.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/performance_testing_framework.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/production_deployment_framework.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/production_deployment_framework.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/production_deployment_framework.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/production_deployment_framework.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/production_deployment_framework.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/production_deployment_framework.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/production_deployment_framework.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/production_deployment_package.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/production_deployment_package.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/production_deployment_package.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/production_deployment_package.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/production_deployment_package.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/production_deployment_package.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/production_deployment_package.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/quality_prediction_cache.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/quality_prediction_cache.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/quality_prediction_cache.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/quality_prediction_cache.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/quality_prediction_cache.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/quality_prediction_cache.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/quality_prediction_cache.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/quality_prediction_integration.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/quality_prediction_integration.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/quality_prediction_integration.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/quality_prediction_integration.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/quality_prediction_integration.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/quality_prediction_integration.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/quality_prediction_integration.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/real_time_monitor.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/real_time_monitor.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/real_time_monitor.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/real_time_monitor.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/real_time_monitor.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/real_time_monitor.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/real_time_monitor.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/refined_correlation_formulas.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/refined_correlation_formulas.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/refined_correlation_formulas.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/refined_correlation_formulas.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/refined_correlation_formulas.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/refined_correlation_formulas.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/refined_correlation_formulas.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/regression_optimizer.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/regression_optimizer.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/regression_optimizer.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/regression_optimizer.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/regression_optimizer.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/regression_optimizer.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/regression_optimizer.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/resource_monitor.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/resource_monitor.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/resource_monitor.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/resource_monitor.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/resource_monitor.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/resource_monitor.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/resource_monitor.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/reward_functions.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/reward_functions.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/reward_functions.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/reward_functions.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/reward_functions.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/reward_functions.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/reward_functions.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/spatial_analysis.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/spatial_analysis.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/spatial_analysis.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/spatial_analysis.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/spatial_analysis.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/spatial_analysis.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/spatial_analysis.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/statistical_parameter_predictor.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/statistical_parameter_predictor.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/statistical_parameter_predictor.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/statistical_parameter_predictor.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/statistical_parameter_predictor.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/statistical_parameter_predictor.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/statistical_parameter_predictor.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/system_monitoring_analytics.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/system_monitoring_analytics.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/system_monitoring_analytics.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/system_monitoring_analytics.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/system_monitoring_analytics.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/system_monitoring_analytics.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/system_monitoring_analytics.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/tier4_system_orchestrator.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/tier4_system_orchestrator.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/tier4_system_orchestrator.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/tier4_system_orchestrator.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/tier4_system_orchestrator.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/tier4_system_orchestrator.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/tier4_system_orchestrator.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/training_data_manager.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/training_data_manager.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/training_data_manager.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/training_data_manager.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/training_data_manager.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/training_data_manager.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/training_data_manager.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/unified_prediction_api.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/unified_prediction_api.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/unified_prediction_api.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/unified_prediction_api.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/unified_prediction_api.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/unified_prediction_api.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/unified_prediction_api.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/validation_pipeline.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/validation_pipeline.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/validation_pipeline.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/validation_pipeline.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/validation_pipeline.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/validation_pipeline.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/validation_pipeline.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/vtracer_env.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/vtracer_env.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/vtracer_env.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/vtracer_env.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/vtracer_env.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/vtracer_env.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/vtracer_env.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/vtracer_environment.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/vtracer_environment.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/vtracer_environment.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/vtracer_environment.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/vtracer_environment.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/vtracer_environment.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/vtracer_environment.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimization/vtracer_test.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimization/vtracer_test.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimization/vtracer_test.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/vtracer_test.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimization/vtracer_test.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimization/vtracer_test.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimization/vtracer_test.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/optimized_pipeline.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/optimized_pipeline.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/optimized_pipeline.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/optimized_pipeline.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/optimized_pipeline.py" "/Users/nrw/python/svg-ai/backend/ai_modules/optimized_pipeline.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/optimized_pipeline.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/parameter_optimizer.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/parameter_optimizer.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/parameter_optimizer.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/parameter_optimizer.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/parameter_optimizer.py" "/Users/nrw/python/svg-ai/backend/ai_modules/parameter_optimizer.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/parameter_optimizer.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/performance_profiler.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/performance_profiler.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/performance_profiler.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/performance_profiler.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/performance_profiler.py" "/Users/nrw/python/svg-ai/backend/ai_modules/performance_profiler.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/performance_profiler.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/pipeline/unified_ai_pipeline.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/pipeline/unified_ai_pipeline.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/pipeline/unified_ai_pipeline.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/pipeline/unified_ai_pipeline.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/pipeline/unified_ai_pipeline.py" "/Users/nrw/python/svg-ai/backend/ai_modules/pipeline/unified_ai_pipeline.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/pipeline/unified_ai_pipeline.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/prediction/base_predictor.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/prediction/base_predictor.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/prediction/base_predictor.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/prediction/base_predictor.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/prediction/base_predictor.py" "/Users/nrw/python/svg-ai/backend/ai_modules/prediction/base_predictor.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/prediction/base_predictor.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/prediction/statistical_quality_predictor.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/prediction/statistical_quality_predictor.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/prediction/statistical_quality_predictor.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/prediction/statistical_quality_predictor.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/prediction/statistical_quality_predictor.py" "/Users/nrw/python/svg-ai/backend/ai_modules/prediction/statistical_quality_predictor.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/prediction/statistical_quality_predictor.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/quality/enhanced_metrics.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/quality/enhanced_metrics.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/quality/enhanced_metrics.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/quality/enhanced_metrics.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/quality/enhanced_metrics.py" "/Users/nrw/python/svg-ai/backend/ai_modules/quality/enhanced_metrics.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/quality/enhanced_metrics.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/quality/quality_tracker.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/quality/quality_tracker.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/quality/quality_tracker.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/quality/quality_tracker.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/quality/quality_tracker.py" "/Users/nrw/python/svg-ai/backend/ai_modules/quality/quality_tracker.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/quality/quality_tracker.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/routing/hybrid_intelligent_router.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/routing/hybrid_intelligent_router.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/routing/hybrid_intelligent_router.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/routing/hybrid_intelligent_router.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/routing/hybrid_intelligent_router.py" "/Users/nrw/python/svg-ai/backend/ai_modules/routing/hybrid_intelligent_router.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/routing/hybrid_intelligent_router.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/rule_based_classifier.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/rule_based_classifier.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/rule_based_classifier.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/rule_based_classifier.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/rule_based_classifier.py" "/Users/nrw/python/svg-ai/backend/ai_modules/rule_based_classifier.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/rule_based_classifier.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/smart_cache.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/smart_cache.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/smart_cache.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/smart_cache.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/smart_cache.py" "/Users/nrw/python/svg-ai/backend/ai_modules/smart_cache.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/smart_cache.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/testing/ab_framework.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/testing/ab_framework.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/testing/ab_framework.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/testing/ab_framework.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/testing/ab_framework.py" "/Users/nrw/python/svg-ai/backend/ai_modules/testing/ab_framework.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/testing/ab_framework.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/testing/statistical_analysis.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/testing/statistical_analysis.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/testing/statistical_analysis.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/testing/statistical_analysis.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/testing/statistical_analysis.py" "/Users/nrw/python/svg-ai/backend/ai_modules/testing/statistical_analysis.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/testing/statistical_analysis.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/testing/visual_comparison.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/testing/visual_comparison.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/testing/visual_comparison.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/testing/visual_comparison.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/testing/visual_comparison.py" "/Users/nrw/python/svg-ai/backend/ai_modules/testing/visual_comparison.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/testing/visual_comparison.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/utils/logging_config.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/utils/logging_config.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/utils/logging_config.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/utils/logging_config.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/utils/logging_config.py" "/Users/nrw/python/svg-ai/backend/ai_modules/utils/logging_config.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/utils/logging_config.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/utils/model_adapter.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/utils/model_adapter.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/utils/model_adapter.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/utils/model_adapter.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/utils/model_adapter.py" "/Users/nrw/python/svg-ai/backend/ai_modules/utils/model_adapter.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/utils/model_adapter.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/ai_modules/utils/request_queue.py
if [ -f "cleanup_backup/20250930_200223/backend/ai_modules/utils/request_queue.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/ai_modules/utils/request_queue.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/ai_modules/utils/request_queue.py")"
    cp "cleanup_backup/20250930_200223/backend/ai_modules/utils/request_queue.py" "/Users/nrw/python/svg-ai/backend/ai_modules/utils/request_queue.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/ai_modules/utils/request_queue.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/converters/ai_enhanced_converter.py
if [ -f "cleanup_backup/20250930_200223/backend/converters/ai_enhanced_converter.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/converters/ai_enhanced_converter.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/converters/ai_enhanced_converter.py")"
    cp "cleanup_backup/20250930_200223/backend/converters/ai_enhanced_converter.py" "/Users/nrw/python/svg-ai/backend/converters/ai_enhanced_converter.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/converters/ai_enhanced_converter.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/converters/alpha_converter.py
if [ -f "cleanup_backup/20250930_200223/backend/converters/alpha_converter.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/converters/alpha_converter.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/converters/alpha_converter.py")"
    cp "cleanup_backup/20250930_200223/backend/converters/alpha_converter.py" "/Users/nrw/python/svg-ai/backend/converters/alpha_converter.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/converters/alpha_converter.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/converters/base.py
if [ -f "cleanup_backup/20250930_200223/backend/converters/base.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/converters/base.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/converters/base.py")"
    cp "cleanup_backup/20250930_200223/backend/converters/base.py" "/Users/nrw/python/svg-ai/backend/converters/base.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/converters/base.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/converters/potrace_converter.py
if [ -f "cleanup_backup/20250930_200223/backend/converters/potrace_converter.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/converters/potrace_converter.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/converters/potrace_converter.py")"
    cp "cleanup_backup/20250930_200223/backend/converters/potrace_converter.py" "/Users/nrw/python/svg-ai/backend/converters/potrace_converter.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/converters/potrace_converter.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/converters/smart_auto_converter.py
if [ -f "cleanup_backup/20250930_200223/backend/converters/smart_auto_converter.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/converters/smart_auto_converter.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/converters/smart_auto_converter.py")"
    cp "cleanup_backup/20250930_200223/backend/converters/smart_auto_converter.py" "/Users/nrw/python/svg-ai/backend/converters/smart_auto_converter.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/converters/smart_auto_converter.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/converters/smart_potrace_converter.py
if [ -f "cleanup_backup/20250930_200223/backend/converters/smart_potrace_converter.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/converters/smart_potrace_converter.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/converters/smart_potrace_converter.py")"
    cp "cleanup_backup/20250930_200223/backend/converters/smart_potrace_converter.py" "/Users/nrw/python/svg-ai/backend/converters/smart_potrace_converter.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/converters/smart_potrace_converter.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/converters/vtracer_converter.py
if [ -f "cleanup_backup/20250930_200223/backend/converters/vtracer_converter.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/converters/vtracer_converter.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/converters/vtracer_converter.py")"
    cp "cleanup_backup/20250930_200223/backend/converters/vtracer_converter.py" "/Users/nrw/python/svg-ai/backend/converters/vtracer_converter.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/converters/vtracer_converter.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/utils/color_detector.py
if [ -f "cleanup_backup/20250930_200223/backend/utils/color_detector.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/utils/color_detector.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/utils/color_detector.py")"
    cp "cleanup_backup/20250930_200223/backend/utils/color_detector.py" "/Users/nrw/python/svg-ai/backend/utils/color_detector.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/utils/color_detector.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/utils/quality_metrics.py
if [ -f "cleanup_backup/20250930_200223/backend/utils/quality_metrics.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/utils/quality_metrics.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/utils/quality_metrics.py")"
    cp "cleanup_backup/20250930_200223/backend/utils/quality_metrics.py" "/Users/nrw/python/svg-ai/backend/utils/quality_metrics.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/utils/quality_metrics.py"
    ((FAILED++))
fi

# Restore: /Users/nrw/python/svg-ai/backend/utils/validation.py
if [ -f "cleanup_backup/20250930_200223/backend/utils/validation.py" ]; then
    echo "  ✓ Restoring /Users/nrw/python/svg-ai/backend/utils/validation.py"
    mkdir -p "$(dirname "/Users/nrw/python/svg-ai/backend/utils/validation.py")"
    cp "cleanup_backup/20250930_200223/backend/utils/validation.py" "/Users/nrw/python/svg-ai/backend/utils/validation.py"
    ((RESTORED++))
else
    echo "  ✗ Backup not found: cleanup_backup/20250930_200223/backend/utils/validation.py"
    ((FAILED++))
fi

echo
echo "📊 Restoration Summary:"
echo "  Restored: $RESTORED files"
echo "  Failed: $FAILED files"
echo

if [ $FAILED -eq 0 ]; then
    echo "✅ All files restored successfully!"
else
    echo "⚠️ Some files could not be restored. Check backup directory."
fi
