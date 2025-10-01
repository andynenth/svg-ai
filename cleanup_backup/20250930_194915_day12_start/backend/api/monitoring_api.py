#!/usr/bin/env python3
"""
Monitoring API for System Monitoring and Analytics Platform
Provides REST API endpoints for monitoring data collection and dashboard access
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path

# Import monitoring platform
import sys
sys.path.append(str(Path(__file__).parent.parent))
from ai_modules.optimization.system_monitoring_analytics import (
    get_global_monitoring_platform,
    start_monitoring_platform,
    stop_monitoring_platform,
    get_monitoring_dashboard,
    generate_monitoring_reports
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SVG-AI System Monitoring API",
    description="API for system monitoring, analytics, and reporting",
    version="1.0.0"
)


# Pydantic models for API requests
class APIRequestModel(BaseModel):
    response_time: float
    error: bool = False
    endpoint: str
    timestamp: Optional[float] = None


class ConversionModel(BaseModel):
    processing_time: float
    quality_before: float
    quality_after: float
    method: str
    logo_type: str
    success: bool
    cost: float = 0.0
    user_satisfaction: float = 5.0
    user_id: Optional[str] = None


class QueueTaskModel(BaseModel):
    task_id: str
    action: str  # "add" or "remove"


class ReportRequestModel(BaseModel):
    report_type: str  # "daily", "weekly", "custom"
    hours: Optional[int] = 24
    include_visualizations: bool = True


# Global monitoring platform instance
monitoring_platform = None


@app.on_event("startup")
async def startup_event():
    """Initialize monitoring platform on startup"""
    global monitoring_platform
    monitoring_platform = get_global_monitoring_platform()
    start_monitoring_platform()
    logger.info("🚀 Monitoring API started")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown of monitoring platform"""
    stop_monitoring_platform()
    logger.info("🛑 Monitoring API stopped")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "platform_active": monitoring_platform.platform_active if monitoring_platform else False
    }


# Real-time system monitoring endpoints
@app.post("/api/v1/monitoring/api-request")
async def record_api_request(request: APIRequestModel):
    """Record API request for monitoring"""
    try:
        if not monitoring_platform:
            raise HTTPException(status_code=503, detail="Monitoring platform not available")

        monitoring_platform.record_api_request(
            response_time=request.response_time,
            error=request.error
        )

        return {"status": "recorded", "timestamp": time.time()}

    except Exception as e:
        logger.error(f"Error recording API request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/monitoring/conversion")
async def record_conversion(conversion: ConversionModel):
    """Record conversion for monitoring"""
    try:
        if not monitoring_platform:
            raise HTTPException(status_code=503, detail="Monitoring platform not available")

        monitoring_platform.record_conversion(
            processing_time=conversion.processing_time,
            quality_before=conversion.quality_before,
            quality_after=conversion.quality_after,
            method=conversion.method,
            logo_type=conversion.logo_type,
            success=conversion.success,
            cost=conversion.cost,
            user_satisfaction=conversion.user_satisfaction
        )

        return {"status": "recorded", "timestamp": time.time()}

    except Exception as e:
        logger.error(f"Error recording conversion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/monitoring/queue")
async def manage_queue(task: QueueTaskModel):
    """Manage processing queue"""
    try:
        if not monitoring_platform:
            raise HTTPException(status_code=503, detail="Monitoring platform not available")

        if task.action == "add":
            monitoring_platform.add_to_processing_queue(task.task_id)
        elif task.action == "remove":
            monitoring_platform.remove_from_processing_queue(task.task_id)
        else:
            raise HTTPException(status_code=400, detail="Invalid action. Use 'add' or 'remove'")

        return {"status": "updated", "task_id": task.task_id, "action": task.action}

    except Exception as e:
        logger.error(f"Error managing queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Quality and performance analytics endpoints
@app.get("/api/v1/analytics/dashboard")
async def get_dashboard():
    """Get comprehensive monitoring dashboard"""
    try:
        if not monitoring_platform:
            raise HTTPException(status_code=503, detail="Monitoring platform not available")

        dashboard = monitoring_platform.get_comprehensive_dashboard()
        return dashboard

    except Exception as e:
        logger.error(f"Error getting dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/analytics/quality-trends")
async def get_quality_trends(hours: int = 24):
    """Get quality improvement trends"""
    try:
        if not monitoring_platform:
            raise HTTPException(status_code=503, detail="Monitoring platform not available")

        trends = monitoring_platform.quality_analytics.analyze_quality_trends(hours)
        return trends

    except Exception as e:
        logger.error(f"Error getting quality trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/analytics/method-effectiveness")
async def get_method_effectiveness(hours: int = 168):
    """Get method effectiveness analysis"""
    try:
        if not monitoring_platform:
            raise HTTPException(status_code=503, detail="Monitoring platform not available")

        effectiveness = monitoring_platform.quality_analytics.analyze_method_effectiveness(hours)
        return effectiveness

    except Exception as e:
        logger.error(f"Error getting method effectiveness: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/analytics/user-satisfaction")
async def get_user_satisfaction():
    """Get user satisfaction analysis"""
    try:
        if not monitoring_platform:
            raise HTTPException(status_code=503, detail="Monitoring platform not available")

        satisfaction = monitoring_platform.quality_analytics.analyze_user_satisfaction()
        return satisfaction

    except Exception as e:
        logger.error(f"Error getting user satisfaction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/analytics/performance-regression")
async def get_performance_regression():
    """Get performance regression analysis"""
    try:
        if not monitoring_platform:
            raise HTTPException(status_code=503, detail="Monitoring platform not available")

        regression = monitoring_platform.quality_analytics.detect_performance_regression()
        return regression

    except Exception as e:
        logger.error(f"Error getting performance regression: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Comprehensive reporting endpoints
@app.post("/api/v1/reports/generate")
async def generate_report(background_tasks: BackgroundTasks, request: ReportRequestModel):
    """Generate monitoring report"""
    try:
        if not monitoring_platform:
            raise HTTPException(status_code=503, detail="Monitoring platform not available")

        if request.report_type == "daily":
            background_tasks.add_task(generate_daily_report_task)
        elif request.report_type == "weekly":
            background_tasks.add_task(generate_weekly_report_task)
        elif request.report_type == "all":
            background_tasks.add_task(generate_all_reports_task)
        else:
            raise HTTPException(status_code=400, detail="Invalid report type")

        return {
            "status": "generating",
            "report_type": request.report_type,
            "estimated_completion": "2-5 minutes"
        }

    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/reports/list")
async def list_reports():
    """List available reports"""
    try:
        reports_dir = Path("reports")
        if not reports_dir.exists():
            return {"reports": []}

        reports = []
        for report_file in reports_dir.glob("*.json"):
            reports.append({
                "filename": report_file.name,
                "path": str(report_file),
                "size": report_file.stat().st_size,
                "created": report_file.stat().st_mtime,
                "type": "daily" if "daily" in report_file.name else "weekly" if "weekly" in report_file.name else "other"
            })

        return {"reports": sorted(reports, key=lambda x: x["created"], reverse=True)}

    except Exception as e:
        logger.error(f"Error listing reports: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/reports/download/{filename}")
async def download_report(filename: str):
    """Download report file"""
    try:
        reports_dir = Path("reports")
        report_file = reports_dir / filename

        if not report_file.exists():
            raise HTTPException(status_code=404, detail="Report not found")

        return FileResponse(
            path=report_file,
            filename=filename,
            media_type="application/json"
        )

    except Exception as e:
        logger.error(f"Error downloading report: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Predictive analytics and optimization endpoints
@app.get("/api/v1/predictive/capacity-planning")
async def get_capacity_planning(forecast_days: int = 30):
    """Get capacity planning analysis"""
    try:
        if not monitoring_platform:
            raise HTTPException(status_code=503, detail="Monitoring platform not available")

        analysis = monitoring_platform.predictive_optimizer.capacity_planning_analysis(forecast_days)
        return analysis

    except Exception as e:
        logger.error(f"Error getting capacity planning: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/predictive/maintenance-schedule")
async def get_maintenance_schedule():
    """Get predictive maintenance schedule"""
    try:
        if not monitoring_platform:
            raise HTTPException(status_code=503, detail="Monitoring platform not available")

        schedule = monitoring_platform.predictive_optimizer.predictive_maintenance_scheduling()
        return schedule

    except Exception as e:
        logger.error(f"Error getting maintenance schedule: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/predictive/optimization-recommendations")
async def get_optimization_recommendations():
    """Get performance optimization recommendations"""
    try:
        if not monitoring_platform:
            raise HTTPException(status_code=503, detail="Monitoring platform not available")

        recommendations = monitoring_platform.predictive_optimizer.generate_performance_optimization_recommendations()
        return recommendations

    except Exception as e:
        logger.error(f"Error getting optimization recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/predictive/cost-optimization")
async def get_cost_optimization():
    """Get cost optimization analysis"""
    try:
        if not monitoring_platform:
            raise HTTPException(status_code=503, detail="Monitoring platform not available")

        optimization = monitoring_platform.predictive_optimizer.cost_optimization_analysis()
        return optimization

    except Exception as e:
        logger.error(f"Error getting cost optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Background tasks for report generation
async def generate_daily_report_task():
    """Background task to generate daily report"""
    try:
        if monitoring_platform:
            reports = monitoring_platform.reporting_system.generate_daily_report()
            logger.info(f"Daily report generated: {reports}")
    except Exception as e:
        logger.error(f"Error in daily report task: {e}")


async def generate_weekly_report_task():
    """Background task to generate weekly report"""
    try:
        if monitoring_platform:
            reports = monitoring_platform.reporting_system.generate_weekly_report()
            logger.info(f"Weekly report generated: {reports}")
    except Exception as e:
        logger.error(f"Error in weekly report task: {e}")


async def generate_all_reports_task():
    """Background task to generate all reports"""
    try:
        if monitoring_platform:
            reports = monitoring_platform.generate_all_reports()
            logger.info(f"All reports generated: {reports}")
    except Exception as e:
        logger.error(f"Error in all reports task: {e}")


# WebSocket endpoint for real-time updates
@app.websocket("/ws/monitoring")
async def websocket_monitoring(websocket):
    """WebSocket endpoint for real-time monitoring updates"""
    await websocket.accept()

    try:
        while True:
            if monitoring_platform and monitoring_platform.platform_active:
                # Get current metrics
                current_metrics = monitoring_platform._get_current_metrics()

                # Send to client
                await websocket.send_json({
                    "type": "metrics_update",
                    "data": current_metrics,
                    "timestamp": time.time()
                })

            await asyncio.sleep(5)  # Update every 5 seconds

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()


# Admin endpoints
@app.post("/api/v1/admin/start-monitoring")
async def start_monitoring():
    """Start monitoring platform"""
    try:
        start_monitoring_platform()
        return {"status": "started", "timestamp": time.time()}
    except Exception as e:
        logger.error(f"Error starting monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/admin/stop-monitoring")
async def stop_monitoring():
    """Stop monitoring platform"""
    try:
        stop_monitoring_platform()
        return {"status": "stopped", "timestamp": time.time()}
    except Exception as e:
        logger.error(f"Error stopping monitoring: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/admin/status")
async def get_monitoring_status():
    """Get monitoring platform status"""
    try:
        return {
            "platform_active": monitoring_platform.platform_active if monitoring_platform else False,
            "components_status": {
                "real_time_monitor": monitoring_platform.real_time_monitor.active if monitoring_platform else False,
                "database_connected": True,  # Would check actual DB connection
                "reports_directory": str(Path("reports").absolute()),
                "data_directory": str(monitoring_platform.data_dir) if monitoring_platform else None
            },
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "monitoring_api:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )