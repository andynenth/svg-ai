#!/usr/bin/env python3
"""
Automated Monitoring Reports Generation Script
Generates daily, weekly, and custom reports for system monitoring and analytics
"""

import argparse
import json
import logging
import smtplib
import sys
import time
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from backend.ai_modules.optimization.system_monitoring_analytics import (
    get_global_monitoring_platform,
    start_monitoring_platform
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MonitoringReportGenerator:
    """Automated monitoring report generator"""

    def __init__(self, output_dir: str = "reports", email_config: Optional[Dict] = None):
        """
        Initialize report generator

        Args:
            output_dir: Directory to save reports
            email_config: Email configuration for sending reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.email_config = email_config
        self.monitoring_platform = get_global_monitoring_platform()

        # Ensure monitoring is started
        start_monitoring_platform()

        logger.info("📊 Monitoring Report Generator initialized")

    def generate_daily_report(self, date: Optional[str] = None, send_email: bool = False) -> str:
        """Generate daily monitoring report"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        logger.info(f"🗓️ Generating daily report for {date}")

        try:
            # Generate report using monitoring platform
            report_file = self.monitoring_platform.reporting_system.generate_daily_report(date)

            # Load and enhance report
            with open(report_file, 'r') as f:
                report_data = json.load(f)

            # Add executive summary
            executive_summary = self._generate_executive_summary(report_data, 'daily')
            report_data['executive_summary'] = executive_summary

            # Add alerts and recommendations
            alerts = self._generate_alerts(report_data)
            report_data['alerts'] = alerts

            # Save enhanced report
            enhanced_report_file = self.output_dir / f"daily_report_enhanced_{date}.json"
            with open(enhanced_report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)

            # Generate HTML version
            html_report = self._generate_html_report(report_data, 'daily')
            html_file = self.output_dir / f"daily_report_{date}.html"
            with open(html_file, 'w') as f:
                f.write(html_report)

            # Send email if requested
            if send_email and self.email_config:
                self._send_email_report(
                    subject=f"Daily System Report - {date}",
                    html_content=html_report,
                    attachments=[str(enhanced_report_file)]
                )

            logger.info(f"✅ Daily report generated: {enhanced_report_file}")
            return str(enhanced_report_file)

        except Exception as e:
            logger.error(f"❌ Error generating daily report: {e}")
            raise

    def generate_weekly_report(self, send_email: bool = False) -> str:
        """Generate weekly monitoring report"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        week_str = end_date.strftime('%Y-W%U')

        logger.info(f"📈 Generating weekly report for {week_str}")

        try:
            # Generate report using monitoring platform
            report_file = self.monitoring_platform.reporting_system.generate_weekly_report()

            # Load and enhance report
            with open(report_file, 'r') as f:
                report_data = json.load(f)

            # Add executive summary
            executive_summary = self._generate_executive_summary(report_data, 'weekly')
            report_data['executive_summary'] = executive_summary

            # Add trend analysis
            trend_analysis = self._generate_trend_analysis(report_data)
            report_data['trend_analysis'] = trend_analysis

            # Add strategic recommendations
            strategic_recommendations = self._generate_strategic_recommendations(report_data)
            report_data['strategic_recommendations'] = strategic_recommendations

            # Save enhanced report
            enhanced_report_file = self.output_dir / f"weekly_report_enhanced_{week_str}.json"
            with open(enhanced_report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)

            # Generate HTML version
            html_report = self._generate_html_report(report_data, 'weekly')
            html_file = self.output_dir / f"weekly_report_{week_str}.html"
            with open(html_file, 'w') as f:
                f.write(html_report)

            # Generate PDF version (would require additional libraries)
            # pdf_file = self._generate_pdf_report(html_report, week_str)

            # Send email if requested
            if send_email and self.email_config:
                self._send_email_report(
                    subject=f"Weekly System Report - {week_str}",
                    html_content=html_report,
                    attachments=[str(enhanced_report_file)]
                )

            logger.info(f"✅ Weekly report generated: {enhanced_report_file}")
            return str(enhanced_report_file)

        except Exception as e:
            logger.error(f"❌ Error generating weekly report: {e}")
            raise

    def generate_custom_report(self, hours: int, report_name: str, send_email: bool = False) -> str:
        """Generate custom timeframe report"""
        logger.info(f"📋 Generating custom report for last {hours} hours")

        try:
            # Get data for custom timeframe
            system_data = self.monitoring_platform.db.get_metrics_range('system_metrics', hours)
            quality_data = self.monitoring_platform.db.get_metrics_range('quality_metrics', hours)

            # Generate custom analysis
            custom_analysis = self._analyze_custom_timeframe(system_data, quality_data, hours)

            # Create report structure
            report_data = {
                'report_name': report_name,
                'timeframe_hours': hours,
                'generation_time': datetime.now().isoformat(),
                'custom_analysis': custom_analysis,
                'data_points': {
                    'system_metrics': len(system_data),
                    'quality_metrics': len(quality_data)
                }
            }

            # Add recommendations
            recommendations = self._generate_custom_recommendations(custom_analysis)
            report_data['recommendations'] = recommendations

            # Save report
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = self.output_dir / f"custom_report_{report_name}_{timestamp}.json"
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)

            # Generate HTML version
            html_report = self._generate_html_report(report_data, 'custom')
            html_file = self.output_dir / f"custom_report_{report_name}_{timestamp}.html"
            with open(html_file, 'w') as f:
                f.write(html_report)

            # Send email if requested
            if send_email and self.email_config:
                self._send_email_report(
                    subject=f"Custom System Report - {report_name}",
                    html_content=html_report,
                    attachments=[str(report_file)]
                )

            logger.info(f"✅ Custom report generated: {report_file}")
            return str(report_file)

        except Exception as e:
            logger.error(f"❌ Error generating custom report: {e}")
            raise

    def _generate_executive_summary(self, report_data: Dict, report_type: str) -> Dict[str, str]:
        """Generate executive summary"""
        summary = {
            'period': report_type,
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'key_metrics': {},
            'health_status': 'good',
            'critical_issues': [],
            'achievements': []
        }

        try:
            # Extract key metrics based on report type
            if report_type == 'daily':
                system_perf = report_data.get('system_performance', {})
                summary['key_metrics'] = {
                    'api_response_time': f"{system_perf.get('avg_api_response_time', 0):.3f}s",
                    'success_rate': f"{system_perf.get('success_rate', 0):.1%}",
                    'total_conversions': str(system_perf.get('total_successful_conversions', 0))
                }

                # Determine health status
                if system_perf.get('avg_api_response_time', 0) > 0.2:
                    summary['health_status'] = 'concerning'
                    summary['critical_issues'].append('High API response times detected')

                if system_perf.get('success_rate', 1) < 0.95:
                    summary['health_status'] = 'concerning'
                    summary['critical_issues'].append('Low conversion success rate')

            elif report_type == 'weekly':
                # Weekly summary logic
                trends = report_data.get('weekly_trends', {})
                summary['key_metrics'] = {
                    'total_conversions_week': str(sum(trends.get('daily_performance_trend', {}).get('successful_conversions', {}).values()) if trends.get('daily_performance_trend') else 0),
                    'avg_response_time_week': f"{sum(trends.get('daily_performance_trend', {}).get('api_response_time', {}).values()) / max(len(trends.get('daily_performance_trend', {}).get('api_response_time', {})), 1):.3f}s" if trends.get('daily_performance_trend') else "0.000s"
                }

            # Add achievements
            if summary['health_status'] == 'good':
                summary['achievements'].append('System operating within optimal parameters')

        except Exception as e:
            logger.warning(f"Error generating executive summary: {e}")

        return summary

    def _generate_alerts(self, report_data: Dict) -> List[Dict[str, str]]:
        """Generate alerts from report data"""
        alerts = []

        try:
            system_perf = report_data.get('system_performance', {})

            # High response time alert
            avg_response_time = system_perf.get('avg_api_response_time', 0)
            if avg_response_time > 0.2:
                alerts.append({
                    'level': 'critical' if avg_response_time > 0.5 else 'warning',
                    'message': f'High API response time: {avg_response_time:.3f}s',
                    'recommendation': 'Consider scaling or optimization'
                })

            # Low success rate alert
            success_rate = system_perf.get('success_rate', 1)
            if success_rate < 0.95:
                alerts.append({
                    'level': 'critical' if success_rate < 0.9 else 'warning',
                    'message': f'Low success rate: {success_rate:.1%}',
                    'recommendation': 'Investigate failure causes'
                })

            # Resource utilization alerts
            resource_util = report_data.get('resource_utilization', {})
            max_cpu = resource_util.get('max_cpu_percent', 0)
            if max_cpu > 80:
                alerts.append({
                    'level': 'warning',
                    'message': f'High CPU usage: {max_cpu:.1f}%',
                    'recommendation': 'Monitor for capacity issues'
                })

        except Exception as e:
            logger.warning(f"Error generating alerts: {e}")

        return alerts

    def _generate_trend_analysis(self, report_data: Dict) -> Dict[str, str]:
        """Generate trend analysis for weekly reports"""
        trends = {}

        try:
            weekly_trends = report_data.get('weekly_trends', {})
            daily_performance = weekly_trends.get('daily_performance_trend', {})

            if daily_performance:
                # Analyze response time trend
                response_times = list(daily_performance.get('api_response_time', {}).values())
                if len(response_times) >= 3:
                    if response_times[-1] > response_times[0] * 1.1:
                        trends['response_time'] = 'Increasing - monitor performance'
                    elif response_times[-1] < response_times[0] * 0.9:
                        trends['response_time'] = 'Improving - good optimization'
                    else:
                        trends['response_time'] = 'Stable'

                # Analyze conversion trend
                conversions = list(daily_performance.get('successful_conversions', {}).values())
                if len(conversions) >= 3:
                    if conversions[-1] > conversions[0] * 1.2:
                        trends['conversions'] = 'Growing - increased usage'
                    elif conversions[-1] < conversions[0] * 0.8:
                        trends['conversions'] = 'Declining - investigate causes'
                    else:
                        trends['conversions'] = 'Stable'

        except Exception as e:
            logger.warning(f"Error generating trend analysis: {e}")

        return trends

    def _generate_strategic_recommendations(self, report_data: Dict) -> List[str]:
        """Generate strategic recommendations for weekly reports"""
        recommendations = []

        try:
            # Cost optimization recommendations
            cost_analysis = report_data.get('cost_analysis', {})
            if cost_analysis.get('total_cost', 0) > 0:
                cost_per_conversion = cost_analysis.get('cost_per_conversion', 0)
                if cost_per_conversion > 2.0:  # Arbitrary threshold
                    recommendations.append(f"🏦 High cost per conversion ({cost_per_conversion:.2f}) - review optimization methods")

            # Performance recommendations
            method_performance = report_data.get('method_performance_analysis', {})
            if method_performance and not method_performance.get('no_data'):
                recommendations.append("📊 Review method performance data for optimization opportunities")

            # Resource optimization
            resource_opps = report_data.get('resource_optimization_opportunities', [])
            if resource_opps:
                recommendations.append(f"⚡ {len(resource_opps)} resource optimization opportunities identified")

            # Quality improvements
            quality_stats = report_data.get('quality_improvement_statistics', {})
            if quality_stats and not quality_stats.get('no_data'):
                avg_improvement = quality_stats.get('avg_improvement_by_method', {})
                if avg_improvement:
                    best_method = max(avg_improvement, key=avg_improvement.get)
                    recommendations.append(f"🎯 {best_method} shows best quality improvements - consider prioritizing")

        except Exception as e:
            logger.warning(f"Error generating strategic recommendations: {e}")

        if not recommendations:
            recommendations.append("✅ System operating optimally - continue monitoring")

        return recommendations

    def _analyze_custom_timeframe(self, system_data: List, quality_data: List, hours: int) -> Dict:
        """Analyze data for custom timeframe"""
        analysis = {
            'timeframe_hours': hours,
            'data_summary': {
                'system_data_points': len(system_data),
                'quality_data_points': len(quality_data),
                'data_coverage': f"{len(system_data) / (hours * 6):.1%}"  # Assuming 10-minute intervals
            }
        }

        if system_data:
            import pandas as pd
            df = pd.DataFrame(system_data)

            analysis['performance_summary'] = {
                'avg_api_response_time': df['api_response_time'].mean(),
                'max_api_response_time': df['api_response_time'].max(),
                'avg_cpu_percent': df['cpu_percent'].mean(),
                'max_cpu_percent': df['cpu_percent'].max(),
                'avg_memory_percent': df['memory_percent'].mean(),
                'max_memory_percent': df['memory_percent'].max()
            }

        if quality_data:
            import pandas as pd
            df_quality = pd.DataFrame(quality_data)
            df_quality['improvement'] = df_quality['quality_after'] - df_quality['quality_before']

            analysis['quality_summary'] = {
                'total_conversions': len(df_quality),
                'avg_quality_improvement': df_quality['improvement'].mean(),
                'success_rate': df_quality['success'].mean(),
                'avg_processing_time': df_quality['processing_time'].mean()
            }

        return analysis

    def _generate_custom_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations for custom reports"""
        recommendations = []

        perf_summary = analysis.get('performance_summary', {})
        quality_summary = analysis.get('quality_summary', {})

        if perf_summary:
            avg_response_time = perf_summary.get('avg_api_response_time', 0)
            if avg_response_time > 0.15:
                recommendations.append(f"🔍 API response time ({avg_response_time:.3f}s) above optimal - investigate")

            max_cpu = perf_summary.get('max_cpu_percent', 0)
            if max_cpu > 85:
                recommendations.append(f"⚠️ High CPU peaks ({max_cpu:.1f}%) detected - monitor capacity")

        if quality_summary:
            success_rate = quality_summary.get('success_rate', 1)
            if success_rate < 0.95:
                recommendations.append(f"📉 Success rate ({success_rate:.1%}) below target - review failures")

        if not recommendations:
            recommendations.append("✅ Performance within acceptable range for this timeframe")

        return recommendations

    def _generate_html_report(self, report_data: Dict, report_type: str) -> str:
        """Generate HTML version of report"""
        html_template = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>SVG-AI System Monitoring Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
                .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }
                .alert { padding: 10px; margin: 5px 0; border-radius: 3px; }
                .alert.critical { background-color: #ffebee; border-left: 4px solid #f44336; }
                .alert.warning { background-color: #fff3e0; border-left: 4px solid #ff9800; }
                .recommendation { background-color: #e8f5e8; padding: 10px; margin: 5px 0; border-radius: 3px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>SVG-AI System Monitoring Report</h1>
                <p><strong>Report Type:</strong> {report_type}</p>
                <p><strong>Generated:</strong> {generation_time}</p>
            </div>
        '''.format(
            report_type=report_type.title(),
            generation_time=report_data.get('generation_time', datetime.now().isoformat())
        )

        # Executive Summary
        if 'executive_summary' in report_data:
            summary = report_data['executive_summary']
            html_template += f'''
            <div class="section">
                <h2>Executive Summary</h2>
                <p><strong>Health Status:</strong> {summary.get('health_status', 'Unknown').title()}</p>
                <div class="metrics">
            '''

            for metric, value in summary.get('key_metrics', {}).items():
                html_template += f'<div class="metric"><strong>{metric.replace("_", " ").title()}:</strong> {value}</div>'

            html_template += '</div>'

            if summary.get('critical_issues'):
                html_template += '<h3>Critical Issues</h3><ul>'
                for issue in summary['critical_issues']:
                    html_template += f'<li>{issue}</li>'
                html_template += '</ul>'

            if summary.get('achievements'):
                html_template += '<h3>Achievements</h3><ul>'
                for achievement in summary['achievements']:
                    html_template += f'<li>{achievement}</li>'
                html_template += '</ul>'

            html_template += '</div>'

        # Alerts
        if 'alerts' in report_data and report_data['alerts']:
            html_template += '<div class="section"><h2>Alerts</h2>'
            for alert in report_data['alerts']:
                html_template += f'''
                <div class="alert {alert.get('level', 'info')}">
                    <strong>{alert.get('level', 'Info').title()}:</strong> {alert.get('message', '')}
                    <br><em>Recommendation: {alert.get('recommendation', '')}</em>
                </div>
                '''
            html_template += '</div>'

        # Recommendations
        recommendations = report_data.get('recommendations', [])
        if not recommendations:
            recommendations = report_data.get('strategic_recommendations', [])

        if recommendations:
            html_template += '<div class="section"><h2>Recommendations</h2>'
            for rec in recommendations:
                html_template += f'<div class="recommendation">{rec}</div>'
            html_template += '</div>'

        html_template += '''
        </body>
        </html>
        '''

        return html_template

    def _send_email_report(self, subject: str, html_content: str, attachments: List[str] = None):
        """Send email report"""
        if not self.email_config:
            logger.warning("Email configuration not provided")
            return

        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_config['from_email']
            msg['To'] = ', '.join(self.email_config['to_emails'])

            # Add HTML content
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)

            # Add attachments
            if attachments:
                for attachment_path in attachments:
                    with open(attachment_path, 'rb') as f:
                        attachment = MIMEApplication(f.read())
                        attachment.add_header(
                            'Content-Disposition',
                            'attachment',
                            filename=Path(attachment_path).name
                        )
                        msg.attach(attachment)

            # Send email
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            if self.email_config.get('use_tls'):
                server.starttls()
            if self.email_config.get('username') and self.email_config.get('password'):
                server.login(self.email_config['username'], self.email_config['password'])

            server.send_message(msg)
            server.quit()

            logger.info(f"📧 Email report sent: {subject}")

        except Exception as e:
            logger.error(f"❌ Error sending email: {e}")

    def schedule_daily_reports(self, time_str: str = "09:00", send_email: bool = False):
        """Schedule daily reports at specified time"""
        logger.info(f"📅 Scheduling daily reports at {time_str}")
        # This would integrate with a scheduler like cron or APScheduler
        # For now, just log the scheduling intent
        logger.info("Use cron job: 0 9 * * * /path/to/script --daily --email")

    def schedule_weekly_reports(self, day_of_week: str = "monday", time_str: str = "08:00", send_email: bool = False):
        """Schedule weekly reports"""
        logger.info(f"📅 Scheduling weekly reports on {day_of_week} at {time_str}")
        # This would integrate with a scheduler
        logger.info("Use cron job: 0 8 * * 1 /path/to/script --weekly --email")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Generate monitoring reports')
    parser.add_argument('--daily', action='store_true', help='Generate daily report')
    parser.add_argument('--weekly', action='store_true', help='Generate weekly report')
    parser.add_argument('--custom', type=int, help='Generate custom report for specified hours')
    parser.add_argument('--custom-name', type=str, default='custom', help='Name for custom report')
    parser.add_argument('--date', type=str, help='Date for daily report (YYYY-MM-DD)')
    parser.add_argument('--email', action='store_true', help='Send report via email')
    parser.add_argument('--output-dir', type=str, default='reports', help='Output directory for reports')
    parser.add_argument('--config', type=str, help='Configuration file for email settings')

    args = parser.parse_args()

    # Load email configuration if provided
    email_config = None
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            email_config = json.load(f).get('email', {})

    # Initialize generator
    generator = MonitoringReportGenerator(
        output_dir=args.output_dir,
        email_config=email_config
    )

    try:
        if args.daily:
            report_file = generator.generate_daily_report(
                date=args.date,
                send_email=args.email
            )
            print(f"Daily report generated: {report_file}")

        elif args.weekly:
            report_file = generator.generate_weekly_report(send_email=args.email)
            print(f"Weekly report generated: {report_file}")

        elif args.custom:
            report_file = generator.generate_custom_report(
                hours=args.custom,
                report_name=args.custom_name,
                send_email=args.email
            )
            print(f"Custom report generated: {report_file}")

        else:
            print("Please specify --daily, --weekly, or --custom")
            return 1

    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())