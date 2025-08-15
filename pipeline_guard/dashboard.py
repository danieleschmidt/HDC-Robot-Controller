#!/usr/bin/env python3
"""
Real-time Health Monitoring Dashboard for Pipeline Guard
Interactive web-based dashboard for visualizing pipeline health and metrics

Features:
- Real-time pipeline health monitoring
- Interactive charts and visualizations
- Alert management and history
- Repair action tracking
- Custom metric dashboards
- Export and reporting capabilities
"""

import json
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import asdict
from pathlib import Path

# Web framework imports
try:
    from flask import Flask, render_template_string, jsonify, request, send_from_directory
    from flask_socketio import SocketIO, emit
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None
    SocketIO = None

# Import pipeline guard types
from .core import PipelineGuard, GuardStatus, GuardAlert, AlertSeverity
from .monitoring import PipelineMonitor

class HealthDashboard:
    """
    Real-time web dashboard for pipeline health monitoring
    
    Provides interactive visualization of pipeline metrics, alerts,
    and repair activities with real-time updates via WebSocket.
    """
    
    def __init__(self, pipeline_guard: PipelineGuard, 
                 pipeline_monitor: PipelineMonitor,
                 host: str = "0.0.0.0", 
                 port: int = 8080):
        
        if not FLASK_AVAILABLE:
            raise ImportError("Flask and Flask-SocketIO are required for dashboard functionality")
        
        self.pipeline_guard = pipeline_guard
        self.pipeline_monitor = pipeline_monitor
        self.host = host
        self.port = port
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'pipeline-guard-dashboard-secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Dashboard state
        self.connected_clients = set()
        self.update_interval = 5  # seconds
        self.update_task = None
        
        self.logger = logging.getLogger('dashboard')
        
        # Register routes
        self._register_routes()
        self._register_websocket_handlers()
        
        self.logger.info(f"Dashboard initialized on {host}:{port}")
    
    def _register_routes(self):
        """Register Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard page"""
            return render_template_string(self._get_dashboard_template())
        
        @self.app.route('/api/status')
        def api_status():
            """Get overall system status"""
            try:
                guard_summary = self.pipeline_guard.get_guard_summary()
                monitor_stats = self.pipeline_monitor.get_monitoring_statistics()
                
                return jsonify({
                    "timestamp": datetime.now().isoformat(),
                    "guard_status": guard_summary,
                    "monitoring_stats": monitor_stats
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/pipelines')
        def api_pipelines():
            """Get pipeline summaries"""
            try:
                pipelines = {}
                
                for pipeline_id in self.pipeline_guard.active_pipelines:
                    pipeline_health = self.pipeline_guard.get_pipeline_health(pipeline_id)
                    monitor_summary = self.pipeline_monitor.get_pipeline_summary(pipeline_id)
                    
                    pipelines[pipeline_id] = {
                        "health": pipeline_health,
                        "monitoring": monitor_summary
                    }
                
                return jsonify(pipelines)
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/alerts')
        def api_alerts():
            """Get recent alerts"""
            try:
                hours = int(request.args.get('hours', 24))
                cutoff_time = datetime.now() - timedelta(hours=hours)
                
                recent_alerts = [
                    {
                        "id": alert.id,
                        "timestamp": alert.timestamp.isoformat(),
                        "severity": alert.severity.value,
                        "pipeline_id": alert.pipeline_id,
                        "title": alert.title,
                        "message": alert.message,
                        "resolved": alert.resolved,
                        "auto_repair_attempted": alert.auto_repair_attempted
                    }
                    for alert in self.pipeline_guard.alert_history
                    if alert.timestamp > cutoff_time
                ]
                
                return jsonify(recent_alerts)
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/metrics/<pipeline_id>')
        def api_metrics(pipeline_id):
            """Get metrics for specific pipeline"""
            try:
                hours = int(request.args.get('hours', 24))
                
                current_metrics = self.pipeline_monitor.get_current_metrics(pipeline_id)
                
                # Get history for key metrics
                history_data = {}
                key_metrics = ["build_duration_seconds", "system_cpu_percent", "system_memory_percent"]
                
                for metric_name in key_metrics:
                    if metric_name in current_metrics:
                        history = self.pipeline_monitor.get_metric_history(pipeline_id, metric_name, hours)
                        history_data[metric_name] = [
                            {
                                "timestamp": point.timestamp.isoformat(),
                                "value": point.value
                            }
                            for point in history
                        ]
                
                return jsonify({
                    "current_metrics": current_metrics,
                    "history": history_data
                })
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/api/export/<pipeline_id>')
        def api_export(pipeline_id):
            """Export pipeline data"""
            try:
                format_type = request.args.get('format', 'json')
                
                if format_type == 'json':
                    data = self.pipeline_monitor.export_metrics(pipeline_id, format_type)
                    return data, 200, {'Content-Type': 'application/json'}
                else:
                    return jsonify({"error": "Unsupported format"}), 400
            except Exception as e:
                return jsonify({"error": str(e)}), 500
    
    def _register_websocket_handlers(self):
        """Register WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            self.connected_clients.add(request.sid)
            self.logger.info(f"Client connected: {request.sid}")
            
            # Send initial data to new client
            self._send_initial_data(request.sid)
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            self.connected_clients.discard(request.sid)
            self.logger.info(f"Client disconnected: {request.sid}")
        
        @self.socketio.on('subscribe_pipeline')
        def handle_subscribe_pipeline(data):
            """Handle pipeline subscription"""
            pipeline_id = data.get('pipeline_id')
            if pipeline_id:
                # Send current pipeline data
                pipeline_data = self._get_pipeline_data(pipeline_id)
                emit('pipeline_update', {
                    'pipeline_id': pipeline_id,
                    'data': pipeline_data
                })
        
        @self.socketio.on('acknowledge_alert')
        def handle_acknowledge_alert(data):
            """Handle alert acknowledgment"""
            alert_id = data.get('alert_id')
            # In real implementation, this would update alert status
            self.logger.info(f"Alert acknowledged: {alert_id}")
    
    def _send_initial_data(self, client_id: str):
        """Send initial dashboard data to new client"""
        
        try:
            # System status
            guard_summary = self.pipeline_guard.get_guard_summary()
            monitor_stats = self.pipeline_monitor.get_monitoring_statistics()
            
            self.socketio.emit('system_status', {
                'guard_status': guard_summary,
                'monitoring_stats': monitor_stats
            }, room=client_id)
            
            # Pipeline data
            for pipeline_id in self.pipeline_guard.active_pipelines:
                pipeline_data = self._get_pipeline_data(pipeline_id)
                self.socketio.emit('pipeline_update', {
                    'pipeline_id': pipeline_id,
                    'data': pipeline_data
                }, room=client_id)
            
        except Exception as e:
            self.logger.error(f"Failed to send initial data to {client_id}: {e}")
    
    def _get_pipeline_data(self, pipeline_id: str) -> Dict[str, Any]:
        """Get comprehensive pipeline data"""
        
        try:
            pipeline_health = self.pipeline_guard.get_pipeline_health(pipeline_id)
            monitor_summary = self.pipeline_monitor.get_pipeline_summary(pipeline_id)
            current_metrics = self.pipeline_monitor.get_current_metrics(pipeline_id)
            
            return {
                "health": pipeline_health,
                "monitoring": monitor_summary,
                "metrics": current_metrics,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to get pipeline data for {pipeline_id}: {e}")
            return {"error": str(e)}
    
    def start_dashboard(self, debug: bool = False):
        """Start the dashboard server"""
        
        self.logger.info(f"Starting dashboard server on {self.host}:{self.port}")
        
        # Start real-time update task
        self._start_update_task()
        
        # Run Flask app with SocketIO
        self.socketio.run(
            self.app,
            host=self.host,
            port=self.port,
            debug=debug,
            allow_unsafe_werkzeug=True
        )
    
    def _start_update_task(self):
        """Start background task for real-time updates"""
        
        def update_loop():
            while True:
                try:
                    if self.connected_clients:
                        self._broadcast_updates()
                    time.sleep(self.update_interval)
                except Exception as e:
                    self.logger.error(f"Error in update loop: {e}")
                    time.sleep(self.update_interval)
        
        import threading
        self.update_task = threading.Thread(target=update_loop, daemon=True)
        self.update_task.start()
    
    def _broadcast_updates(self):
        """Broadcast real-time updates to all connected clients"""
        
        try:
            # System status update
            guard_summary = self.pipeline_guard.get_guard_summary()
            monitor_stats = self.pipeline_monitor.get_monitoring_statistics()
            
            self.socketio.emit('system_status', {
                'guard_status': guard_summary,
                'monitoring_stats': monitor_stats,
                'timestamp': datetime.now().isoformat()
            })
            
            # Pipeline updates
            for pipeline_id in self.pipeline_guard.active_pipelines:
                pipeline_data = self._get_pipeline_data(pipeline_id)
                self.socketio.emit('pipeline_update', {
                    'pipeline_id': pipeline_id,
                    'data': pipeline_data
                })
            
            # Alert updates
            recent_alerts = [
                {
                    "id": alert.id,
                    "timestamp": alert.timestamp.isoformat(),
                    "severity": alert.severity.value,
                    "pipeline_id": alert.pipeline_id,
                    "title": alert.title,
                    "message": alert.message,
                    "resolved": alert.resolved
                }
                for alert in self.pipeline_guard.alert_history[-10:]  # Last 10 alerts
            ]
            
            self.socketio.emit('alerts_update', recent_alerts)
            
        except Exception as e:
            self.logger.error(f"Failed to broadcast updates: {e}")
    
    def _get_dashboard_template(self) -> str:
        """Get HTML template for dashboard"""
        
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Guard Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0f1419;
            color: #e6e8ea;
            line-height: 1.6;
        }
        
        .header {
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            padding: 1rem 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        
        .header h1 {
            color: white;
            font-size: 1.8rem;
            font-weight: 600;
        }
        
        .header .subtitle {
            color: #bfdbfe;
            font-size: 0.9rem;
            margin-top: 0.25rem;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .status-bar {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }
        
        .status-card {
            background: #1a1f2e;
            border: 1px solid #2d3748;
            border-radius: 8px;
            padding: 1.5rem;
            transition: transform 0.2s ease;
        }
        
        .status-card:hover {
            transform: translateY(-2px);
        }
        
        .status-card h3 {
            color: #60a5fa;
            font-size: 0.9rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.5rem;
        }
        
        .status-value {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        
        .status-healthy { color: #10b981; }
        .status-warning { color: #f59e0b; }
        .status-critical { color: #ef4444; }
        
        .status-description {
            color: #9ca3af;
            font-size: 0.8rem;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 2rem;
            margin-bottom: 2rem;
        }
        
        .pipelines-section, .alerts-section {
            background: #1a1f2e;
            border: 1px solid #2d3748;
            border-radius: 8px;
            padding: 1.5rem;
        }
        
        .section-title {
            color: #60a5fa;
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .pipeline-item {
            background: #111827;
            border: 1px solid #374151;
            border-radius: 6px;
            padding: 1rem;
            margin-bottom: 1rem;
            transition: border-color 0.2s ease;
        }
        
        .pipeline-item:hover {
            border-color: #60a5fa;
        }
        
        .pipeline-header {
            display: flex;
            justify-content: between;
            align-items: center;
            margin-bottom: 0.5rem;
        }
        
        .pipeline-name {
            font-weight: 600;
            color: #f3f4f6;
        }
        
        .pipeline-status {
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 500;
            text-transform: uppercase;
        }
        
        .status-healthy-bg { background: #10b981; color: white; }
        .status-warning-bg { background: #f59e0b; color: white; }
        .status-critical-bg { background: #ef4444; color: white; }
        
        .pipeline-metrics {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin-top: 1rem;
        }
        
        .metric {
            text-align: center;
        }
        
        .metric-value {
            font-size: 1.2rem;
            font-weight: 600;
            color: #60a5fa;
        }
        
        .metric-label {
            font-size: 0.75rem;
            color: #9ca3af;
            text-transform: uppercase;
        }
        
        .alert-item {
            background: #111827;
            border-left: 4px solid;
            padding: 1rem;
            margin-bottom: 0.75rem;
            border-radius: 0 4px 4px 0;
        }
        
        .alert-critical { border-left-color: #ef4444; }
        .alert-warning { border-left-color: #f59e0b; }
        .alert-info { border-left-color: #3b82f6; }
        
        .alert-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 0.5rem;
        }
        
        .alert-title {
            font-weight: 600;
            color: #f3f4f6;
            flex: 1;
        }
        
        .alert-time {
            font-size: 0.75rem;
            color: #9ca3af;
        }
        
        .alert-message {
            color: #d1d5db;
            font-size: 0.9rem;
            line-height: 1.4;
        }
        
        .chart-container {
            background: #1a1f2e;
            border: 1px solid #2d3748;
            border-radius: 8px;
            padding: 1.5rem;
            margin-top: 2rem;
        }
        
        .connection-status {
            position: fixed;
            top: 1rem;
            right: 1rem;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
            z-index: 1000;
        }
        
        .connected { background: #10b981; color: white; }
        .disconnected { background: #ef4444; color: white; }
        
        .loading {
            text-align: center;
            padding: 2rem;
            color: #9ca3af;
        }
        
        .spinner {
            border: 2px solid #374151;
            border-top: 2px solid #60a5fa;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            
            .status-bar {
                grid-template-columns: 1fr;
            }
            
            .pipeline-metrics {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="connection-status" id="connectionStatus">
        <span id="connectionText">Connecting...</span>
    </div>
    
    <header class="header">
        <h1>🛡️ Pipeline Guard Dashboard</h1>
        <div class="subtitle">Real-time CI/CD Pipeline Health Monitoring</div>
    </header>
    
    <div class="container">
        <div class="status-bar">
            <div class="status-card">
                <h3>Guard Status</h3>
                <div class="status-value status-healthy" id="guardStatus">HEALTHY</div>
                <div class="status-description" id="guardDescription">All systems operational</div>
            </div>
            <div class="status-card">
                <h3>Active Pipelines</h3>
                <div class="status-value" id="activePipelines">0</div>
                <div class="status-description">Monitored pipelines</div>
            </div>
            <div class="status-card">
                <h3>Active Alerts</h3>
                <div class="status-value status-warning" id="activeAlerts">0</div>
                <div class="status-description">Requiring attention</div>
            </div>
            <div class="status-card">
                <h3>Auto Repairs</h3>
                <div class="status-value status-healthy" id="autoRepairs">0</div>
                <div class="status-description">Successful today</div>
            </div>
        </div>
        
        <div class="main-content">
            <div class="pipelines-section">
                <h2 class="section-title">
                    <span>📊</span>
                    Pipeline Status
                </h2>
                <div id="pipelinesList">
                    <div class="loading">
                        <div class="spinner"></div>
                        Loading pipeline data...
                    </div>
                </div>
            </div>
            
            <div class="alerts-section">
                <h2 class="section-title">
                    <span>⚠️</span>
                    Recent Alerts
                </h2>
                <div id="alertsList">
                    <div class="loading">
                        <div class="spinner"></div>
                        Loading alerts...
                    </div>
                </div>
            </div>
        </div>
        
        <div class="chart-container">
            <h2 class="section-title">
                <span>📈</span>
                System Performance
            </h2>
            <canvas id="performanceChart" width="400" height="150"></canvas>
        </div>
    </div>
    
    <script>
        // Initialize WebSocket connection
        const socket = io();
        
        // Connection status
        const connectionStatus = document.getElementById('connectionStatus');
        const connectionText = document.getElementById('connectionText');
        
        socket.on('connect', function() {
            connectionStatus.className = 'connection-status connected';
            connectionText.textContent = 'Connected';
        });
        
        socket.on('disconnect', function() {
            connectionStatus.className = 'connection-status disconnected';
            connectionText.textContent = 'Disconnected';
        });
        
        // Chart initialization
        const ctx = document.getElementById('performanceChart').getContext('2d');
        const performanceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'System Health',
                    data: [],
                    borderColor: '#60a5fa',
                    backgroundColor: 'rgba(96, 165, 250, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        labels: {
                            color: '#e6e8ea'
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: { color: '#9ca3af' },
                        grid: { color: '#374151' }
                    },
                    y: {
                        ticks: { color: '#9ca3af' },
                        grid: { color: '#374151' },
                        min: 0,
                        max: 1
                    }
                }
            }
        });
        
        // Data handlers
        socket.on('system_status', function(data) {
            updateSystemStatus(data);
        });
        
        socket.on('pipeline_update', function(data) {
            updatePipelineData(data);
        });
        
        socket.on('alerts_update', function(alerts) {
            updateAlerts(alerts);
        });
        
        function updateSystemStatus(data) {
            const guardStatus = data.guard_status;
            const monitorStats = data.monitoring_stats;
            
            // Update status cards
            document.getElementById('guardStatus').textContent = guardStatus.guard_status.toUpperCase();
            document.getElementById('activePipelines').textContent = guardStatus.monitored_pipelines;
            document.getElementById('activeAlerts').textContent = guardStatus.active_alerts;
            document.getElementById('autoRepairs').textContent = guardStatus.successful_repairs || 0;
            
            // Update chart
            const now = new Date().toLocaleTimeString();
            const healthScore = calculateOverallHealth(guardStatus);
            
            performanceChart.data.labels.push(now);
            performanceChart.data.datasets[0].data.push(healthScore);
            
            // Keep only last 20 data points
            if (performanceChart.data.labels.length > 20) {
                performanceChart.data.labels.shift();
                performanceChart.data.datasets[0].data.shift();
            }
            
            performanceChart.update('none');
        }
        
        function updatePipelineData(data) {
            const pipelinesList = document.getElementById('pipelinesList');
            
            // Find existing pipeline element or create new one
            let pipelineElement = document.getElementById('pipeline-' + data.pipeline_id);
            if (!pipelineElement) {
                pipelineElement = createPipelineElement(data.pipeline_id);
                pipelinesList.appendChild(pipelineElement);
                
                // Remove loading message if this is first pipeline
                const loading = pipelinesList.querySelector('.loading');
                if (loading) loading.remove();
            }
            
            updatePipelineElement(pipelineElement, data);
        }
        
        function createPipelineElement(pipelineId) {
            const element = document.createElement('div');
            element.className = 'pipeline-item';
            element.id = 'pipeline-' + pipelineId;
            
            element.innerHTML = `
                <div class="pipeline-header">
                    <div class="pipeline-name">${pipelineId}</div>
                    <div class="pipeline-status">Unknown</div>
                </div>
                <div class="pipeline-metrics">
                    <div class="metric">
                        <div class="metric-value">--</div>
                        <div class="metric-label">Health</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">--</div>
                        <div class="metric-label">Duration</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">--</div>
                        <div class="metric-label">Success Rate</div>
                    </div>
                </div>
            `;
            
            return element;
        }
        
        function updatePipelineElement(element, data) {
            const health = data.data.health;
            const monitoring = data.data.monitoring;
            
            if (!health) return;
            
            // Update status
            const statusElement = element.querySelector('.pipeline-status');
            statusElement.textContent = health.status.toUpperCase();
            statusElement.className = 'pipeline-status status-' + health.status + '-bg';
            
            // Update metrics
            const metrics = element.querySelectorAll('.metric-value');
            if (metrics.length >= 3) {
                metrics[0].textContent = (health.health_score || 0).toFixed(2);
                metrics[1].textContent = formatDuration(health.duration || 0);
                metrics[2].textContent = formatPercentage(health.success_rate || 0);
            }
        }
        
        function updateAlerts(alerts) {
            const alertsList = document.getElementById('alertsList');
            alertsList.innerHTML = '';
            
            if (alerts.length === 0) {
                alertsList.innerHTML = '<div class="loading">No recent alerts</div>';
                return;
            }
            
            alerts.slice(0, 10).forEach(alert => {
                const alertElement = document.createElement('div');
                alertElement.className = 'alert-item alert-' + alert.severity;
                
                alertElement.innerHTML = `
                    <div class="alert-header">
                        <div class="alert-title">${alert.title}</div>
                        <div class="alert-time">${formatTime(alert.timestamp)}</div>
                    </div>
                    <div class="alert-message">${alert.message}</div>
                `;
                
                alertsList.appendChild(alertElement);
            });
        }
        
        // Utility functions
        function calculateOverallHealth(guardStatus) {
            const pipelineHealth = Object.values(guardStatus.pipeline_health || {});
            if (pipelineHealth.length === 0) return 0.5;
            
            const avgHealth = pipelineHealth.reduce((sum, p) => sum + (p.health_score || 0), 0) / pipelineHealth.length;
            return Math.max(0, Math.min(1, avgHealth));
        }
        
        function formatDuration(seconds) {
            if (seconds < 60) return seconds.toFixed(0) + 's';
            if (seconds < 3600) return (seconds / 60).toFixed(1) + 'm';
            return (seconds / 3600).toFixed(1) + 'h';
        }
        
        function formatPercentage(value) {
            return (value * 100).toFixed(1) + '%';
        }
        
        function formatTime(timestamp) {
            return new Date(timestamp).toLocaleTimeString();
        }
    </script>
</body>
</html>
        '''

# Example usage
if __name__ == "__main__":
    # This would normally be initialized with actual PipelineGuard and PipelineMonitor instances
    print("Dashboard module loaded. Use HealthDashboard class to create dashboard instance.")
    print("Example:")
    print("  dashboard = HealthDashboard(pipeline_guard, pipeline_monitor)")
    print("  dashboard.start_dashboard()")