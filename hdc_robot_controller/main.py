#!/usr/bin/env python3
"""
Main entry point for HDC Robot Controller production deployment.

Provides HTTP API endpoints, health checks, and system management
for enterprise robotics applications.
"""

import os
import sys
import time
import signal
import threading
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Core HDC imports
from hdc_robot_controller.core.hypervector import HyperVector
from hdc_robot_controller.core.memory import HierarchicalMemory
from hdc_robot_controller.core.sensor_encoder import SensorEncoder
from hdc_robot_controller.core.behavior_learner import BehaviorLearner
from hdc_robot_controller.core.logging_system import setup_production_logging
from hdc_robot_controller.robustness.advanced_error_recovery import AdvancedErrorRecovery, FaultSeverity
from hdc_robot_controller.optimization.gpu_accelerator import get_gpu_accelerator
from hdc_robot_controller.scaling.distributed_coordinator import DistributedCoordinator

# HTTP server imports
try:
    from flask import Flask, jsonify, request, render_template_string
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    Flask = None

# Configuration
HDC_CONFIG = {
    'dimension': int(os.getenv('HDC_DIMENSION', '10000')),
    'log_level': os.getenv('HDC_LOG_LEVEL', 'INFO'),
    'enable_gpu': os.getenv('HDC_ENABLE_GPU', 'true').lower() == 'true',
    'enable_monitoring': os.getenv('HDC_ENABLE_MONITORING', 'true').lower() == 'true',
    'coordinator_port': int(os.getenv('HDC_COORDINATOR_PORT', '8888')),
    'health_check_port': int(os.getenv('HDC_HEALTH_CHECK_PORT', '8080')),
    'log_file': os.getenv('HDC_LOG_FILE', '/app/logs/hdc_robot_controller.log')
}


class HDCRobotControllerService:
    """Main HDC Robot Controller service."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the HDC service."""
        self.config = config
        self.running = False
        
        # Setup logging
        self.logger = setup_production_logging(
            config['log_file'],
            config['log_level'],
            config['enable_monitoring']
        )
        
        # Initialize core components
        self.logger.info("Initializing HDC Robot Controller service", config=config)
        
        self.memory = HierarchicalMemory(config['dimension'])
        self.sensor_encoder = SensorEncoder(config['dimension'])
        self.behavior_learner = BehaviorLearner(config['dimension'])
        self.error_recovery = AdvancedErrorRecovery(config['dimension'])
        
        # Initialize GPU acceleration
        if config['enable_gpu']:
            self.gpu_accelerator = get_gpu_accelerator()
            self.logger.info("GPU acceleration initialized", 
                           gpu_available=self.gpu_accelerator.is_gpu_available())
        else:
            self.gpu_accelerator = None
        
        # Initialize distributed coordinator
        self.coordinator = DistributedCoordinator(
            coordinator_port=config['coordinator_port'],
            max_workers=4
        )
        
        # Flask web interface
        if FLASK_AVAILABLE:
            self.app = self._create_flask_app()
        else:
            self.app = None
            self.logger.warning("Flask not available, web interface disabled")
        
        # Service statistics
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        
        self.logger.info("HDC Robot Controller service initialized successfully")
    
    def start(self):
        """Start the HDC service."""
        if self.running:
            return
        
        self.running = True
        self.logger.info("Starting HDC Robot Controller service")
        
        # Start core services
        self.error_recovery.start_monitoring()
        self.coordinator.start_coordinator()
        
        # Start web server if available
        if self.app:
            self.logger.info("Starting web interface on port", port=self.config['health_check_port'])
            
            # Run Flask in a separate thread
            flask_thread = threading.Thread(
                target=self._run_flask_app,
                daemon=True
            )
            flask_thread.start()
        
        self.logger.info("HDC Robot Controller service started successfully")
    
    def stop(self):
        """Stop the HDC service."""
        if not self.running:
            return
        
        self.logger.info("Stopping HDC Robot Controller service")
        
        self.running = False
        self.error_recovery.stop_monitoring()
        self.coordinator.stop_coordinator()
        
        self.logger.info("HDC Robot Controller service stopped")
    
    def _create_flask_app(self) -> Flask:
        """Create Flask application with API endpoints."""
        app = Flask(__name__)
        
        @app.route('/health')
        def health_check():
            """Health check endpoint."""
            try:
                self.request_count += 1
                
                # Check system health
                health_report = self.error_recovery.get_system_health_report()
                cluster_status = self.coordinator.get_cluster_status()
                
                # Basic system checks
                uptime = time.time() - self.start_time
                
                health_status = {
                    'status': 'healthy',
                    'timestamp': time.time(),
                    'uptime_seconds': uptime,
                    'service_version': '5.0.0',
                    'system_health': health_report['overall_health'],
                    'active_faults': health_report['active_faults_count'],
                    'cluster_nodes': cluster_status['total_nodes'],
                    'request_count': self.request_count,
                    'error_count': self.error_count
                }
                
                # Determine overall health
                if health_report['overall_health'] < 0.7 or health_report['active_faults_count'] > 0:
                    health_status['status'] = 'degraded'
                
                return jsonify(health_status)
                
            except Exception as e:
                self.error_count += 1
                self.logger.error("Health check error", error=str(e))
                
                return jsonify({
                    'status': 'error',
                    'error': str(e),
                    'timestamp': time.time()
                }), 500
        
        @app.route('/api/encode', methods=['POST'])
        def encode_sensor_data():
            """Encode sensor data endpoint."""
            try:
                self.request_count += 1
                data = request.get_json()
                
                if not data:
                    return jsonify({'error': 'No data provided'}), 400
                
                # Encode sensor data
                encoded = self.sensor_encoder.encode_multimodal_state(data)
                
                return jsonify({
                    'encoded_vector': {
                        'dimension': encoded.dimension,
                        'sparsity': encoded.sparsity(),
                        'entropy': encoded.entropy()
                    },
                    'timestamp': time.time()
                })
                
            except Exception as e:
                self.error_count += 1
                self.logger.error("Encoding error", error=str(e))
                return jsonify({'error': str(e)}), 500
        
        @app.route('/api/learn', methods=['POST'])
        def learn_behavior():
            """Learn behavior from demonstration."""
            try:
                self.request_count += 1
                data = request.get_json()
                
                if 'demonstration' not in data or 'behavior_name' not in data:
                    return jsonify({'error': 'Missing demonstration or behavior_name'}), 400
                
                confidence = self.behavior_learner.learn_from_demonstration(
                    data['demonstration'],
                    data['behavior_name']
                )
                
                return jsonify({
                    'behavior_name': data['behavior_name'],
                    'learning_confidence': confidence,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                self.error_count += 1
                self.logger.error("Learning error", error=str(e))
                return jsonify({'error': str(e)}), 500
        
        @app.route('/api/execute', methods=['POST'])
        def execute_behavior():
            """Execute learned behavior."""
            try:
                self.request_count += 1
                data = request.get_json()
                
                if 'behavior_name' not in data or 'current_state' not in data:
                    return jsonify({'error': 'Missing behavior_name or current_state'}), 400
                
                result = self.behavior_learner.execute_behavior(
                    data['behavior_name'],
                    data['current_state']
                )
                
                return jsonify(result)
                
            except Exception as e:
                self.error_count += 1
                self.logger.error("Execution error", error=str(e))
                return jsonify({'error': str(e)}), 500
        
        @app.route('/api/status')
        def system_status():
            """Get comprehensive system status."""
            try:
                self.request_count += 1
                
                health_report = self.error_recovery.get_system_health_report()
                cluster_status = self.coordinator.get_cluster_status()
                learning_analytics = self.behavior_learner.analyze_learning_patterns()
                
                if self.gpu_accelerator:
                    gpu_stats = self.gpu_accelerator.get_performance_stats()
                else:
                    gpu_stats = {'gpu_available': False}
                
                return jsonify({
                    'service': {
                        'uptime_seconds': time.time() - self.start_time,
                        'request_count': self.request_count,
                        'error_count': self.error_count,
                        'version': '5.0.0'
                    },
                    'system_health': health_report,
                    'cluster_status': cluster_status,
                    'learning_analytics': learning_analytics,
                    'performance': gpu_stats,
                    'timestamp': time.time()
                })
                
            except Exception as e:
                self.error_count += 1
                self.logger.error("Status error", error=str(e))
                return jsonify({'error': str(e)}), 500
        
        @app.route('/dashboard')
        def dashboard():
            """Simple web dashboard."""
            dashboard_html = '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>HDC Robot Controller Dashboard</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                    .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; }
                    .card { background: white; margin: 20px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                    .status-good { color: #27ae60; font-weight: bold; }
                    .status-warning { color: #f39c12; font-weight: bold; }
                    .status-error { color: #e74c3c; font-weight: bold; }
                    .metric { display: inline-block; margin: 10px 20px 10px 0; }
                    .metric-value { font-size: 24px; font-weight: bold; }
                    .metric-label { font-size: 14px; color: #666; }
                </style>
                <script>
                    function updateStatus() {
                        fetch('/api/status')
                            .then(response => response.json())
                            .then(data => {
                                document.getElementById('status').innerHTML = JSON.stringify(data, null, 2);
                                document.getElementById('uptime').innerText = Math.floor(data.service.uptime_seconds) + 's';
                                document.getElementById('requests').innerText = data.service.request_count;
                                document.getElementById('health').innerText = data.system_health.overall_health.toFixed(3);
                            })
                            .catch(error => {
                                document.getElementById('status').innerText = 'Error loading status: ' + error;
                            });
                    }
                    
                    // Update every 10 seconds
                    setInterval(updateStatus, 10000);
                    // Initial load
                    window.onload = updateStatus;
                </script>
            </head>
            <body>
                <div class="header">
                    <h1>🤖 HDC Robot Controller Dashboard</h1>
                    <p>Enterprise Hyperdimensional Computing for Robotics</p>
                </div>
                
                <div class="card">
                    <h2>System Metrics</h2>
                    <div class="metric">
                        <div class="metric-value" id="uptime">-</div>
                        <div class="metric-label">Uptime</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="requests">-</div>
                        <div class="metric-label">Requests</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value" id="health">-</div>
                        <div class="metric-label">Health Score</div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>API Endpoints</h2>
                    <ul>
                        <li><strong>GET /health</strong> - System health check</li>
                        <li><strong>POST /api/encode</strong> - Encode sensor data</li>
                        <li><strong>POST /api/learn</strong> - Learn behavior from demonstration</li>
                        <li><strong>POST /api/execute</strong> - Execute learned behavior</li>
                        <li><strong>GET /api/status</strong> - Comprehensive system status</li>
                    </ul>
                </div>
                
                <div class="card">
                    <h2>Live System Status</h2>
                    <pre id="status" style="background: #f8f8f8; padding: 15px; overflow: auto; max-height: 400px;">Loading...</pre>
                </div>
            </body>
            </html>
            '''
            return render_template_string(dashboard_html)
        
        return app
    
    def _run_flask_app(self):
        """Run Flask application."""
        if self.app:
            self.app.run(
                host='0.0.0.0',
                port=self.config['health_check_port'],
                debug=False,
                threaded=True
            )


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print(f"\nReceived signal {signum}, shutting down...")
    if hasattr(signal_handler, 'service'):
        signal_handler.service.stop()
    sys.exit(0)


def main():
    """Main entry point."""
    try:
        print("🚀 HDC Robot Controller v5.0 - Starting...")
        
        # Create and start service
        service = HDCRobotControllerService(HDC_CONFIG)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        signal_handler.service = service
        
        # Start service
        service.start()
        
        print(f"✅ HDC Robot Controller service running!")
        print(f"   Health Check: http://localhost:{HDC_CONFIG['health_check_port']}/health")
        print(f"   Dashboard: http://localhost:{HDC_CONFIG['health_check_port']}/dashboard")
        print(f"   API Status: http://localhost:{HDC_CONFIG['health_check_port']}/api/status")
        print(f"   Coordinator Port: {HDC_CONFIG['coordinator_port']}")
        print("   Press Ctrl+C to stop...")
        
        # Keep running
        while service.running:
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"Service error: {str(e)}")
        sys.exit(1)
    finally:
        if 'service' in locals():
            service.stop()


if __name__ == "__main__":
    main()