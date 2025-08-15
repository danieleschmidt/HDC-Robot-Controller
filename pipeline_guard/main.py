#!/usr/bin/env python3
"""
Main entry point for Self-Healing Pipeline Guard
Orchestrates all components and provides unified service management

Features:
- Component initialization and lifecycle management
- Configuration management and validation
- Graceful shutdown handling
- Health monitoring and diagnostics
- Multi-service coordination
"""

import os
import sys
import time
import signal
import argparse
import asyncio
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import Pipeline Guard components
from pipeline_guard.core import PipelineGuard, GuardStatus
from pipeline_guard.monitoring import PipelineMonitor
from pipeline_guard.repair import AutoRepair
from pipeline_guard.security import SecurityFramework
from pipeline_guard.i18n import I18nManager, SupportedLanguage, Region

# Optional dashboard import
try:
    from pipeline_guard.dashboard import HealthDashboard
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False

class PipelineGuardService:
    """Main Pipeline Guard service orchestrator"""
    
    def __init__(self, config_path: Optional[str] = None, debug: bool = False):
        self.config_path = config_path
        self.debug = debug
        self.running = False
        
        # Service components
        self.pipeline_guard: Optional[PipelineGuard] = None
        self.monitor: Optional[PipelineMonitor] = None
        self.auto_repair: Optional[AutoRepair] = None
        self.security: Optional[SecurityFramework] = None
        self.i18n: Optional[I18nManager] = None
        self.dashboard: Optional[HealthDashboard] = None
        
        # Configuration
        self.config = self._load_configuration()
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Signal handling
        self._setup_signal_handlers()
        
        self.logger.info("Pipeline Guard Service initialized", debug=debug)
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load service configuration"""
        default_config = {
            "core": {
                "hdc_dimension": int(os.getenv("HDC_DIMENSION", "10000")),
                "enable_auto_repair": True,
                "enable_predictive_analysis": True,
                "enable_gpu": os.getenv("HDC_ENABLE_GPU", "false").lower() == "true"
            },
            "monitoring": {
                "collection_interval": 30,
                "max_concurrent_collectors": 5
            },
            "security": {
                "enabled": True,
                "rate_limit": {
                    "requests_per_minute": 60,
                    "requests_per_hour": 1000
                }
            },
            "dashboard": {
                "enabled": DASHBOARD_AVAILABLE,
                "host": "0.0.0.0",
                "port": int(os.getenv("HDC_HEALTH_CHECK_PORT", "8080"))
            },
            "i18n": {
                "default_language": os.getenv("DEFAULT_LANGUAGE", "en"),
                "default_region": os.getenv("DEFAULT_REGION", "na")
            },
            "logging": {
                "level": os.getenv("HDC_LOG_LEVEL", "INFO"),
                "file": os.getenv("HDC_LOG_FILE", "/app/logs/pipeline_guard.log")
            }
        }
        
        # Load from file if provided
        if self.config_path and Path(self.config_path).exists():
            try:
                import yaml
                with open(self.config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                
                # Merge configurations
                self._deep_update(default_config, file_config)
            except Exception as e:
                print(f"Failed to load config from {self.config_path}: {e}")
        
        return default_config
    
    def _deep_update(self, base: Dict[str, Any], update: Dict[str, Any]):
        """Deep update dictionary"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def _setup_logging(self) -> logging.Logger:
        """Setup service logging"""
        log_level = getattr(logging, self.config["logging"]["level"].upper())
        log_file = self.config["logging"]["file"]
        
        # Create log directory
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        return logging.getLogger('pipeline_guard_service')
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            self.stop()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def initialize_components(self):
        """Initialize all service components"""
        self.logger.info("Initializing Pipeline Guard components...")
        
        # Initialize I18n first
        self.i18n = I18nManager(
            default_language=SupportedLanguage(self.config["i18n"]["default_language"]),
            translations_dir="translations"
        )
        
        # Set locale
        language = SupportedLanguage(self.config["i18n"]["default_language"])
        region = Region(self.config["i18n"]["default_region"])
        self.i18n.set_locale(language, region)
        
        # Initialize security framework
        if self.config["security"]["enabled"]:
            self.security = SecurityFramework(self.config["security"])
            self.logger.info("Security framework initialized")
        
        # Initialize core pipeline guard
        self.pipeline_guard = PipelineGuard(
            config_path=self.config_path,
            hdc_dimension=self.config["core"]["hdc_dimension"],
            enable_auto_repair=self.config["core"]["enable_auto_repair"],
            enable_predictive_analysis=self.config["core"]["enable_predictive_analysis"]
        )
        self.logger.info("Pipeline Guard core initialized")
        
        # Initialize monitoring
        self.monitor = PipelineMonitor(
            collection_interval=self.config["monitoring"]["collection_interval"]
        )
        self.logger.info("Pipeline Monitor initialized")
        
        # Initialize auto repair
        self.auto_repair = AutoRepair(
            hdc_dimension=self.config["core"]["hdc_dimension"],
            max_concurrent_repairs=3
        )
        self.logger.info("Auto Repair system initialized")
        
        # Initialize dashboard if available and enabled
        if self.config["dashboard"]["enabled"] and DASHBOARD_AVAILABLE:
            self.dashboard = HealthDashboard(
                self.pipeline_guard,
                self.monitor,
                host=self.config["dashboard"]["host"],
                port=self.config["dashboard"]["port"]
            )
            self.logger.info("Health Dashboard initialized")
        elif self.config["dashboard"]["enabled"]:
            self.logger.warning("Dashboard requested but Flask/SocketIO not available")
        
        self.logger.info("All components initialized successfully")
    
    def start(self):
        """Start all service components"""
        if self.running:
            self.logger.warning("Service already running")
            return
        
        self.logger.info("Starting Pipeline Guard Service...")
        self.running = True
        
        try:
            # Initialize components
            self.initialize_components()
            
            # Start monitoring
            if self.monitor:
                pipelines = self._get_initial_pipelines()
                if pipelines:
                    self.monitor.start_monitoring(pipelines)
                    self.logger.info(f"Started monitoring {len(pipelines)} pipelines")
            
            # Start pipeline guard
            if self.pipeline_guard:
                self.pipeline_guard.start_monitoring()
                self.logger.info("Pipeline Guard monitoring started")
            
            # Start dashboard
            if self.dashboard:
                dashboard_thread = threading.Thread(
                    target=self.dashboard.start_dashboard,
                    kwargs={"debug": self.debug},
                    daemon=True
                )
                dashboard_thread.start()
                self.logger.info(f"Health Dashboard started on port {self.config['dashboard']['port']}")
            
            self.logger.info("🛡️ Pipeline Guard Service started successfully!")
            self.logger.info("Service endpoints:")
            if self.dashboard:
                self.logger.info(f"  Dashboard: http://{self.config['dashboard']['host']}:{self.config['dashboard']['port']}")
                self.logger.info(f"  Health Check: http://{self.config['dashboard']['host']}:{self.config['dashboard']['port']}/health")
                self.logger.info(f"  API Status: http://{self.config['dashboard']['host']}:{self.config['dashboard']['port']}/api/status")
            
        except Exception as e:
            self.logger.error(f"Failed to start service: {e}")
            self.stop()
            raise
    
    def stop(self):
        """Stop all service components"""
        if not self.running:
            return
        
        self.logger.info("Stopping Pipeline Guard Service...")
        self.running = False
        
        try:
            # Stop components in reverse order
            if self.pipeline_guard:
                self.pipeline_guard.stop_monitoring()
                self.logger.info("Pipeline Guard monitoring stopped")
            
            if self.monitor:
                self.monitor.stop_monitoring()
                self.logger.info("Pipeline Monitor stopped")
            
            # Dashboard stops automatically with daemon threads
            
            self.logger.info("Pipeline Guard Service stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def _get_initial_pipelines(self) -> list:
        """Get initial list of pipelines to monitor"""
        # This would normally load from configuration or discovery
        # For now, return empty list - pipelines will be registered via API
        return []
    
    def run(self):
        """Run the service (blocking)"""
        self.start()
        
        try:
            # Keep service running
            while self.running:
                time.sleep(1.0)
                
                # Periodic health checks
                if self.pipeline_guard:
                    status = self.pipeline_guard.status
                    if status != GuardStatus.HEALTHY:
                        self.logger.warning(f"Pipeline Guard status: {status.value}")
        
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        finally:
            self.stop()
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status"""
        status = {
            "service": {
                "running": self.running,
                "version": "1.0.0",
                "uptime": "unknown"
            },
            "components": {}
        }
        
        if self.pipeline_guard:
            status["components"]["pipeline_guard"] = {
                "status": self.pipeline_guard.status.value,
                "active_pipelines": len(self.pipeline_guard.active_pipelines),
                "alerts": len(self.pipeline_guard.alert_history)
            }
        
        if self.monitor:
            monitor_stats = self.monitor.get_monitoring_statistics()
            status["components"]["monitor"] = {
                "active": monitor_stats["monitoring_active"],
                "collectors": len(monitor_stats["collectors"]),
                "metrics_collected": monitor_stats["data_storage"]["metrics_in_buffer"]
            }
        
        if self.auto_repair:
            repair_stats = self.auto_repair.get_repair_statistics()
            status["components"]["auto_repair"] = {
                "total_attempts": repair_stats["total_attempts"],
                "success_rate": repair_stats.get("overall_success_rate", 0.0)
            }
        
        if self.security:
            security_status = self.security.get_security_status()
            status["components"]["security"] = {
                "enabled": True,
                "blocked_ips": security_status["blocked_ips_count"],
                "active_sessions": security_status["active_sessions"]
            }
        
        status["components"]["dashboard"] = {
            "enabled": self.dashboard is not None,
            "available": DASHBOARD_AVAILABLE
        }
        
        return status

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Self-Healing Pipeline Guard")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Configuration file path"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show service status and exit"
    )
    parser.add_argument(
        "--version", "-v",
        action="store_true",
        help="Show version and exit"
    )
    
    args = parser.parse_args()
    
    if args.version:
        print("Pipeline Guard v1.0.0")
        print("Built by Terragon Labs")
        return
    
    # Initialize service
    service = PipelineGuardService(
        config_path=args.config,
        debug=args.debug
    )
    
    if args.status:
        # Initialize components to get status
        service.initialize_components()
        status = service.get_service_status()
        print(json.dumps(status, indent=2, default=str))
        return
    
    # Run service
    try:
        print("🛡️ Starting Self-Healing Pipeline Guard...")
        print("   Enterprise CI/CD Pipeline Monitoring & Recovery")
        print("   Built by Terragon Labs")
        print()
        
        service.run()
        
    except Exception as e:
        print(f"Service error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()