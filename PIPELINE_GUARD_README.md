# 🛡️ Self-Healing Pipeline Guard

[![Production Ready](https://img.shields.io/badge/Production-Ready-brightgreen.svg)]()
[![Security](https://img.shields.io/badge/Security-Enterprise-blue.svg)]()
[![Global](https://img.shields.io/badge/Global-Multi--Language-purple.svg)]()
[![Autonomous](https://img.shields.io/badge/Autonomous-Self--Healing-gold.svg)]()

**Enterprise-grade self-healing CI/CD pipeline monitoring and recovery system** built on advanced HDC (Hyperdimensional Computing) pattern recognition and autonomous repair mechanisms.

## 🌟 Key Features

### 🚀 **Intelligent Monitoring**
- **Real-time Pipeline Health Assessment** with multi-dimensional analysis
- **HDC-powered Anomaly Detection** using hyperdimensional pattern recognition  
- **Predictive Failure Analysis** with trend-based risk assessment
- **Multi-source Metrics Collection** (Jenkins, GitLab, GitHub Actions, Azure DevOps)

### 🤖 **Autonomous Repair**
- **Self-healing Mechanisms** with intelligent repair strategy selection
- **Automated Root Cause Analysis** using pattern matching
- **Progressive Repair Escalation** from simple fixes to complete rollbacks
- **Learning-based Optimization** that improves repair success over time

### 📊 **Advanced Analytics**
- **Real-time Health Dashboard** with interactive visualizations
- **Comprehensive Audit Logging** for compliance and forensics
- **Statistical Trend Analysis** with seasonal pattern recognition
- **Performance Benchmarking** against historical baselines

### 🔐 **Enterprise Security**
- **Multi-level Authentication** with API key and session management
- **Role-based Access Control** (Readonly, Operator, Admin, Superuser)
- **Rate Limiting & DoS Protection** with automatic IP blocking
- **Input Validation & Sanitization** against injection attacks
- **Comprehensive Security Audit Logging** with threat detection

### 🌍 **Global-First Design**
- **Multi-language Support** (12 languages including RTL)
- **Regional Compliance** (GDPR, CCPA, SOC2)
- **Localized Formatting** for dates, numbers, currencies
- **Cultural Adaptation** for notifications and alerts

## 🏗️ Architecture

```
pipeline_guard/
├── core.py              # Main orchestration and pipeline management
├── detection.py         # Anomaly detection and failure prediction  
├── repair.py           # Automated repair mechanisms
├── monitoring.py       # Metrics collection and processing
├── dashboard.py        # Real-time web dashboard
├── security.py         # Security framework and authentication
└── i18n.py            # Internationalization and localization
```

### Core Components

**Pipeline Guard Core** (`core.py`)
- Central orchestration of monitoring, detection, and repair
- Pipeline registration and health assessment
- Alert generation and escalation management
- HDC-based pattern learning for failure prediction

**Anomaly Detector** (`detection.py`)
- Multi-method anomaly detection (statistical, pattern-based, temporal)
- HDC hypervector similarity analysis for pattern deviation
- Trend analysis with linear regression and seasonal adjustment
- Correlation analysis for detecting unusual metric relationships

**Auto Repair Engine** (`repair.py`)
- Intelligent repair strategy selection using historical success data
- Multiple repair engines (Generic, Jenkins, GitLab, custom)
- Sequential and parallel repair execution
- Learning-based optimization with pattern recognition

**Pipeline Monitor** (`monitoring.py`)
- Multi-source metrics collection with plugin architecture
- Real-time metric processing and aggregation
- Historical data management with automatic cleanup
- Health scoring and trend calculation

**Security Framework** (`security.py`)
- Authentication and authorization with multiple methods
- Input validation and injection attack prevention
- Rate limiting with automatic blocking
- Comprehensive audit logging and threat detection

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/terragon-labs/self-healing-pipeline-guard.git
cd self-healing-pipeline-guard

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp config/pipeline_guard.yaml.example config/pipeline_guard.yaml
# Edit configuration as needed
```

### Basic Usage

```python
#!/usr/bin/env python3
from pipeline_guard import PipelineGuard, PipelineMonitor, HealthDashboard

# Initialize components
guard = PipelineGuard(
    config_path="config/pipeline_guard.yaml",
    enable_auto_repair=True,
    enable_predictive_analysis=True
)

monitor = PipelineMonitor(collection_interval=30)
dashboard = HealthDashboard(guard, monitor, port=8080)

# Register pipelines to monitor
guard.register_pipeline("web-app-ci", {
    "expected_duration": 480,  # 8 minutes
    "success_rate": 0.94,
    "alert_email": "team@company.com"
})

guard.register_pipeline("api-service-cd", {
    "expected_duration": 720,  # 12 minutes
    "success_rate": 0.97,
    "critical": True
})

# Start monitoring
monitor.start_monitoring(["web-app-ci", "api-service-cd"])
guard.start_monitoring()

# Start dashboard (optional)
dashboard.start_dashboard()
```

### Docker Deployment

```bash
# Build container
docker build -t pipeline-guard .

# Run with configuration
docker run -d \
  --name pipeline-guard \
  -p 8080:8080 \
  -v /path/to/config:/app/config \
  -v /path/to/logs:/app/logs \
  pipeline-guard

# Check health
curl http://localhost:8080/health
```

### Kubernetes Deployment

```bash
# Apply deployment
kubectl apply -f k8s/

# Scale for high availability
kubectl scale deployment pipeline-guard --replicas=3

# Check status
kubectl get pods -l app=pipeline-guard
```

## 📊 Dashboard & Monitoring

### Real-time Dashboard

Access the web dashboard at `http://localhost:8080` for:

- **System Overview** - Overall health status and key metrics
- **Pipeline Status** - Individual pipeline health and performance
- **Active Alerts** - Current issues requiring attention
- **Repair History** - Automated repair attempts and success rates
- **Performance Charts** - Real-time performance visualization

### REST API Endpoints

```bash
# System status
GET /api/status

# Pipeline health
GET /api/pipelines

# Recent alerts
GET /api/alerts?hours=24

# Pipeline metrics
GET /api/metrics/{pipeline_id}?hours=24

# Export data
GET /api/export/{pipeline_id}?format=json
```

### WebSocket Real-time Updates

```javascript
const socket = io('http://localhost:8080');

socket.on('system_status', (data) => {
    console.log('System status:', data);
});

socket.on('pipeline_update', (data) => {
    console.log('Pipeline update:', data.pipeline_id, data.data);
});

socket.on('alerts_update', (alerts) => {
    console.log('New alerts:', alerts);
});
```

## 🔧 Configuration

### Pipeline Guard Configuration

```yaml
# config/pipeline_guard.yaml
monitoring:
  interval_seconds: 30
  metrics_retention_days: 30
  alert_retention_days: 90

thresholds:
  health_score_warning: 0.7
  health_score_critical: 0.5
  error_rate_warning: 0.05
  error_rate_critical: 0.15
  performance_degradation_threshold: 0.3

auto_repair:
  enabled: true
  max_attempts: 3
  cooldown_minutes: 10
  escalation_enabled: true

security:
  rate_limit:
    requests_per_minute: 60
    requests_per_hour: 1000
  auto_block_threshold: 10
  session_timeout: 3600

notifications:
  email_enabled: true
  slack_enabled: true
  webhook_url: "https://hooks.slack.com/..."

compliance:
  region: "eu"  # eu, na, apac
  data_retention_days: 365
  audit_logging: true
```

### Environment Variables

```bash
# Core configuration
HDC_DIMENSION=10000
HDC_LOG_LEVEL=INFO
HDC_ENABLE_GPU=true

# Security
PIPELINE_GUARD_SECRET_KEY=your-secret-key
PIPELINE_GUARD_ENCRYPTION_KEY=your-encryption-key

# Database (optional)
DATABASE_URL=postgresql://user:pass@host:5432/pipeline_guard

# External integrations
JENKINS_URL=https://jenkins.company.com
JENKINS_USERNAME=pipeline-guard
JENKINS_TOKEN=your-jenkins-token

GITLAB_URL=https://gitlab.company.com
GITLAB_TOKEN=your-gitlab-token
```

## 🛠️ Advanced Configuration

### Custom Metric Collectors

```python
from pipeline_guard.monitoring import MetricCollector, MetricPoint, MetricSource

class CustomCollector(MetricCollector):
    def collect_metrics(self, pipeline_id: str):
        # Your custom metric collection logic
        return [
            MetricPoint(
                timestamp=datetime.now(),
                pipeline_id=pipeline_id,
                metric_name="custom_metric",
                value=42.0,
                source=MetricSource.CUSTOM
            )
        ]
    
    def get_supported_pipelines(self):
        return ["custom-pipeline"]
    
    def is_available(self):
        return True

# Register custom collector
monitor.register_collector(CustomCollector())
```

### Custom Repair Engines

```python
from pipeline_guard.repair import RepairEngine, RepairAction, RepairStrategy

class CustomRepairEngine(RepairEngine):
    def can_handle(self, alert):
        return "custom" in alert.pipeline_id
    
    def generate_repair_actions(self, alert):
        return [
            RepairAction(
                id="custom-repair-1",
                strategy=RepairStrategy.CUSTOM_SCRIPT,
                description="Custom repair action",
                command="./custom_repair.sh",
                timeout_seconds=300
            )
        ]
    
    def execute_repair(self, action, alert):
        # Your custom repair logic
        pass

# Register custom repair engine
auto_repair.register_engine(CustomRepairEngine())
```

### Multi-language Setup

```python
from pipeline_guard.i18n import SupportedLanguage, Region, set_language

# Set language for notifications
set_language(SupportedLanguage.SPANISH, Region.LATIN_AMERICA)

# Custom translations
i18n = get_i18n_manager()
custom_translations = {
    "custom.alert.title": "Alerta Personalizada",
    "custom.alert.message": "Problema detectado en {pipeline_id}"
}

# Add custom translations
i18n.translations['es'].update(custom_translations)
```

## 🔐 Security Configuration

### User Management

```python
from pipeline_guard.security import SecurityFramework, AccessLevel

security = SecurityFramework()

# Create users
admin_user = security.authenticator.create_user(
    "admin", AccessLevel.ADMIN, ["pipelines:*", "alerts:*"]
)

operator_user = security.authenticator.create_user(
    "operator", AccessLevel.OPERATOR, ["pipelines:read", "repairs:execute"]
)

readonly_user = security.authenticator.create_user(
    "viewer", AccessLevel.READONLY, ["pipelines:read", "alerts:read"]
)

print(f"Admin API Key: {admin_user.api_key}")
print(f"Operator API Key: {operator_user.api_key}")
```

### API Authentication

```bash
# Using API key in header
curl -H "X-API-Key: pg_your_api_key_here" \
     http://localhost:8080/api/status

# Using API key in query parameter
curl "http://localhost:8080/api/status?api_key=pg_your_api_key_here"
```

### Rate Limiting Configuration

```python
# Custom rate limiting
security = SecurityFramework({
    'rate_limit': {
        'requests_per_minute': 30,  # Stricter limit
        'requests_per_hour': 500
    },
    'auto_block_threshold': 5,      # Block after 5 violations
    'block_duration_minutes': 120   # 2-hour block
})
```

## 📈 Monitoring & Alerting

### Alert Types

- **Performance Degradation** - Duration increases, throughput decreases
- **Error Rate Spikes** - Unusual error patterns or failure rates
- **Resource Anomalies** - CPU, memory, or disk usage anomalies
- **Pattern Deviations** - HDC-detected behavioral changes
- **Security Events** - Authentication failures, injection attempts
- **System Health** - Overall system degradation

### Notification Channels

```python
# Email notifications
notifications = {
    "email": {
        "smtp_server": "smtp.company.com",
        "smtp_port": 587,
        "username": "alerts@company.com",
        "password": "smtp-password",
        "recipients": ["team@company.com", "oncall@company.com"]
    },
    "slack": {
        "webhook_url": "https://hooks.slack.com/services/...",
        "channel": "#alerts",
        "username": "Pipeline Guard"
    },
    "webhook": {
        "url": "https://api.company.com/alerts",
        "method": "POST",
        "headers": {"Authorization": "Bearer token"}
    }
}
```

### Custom Alert Rules

```python
# Define custom alert conditions
custom_rules = [
    {
        "name": "High Error Rate",
        "condition": "error_rate > 0.1",
        "severity": "critical",
        "action": "immediate_repair"
    },
    {
        "name": "Performance Degradation",
        "condition": "duration > baseline * 1.5",
        "severity": "warning", 
        "action": "performance_repair"
    },
    {
        "name": "Resource Pressure",
        "condition": "cpu_usage > 0.9 AND memory_usage > 0.85",
        "severity": "high",
        "action": "scale_resources"
    }
]
```

## 🧪 Testing & Validation

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=pipeline_guard --cov-report=html

# Run specific test suite
python -m pytest tests/test_pipeline_guard.py -v

# Run integration tests
python -m pytest tests/test_integration.py -v
```

### Load Testing

```bash
# Install load testing tools
pip install locust

# Run load tests
locust -f tests/load_test.py --host=http://localhost:8080
```

### Security Testing

```bash
# Run security scan
python tests/security_scan.py

# Check for vulnerabilities
safety check

# Audit dependencies
pip-audit
```

## 📋 Production Deployment

### Prerequisites

- Python 3.9+
- Redis (for caching and sessions)
- PostgreSQL (optional, for persistent storage)
- Docker & Kubernetes (for containerized deployment)

### Production Checklist

- [ ] Configure all environment variables
- [ ] Set up monitoring and logging
- [ ] Configure backup and disaster recovery
- [ ] Set up SSL/TLS certificates
- [ ] Configure firewall and network security
- [ ] Set up CI/CD for Pipeline Guard itself
- [ ] Configure monitoring alerts for the monitoring system
- [ ] Test failover and recovery procedures
- [ ] Document runbooks and procedures
- [ ] Train operations team

### High Availability Setup

```yaml
# k8s/pipeline-guard-ha.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pipeline-guard
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  template:
    spec:
      containers:
      - name: pipeline-guard
        image: pipeline-guard:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Monitoring the Monitor

```python
# Set up monitoring for Pipeline Guard itself
pipeline_guard_monitor = PipelineMonitor()
pipeline_guard_monitor.register_pipeline("pipeline-guard-system", {
    "expected_duration": 60,
    "success_rate": 0.99,
    "critical": True
})
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/terragon-labs/self-healing-pipeline-guard.git
cd self-healing-pipeline-guard

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
python -m pytest
```

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

## 🚨 Support

- **Documentation**: [Full Documentation](https://pipeline-guard.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/terragon-labs/self-healing-pipeline-guard/issues)
- **Discussions**: [GitHub Discussions](https://github.com/terragon-labs/self-healing-pipeline-guard/discussions)
- **Enterprise Support**: enterprise@terragon-labs.com
- **Security Issues**: security@terragon-labs.com

## 🏆 Recognition

Built with ❤️ by [Terragon Labs](https://terragon-labs.com) using advanced HDC (Hyperdimensional Computing) research and autonomous system principles.

---

**🛡️ Self-Healing Pipeline Guard** - Because your CI/CD pipelines deserve autonomous protection.

*Intelligent • Autonomous • Global • Secure*