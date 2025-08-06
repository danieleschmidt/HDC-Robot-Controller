# HDC Robot Controller - Production Deployment Guide

## Overview

This guide covers production deployment of the HDC (Hyperdimensional Computing) Robot Controller using containerized infrastructure. The system supports both Docker Compose and Kubernetes deployments with full monitoring, scaling, and fault tolerance.

## Prerequisites

### Hardware Requirements
- **CPU**: 8+ cores (Intel/AMD x86_64)
- **RAM**: 16GB+ (32GB recommended)
- **GPU**: NVIDIA GPU with CUDA 11.8+ (optional but recommended)
- **Storage**: 100GB+ SSD storage
- **Network**: Gigabit Ethernet

### Software Requirements
- **Docker**: 20.10+ with Docker Compose 2.0+
- **Kubernetes**: 1.25+ (for K8s deployment)
- **NVIDIA Container Runtime**: For GPU support
- **Linux**: Ubuntu 20.04+ or RHEL 8+

## Quick Start (Docker Compose)

### 1. Clone and Prepare
```bash
git clone <repository-url>
cd hdc-robot-controller
```

### 2. Configure Environment
```bash
# Copy example configuration
cp config/production.example.yaml config/production.yaml
cp config/redis.example.conf config/redis.conf

# Edit configurations as needed
nano config/production.yaml
```

### 3. Build and Deploy
```bash
# Build production image
docker build -t terragon/hdc-robot-controller:v3.0 .

# Deploy full stack
docker-compose -f docker-compose.prod.yml up -d

# Check deployment status
docker-compose -f docker-compose.prod.yml ps
```

### 4. Verify Deployment
```bash
# Check API health
curl http://localhost:8080/health

# View logs
docker-compose -f docker-compose.prod.yml logs -f hdc-controller

# Access monitoring
open http://localhost:3000  # Grafana (admin/admin123)
open http://localhost:9090  # Prometheus
open http://localhost:5601  # Kibana
```

## Kubernetes Deployment

### 1. Prepare Cluster
```bash
# Create namespace
kubectl create namespace robotics

# Apply GPU device plugin (if using GPUs)
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.13.0/nvidia-device-plugin.yml
```

### 2. Deploy HDC System
```bash
# Deploy HDC Robot Controller
kubectl apply -f k8s/hdc-deployment.yaml

# Check deployment
kubectl get pods -n robotics
kubectl get services -n robotics
```

### 3. Configure Scaling
```bash
# Scale controller replicas
kubectl scale deployment hdc-robot-controller --replicas=3 -n robotics

# Scale worker nodes
kubectl scale daemonset hdc-worker-nodes --replicas=5 -n robotics
```

## Configuration

### Core HDC Configuration (`config/production.yaml`)
```yaml
hdc:
  dimension: 10000              # Hypervector dimension
  learning_rate: 0.01          # Learning adaptation rate
  similarity_threshold: 0.7    # Behavior matching threshold
  max_behaviors: 1000          # Maximum stored behaviors
  gpu_enabled: true            # Enable GPU acceleration

sensors:
  lidar:
    enabled: true
    topic: "/scan"
    encoding_dim: 1000
  camera:
    enabled: true
    topic: "/camera/image_raw"
    encoding_dim: 2000

control:
  frequency: 50                # Control loop frequency (Hz)
  safety_timeout: 1.0         # Safety timeout (seconds)
  max_velocity: 2.0           # Maximum velocity (m/s)

distributed:
  enabled: true               # Enable distributed processing
  chunk_size: 1000           # Processing chunk size

security:
  level: "HIGH"              # Security level (LOW/STANDARD/HIGH)
  session_timeout: 3600      # Session timeout (seconds)
  rate_limit: 100            # Rate limit (requests/minute)
```

### Redis Configuration (`config/redis.conf`)
```conf
# Network and security
bind 0.0.0.0
protected-mode no
port 6379

# Memory and persistence
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10

# Performance
tcp-keepalive 300
timeout 0
```

## Monitoring and Observability

### Grafana Dashboards
Access Grafana at `http://localhost:3000` (admin/admin123):
- **HDC System Overview**: Overall system health and performance
- **ROS2 Metrics**: Robot node performance and message flow
- **GPU Utilization**: CUDA acceleration metrics
- **Distributed Processing**: Worker node and task distribution
- **Security Events**: Authentication and access control logs

### Prometheus Metrics
Key metrics collected:
- `hdc_perception_latency_seconds`: Perception processing time
- `hdc_learning_accuracy_ratio`: Learning success rate
- `hdc_memory_usage_bytes`: Memory consumption
- `hdc_gpu_utilization_percent`: GPU usage
- `hdc_behaviors_stored_total`: Number of learned behaviors
- `hdc_control_loop_frequency_hz`: Control loop performance

### Log Aggregation (ELK Stack)
Logs are centralized in Elasticsearch and visualized in Kibana:
- **Application Logs**: HDC controller and worker logs
- **ROS2 Logs**: Robot node communication logs
- **System Logs**: Container and infrastructure logs
- **Security Logs**: Authentication and access events

## Scaling and Performance

### Horizontal Scaling

**Docker Compose:**
```bash
# Scale worker nodes
docker-compose -f docker-compose.prod.yml up -d --scale hdc-worker-1=3 --scale hdc-worker-2=5
```

**Kubernetes:**
```bash
# Scale controller pods
kubectl scale deployment hdc-robot-controller --replicas=5 -n robotics

# Scale worker daemonset
kubectl patch daemonset hdc-worker-nodes -n robotics -p '{"spec":{"template":{"spec":{"nodeSelector":{"worker":"enabled"}}}}}'
```

### Performance Tuning

**GPU Acceleration:**
- Ensure NVIDIA Container Runtime is installed
- Set `HDC_GPU_ENABLED=true`
- Monitor GPU utilization in Grafana

**Memory Optimization:**
- Adjust `maxmemory` in Redis configuration
- Monitor memory usage metrics
- Use persistent volumes for large datasets

**Network Performance:**
- Use dedicated network for inter-service communication
- Enable connection pooling in distributed workers
- Monitor network latency metrics

## Security Configuration

### Production Security Settings
```yaml
security:
  level: "HIGH"
  encryption_enabled: true
  session_timeout: 1800        # 30 minutes
  rate_limit: 50              # Conservative rate limiting
  audit_logging: true         # Enable audit trails
  cors_enabled: false         # Disable CORS in production
  https_only: true           # Require HTTPS
```

### SSL/TLS Configuration
1. Place SSL certificates in `./ssl/` directory
2. Configure NGINX with proper SSL settings
3. Enable HTTPS-only mode in HDC configuration
4. Set up certificate auto-renewal

### Access Control
- Configure role-based permissions in `config/rbac.yaml`
- Set up API keys for external service access
- Enable audit logging for security events
- Regular security scan and penetration testing

## Troubleshooting

### Common Issues

**GPU Not Detected:**
```bash
# Check GPU availability
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# Verify NVIDIA Container Runtime
docker info | grep nvidia
```

**High Memory Usage:**
```bash
# Check memory consumption
docker stats
kubectl top pods -n robotics

# Optimize Redis memory
redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

**Poor Performance:**
```bash
# Check CPU/GPU utilization
docker exec hdc-robot-controller nvidia-smi
docker exec hdc-robot-controller top

# Validate configuration
docker exec hdc-robot-controller python3 -c "from hdc_robot_controller.core import validate_config; validate_config()"
```

### Health Checks

**System Health:**
```bash
# Overall system health
curl http://localhost:8080/health

# Detailed diagnostics  
curl http://localhost:8080/diagnostics

# Redis connectivity
docker exec hdc-redis redis-cli ping

# Worker node status
curl http://localhost:8080/cluster/status
```

**Performance Validation:**
```bash
# Run benchmark suite
docker exec hdc-robot-controller python3 scripts/benchmark_suite.py

# Quality validation
docker exec hdc-robot-controller python3 scripts/quality_checker.py
```

## Backup and Recovery

### Data Backup
```bash
# Backup HDC data
docker run --rm -v hdc_data:/data -v $(pwd):/backup alpine tar czf /backup/hdc-data-$(date +%Y%m%d).tar.gz /data

# Backup Redis data
docker run --rm -v redis_data:/data -v $(pwd):/backup alpine tar czf /backup/redis-data-$(date +%Y%m%d).tar.gz /data
```

### Disaster Recovery
```bash
# Restore HDC data
docker run --rm -v hdc_data:/data -v $(pwd):/backup alpine tar xzf /backup/hdc-data-YYYYMMDD.tar.gz -C /

# Restore Redis data
docker run --rm -v redis_data:/data -v $(pwd):/backup alpine tar xzf /backup/redis-data-YYYYMMDD.tar.gz -C /

# Restart services
docker-compose -f docker-compose.prod.yml restart
```

## Maintenance

### Regular Maintenance Tasks
- **Weekly**: Check system logs and performance metrics
- **Monthly**: Update container images and security patches  
- **Quarterly**: Full system backup and disaster recovery testing
- **Yearly**: Security audit and penetration testing

### Update Procedure
```bash
# Pull latest image
docker pull terragon/hdc-robot-controller:v3.1

# Rolling update (Docker Compose)
docker-compose -f docker-compose.prod.yml up -d --no-deps hdc-controller

# Rolling update (Kubernetes)
kubectl set image deployment/hdc-robot-controller hdc-controller=terragon/hdc-robot-controller:v3.1 -n robotics
```

## Support and Contact

For production support and enterprise features:
- **Technical Support**: support@terragon-labs.com
- **Documentation**: https://docs.terragon-labs.com/hdc
- **GitHub Issues**: https://github.com/terragon-labs/hdc-robot-controller/issues
- **Enterprise Sales**: enterprise@terragon-labs.com

---

**Deployment Checklist:**
- [ ] Hardware requirements verified
- [ ] Software dependencies installed
- [ ] Configuration files customized
- [ ] SSL certificates configured
- [ ] Monitoring dashboards accessible
- [ ] Health checks passing
- [ ] Backup procedures tested
- [ ] Security settings validated
- [ ] Performance benchmarks run
- [ ] Team training completed

**Status: PRODUCTION READY** ✅