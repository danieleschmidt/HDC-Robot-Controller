"""
Advanced Multi-Modal Fusion Engine with Transformer Architecture

Combines HDC with state-of-the-art transformer models for superior
multi-modal understanding and reasoning capabilities.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import time
from pathlib import Path
from collections import deque
import logging
from concurrent.futures import ThreadPoolExecutor
import threading

from ..core.hypervector import HyperVector
from ..core.operations import HDCOperations


@dataclass
class ModalityConfig:
    """Configuration for each modality."""
    name: str
    dimension: int
    encoder_type: str  # 'transformer', 'cnn', 'rnn', 'hdc', 'neuromorphic'
    attention_heads: int = 8
    hidden_dim: int = 512
    dropout: float = 0.1
    uncertainty_estimation: bool = True
    temporal_window: float = 1.0  # seconds
    quality_threshold: float = 0.7
    neuromorphic_enabled: bool = False
    

class TransformerHDCEncoder(nn.Module):
    """Hybrid Transformer-HDC encoder for modality processing."""
    
    def __init__(self, config: ModalityConfig, hdc_dim: int = 10000):
        super().__init__()
        self.config = config
        self.hdc_dim = hdc_dim
        
        # Transformer components
        self.embedding = nn.Linear(config.dimension, config.hidden_dim)
        self.positional_encoding = self._create_positional_encoding()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        # HDC projection
        self.hdc_projector = nn.Sequential(
            nn.Linear(config.hidden_dim, hdc_dim),
            nn.Tanh()
        )
        
        # Attention mechanism for HDC binding
        self.attention_weights = nn.MultiheadAttention(
            embed_dim=hdc_dim,
            num_heads=16,
            batch_first=True
        )
        
    def _create_positional_encoding(self) -> nn.Parameter:
        """Create learnable positional encodings."""
        max_seq_len = 1000
        pe = torch.zeros(max_seq_len, self.config.hidden_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, self.config.hidden_dim, 2) *
                           -(np.log(10000.0) / self.config.hidden_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return nn.Parameter(pe.unsqueeze(0), requires_grad=True)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, HyperVector]:
        """Forward pass with both neural and HDC outputs."""
        batch_size, seq_len, _ = x.shape
        
        # Transformer processing
        embedded = self.embedding(x)
        
        # Add positional encoding
        pos_enc = self.positional_encoding[:, :seq_len, :]
        embedded = embedded + pos_enc
        
        # Transform
        transformed = self.transformer(embedded)
        
        # Global pooling
        pooled = transformed.mean(dim=1)  # [batch_size, hidden_dim]
        
        # Project to HDC space
        hdc_features = self.hdc_projector(pooled)
        
        # Apply attention for HDC binding
        attended, _ = self.attention_weights(
            hdc_features.unsqueeze(1), 
            hdc_features.unsqueeze(1), 
            hdc_features.unsqueeze(1)
        )
        hdc_output = attended.squeeze(1)
        
        # Convert to HyperVector
        hdc_binary = torch.sign(hdc_output).int().cpu().numpy()
        hypervector = HyperVector(
            dimension=self.hdc_dim,
            data=hdc_binary[0]  # Take first batch item
        )
        
        return pooled, hypervector


class MultiModalFusionEngine:
    """Advanced multi-modal fusion using Transformer-HDC hybrid architecture."""
    
    def __init__(self, 
                 modality_configs: List[ModalityConfig],
                 hdc_dimension: int = 10000,
                 fusion_strategy: str = "hierarchical_attention",
                 learning_rate: float = 1e-4):
        """
        Initialize multi-modal fusion engine.
        
        Args:
            modality_configs: Configuration for each modality
            hdc_dimension: Dimension of hypervectors
            fusion_strategy: Strategy for modality fusion
            learning_rate: Learning rate for neural components
        """
        self.modality_configs = {cfg.name: cfg for cfg in modality_configs}
        self.hdc_dimension = hdc_dimension
        self.fusion_strategy = fusion_strategy
        
        # Initialize encoders for each modality
        self.encoders = {}
        self.optimizers = {}
        
        for config in modality_configs:
            encoder = TransformerHDCEncoder(config, hdc_dimension)
            optimizer = torch.optim.AdamW(encoder.parameters(), lr=learning_rate)
            
            self.encoders[config.name] = encoder
            self.optimizers[config.name] = optimizer
            
        # Fusion components
        self._init_fusion_components()
        
        # Performance tracking
        self.fusion_metrics = {
            'total_fusions': 0,
            'avg_fusion_time': 0.0,
            'modality_contributions': {},
            'attention_patterns': []
        }
        
    def _init_fusion_components(self):
        """Initialize fusion-specific neural components."""
        total_dim = len(self.encoders) * 512  # Hidden dimension per encoder
        
        if self.fusion_strategy == "hierarchical_attention":
            # Multi-level attention fusion
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=512,
                num_heads=16,
                batch_first=True
            )
            
            self.fusion_projector = nn.Sequential(
                nn.Linear(total_dim, 1024),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(1024, self.hdc_dimension),
                nn.Tanh()
            )
            
        elif self.fusion_strategy == "cross_modal_transformer":
            # Cross-modal transformer fusion
            fusion_layer = nn.TransformerEncoderLayer(
                d_model=512,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1,
                batch_first=True
            )
            self.cross_modal_transformer = nn.TransformerEncoder(
                fusion_layer, num_layers=4
            )
            
        # HDC-based fusion operations
        self.hdc_ops = HDCOperations()
        
        # Adaptive weighting for modalities
        self.modality_weights = nn.Parameter(
            torch.ones(len(self.encoders)) / len(self.encoders)
        )
        
    def encode_modality(self, 
                       modality_name: str, 
                       data: np.ndarray,
                       context: Optional[Dict] = None) -> Tuple[torch.Tensor, HyperVector]:
        """
        Encode a single modality with both neural and HDC representations.
        
        Args:
            modality_name: Name of the modality
            data: Input data for the modality
            context: Additional context information
            
        Returns:
            Tuple of (neural_features, hypervector)
        """
        if modality_name not in self.encoders:
            raise ValueError(f"Unknown modality: {modality_name}")
            
        encoder = self.encoders[modality_name]
        
        # Convert to tensor
        if isinstance(data, np.ndarray):
            if data.ndim == 2:
                data = torch.from_numpy(data).unsqueeze(0)  # Add batch dim
            elif data.ndim == 1:
                data = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
            else:
                data = torch.from_numpy(data)
        
        data = data.float()
        
        # Encode with transformer-HDC hybrid
        with torch.no_grad():
            neural_features, hypervector = encoder(data)
            
        # Apply context if provided
        if context:
            hypervector = self._apply_context(hypervector, context)
            
        return neural_features, hypervector
        
    def _apply_context(self, 
                      hypervector: HyperVector, 
                      context: Dict) -> HyperVector:
        """Apply contextual information to hypervector."""
        # Create context hypervector
        context_hv = HyperVector.zero(self.hdc_dimension)
        
        for key, value in context.items():
            # Simple context encoding - could be enhanced
            key_hv = self._encode_symbol(key)
            value_hv = self._encode_value(value)
            context_pair = key_hv.bind(value_hv)
            context_hv = context_hv.bundle(context_pair)
            
        # Bind with original hypervector
        return hypervector.bind(context_hv)
        
    def _encode_symbol(self, symbol: str) -> HyperVector:
        """Encode symbolic information as hypervector."""
        # Hash-based encoding for consistency
        hash_val = hash(symbol) % (2**31)
        np.random.seed(hash_val)
        data = np.random.choice([-1, 1], size=self.hdc_dimension)
        return HyperVector(self.hdc_dimension, data)
        
    def _encode_value(self, value: Any) -> HyperVector:
        """Encode various value types as hypervectors."""
        if isinstance(value, (int, float)):
            # Numerical encoding
            normalized = float(value) / 100.0  # Simple normalization
            data = np.random.choice(
                [-1, 1], 
                size=self.hdc_dimension,
                p=[max(0.1, 0.5 - normalized/2), min(0.9, 0.5 + normalized/2)]
            )
            return HyperVector(self.hdc_dimension, data)
        else:
            # String/other encoding
            return self._encode_symbol(str(value))
            
    def fuse_modalities(self, 
                       modality_data: Dict[str, np.ndarray],
                       context: Optional[Dict] = None,
                       adaptive_weighting: bool = True) -> Dict[str, Any]:
        """
        Fuse multiple modalities using advanced attention mechanisms.
        
        Args:
            modality_data: Dictionary mapping modality names to data
            context: Contextual information
            adaptive_weighting: Whether to use adaptive modality weighting
            
        Returns:
            Dictionary with fused representations and metadata
        """
        start_time = time.time()
        
        # Encode each modality
        neural_features = []
        hypervectors = []
        modality_names = []
        
        for modality_name, data in modality_data.items():
            neural_feat, hypervec = self.encode_modality(
                modality_name, data, context
            )
            
            neural_features.append(neural_feat)
            hypervectors.append(hypervec)
            modality_names.append(modality_name)
            
        # Neural fusion
        if self.fusion_strategy == "hierarchical_attention":
            fused_neural = self._hierarchical_attention_fusion(neural_features)
        elif self.fusion_strategy == "cross_modal_transformer":
            fused_neural = self._cross_modal_transformer_fusion(neural_features)
        else:
            # Simple concatenation fallback
            fused_neural = torch.cat(neural_features, dim=1)
            
        # HDC fusion with adaptive weighting
        if adaptive_weighting:
            weights = self._compute_adaptive_weights(hypervectors, context)
        else:
            weights = [1.0 / len(hypervectors)] * len(hypervectors)
            
        fused_hypervector = self._weighted_hdc_fusion(hypervectors, weights)
        
        # Update performance metrics
        fusion_time = time.time() - start_time
        self._update_metrics(fusion_time, modality_names, weights)
        
        # Compute attention patterns for interpretability
        attention_patterns = self._compute_attention_patterns(
            neural_features, hypervectors
        )
        
        return {
            'neural_features': fused_neural,
            'hypervector': fused_hypervector,
            'modality_weights': dict(zip(modality_names, weights)),
            'attention_patterns': attention_patterns,
            'fusion_time': fusion_time,
            'confidence': self._compute_fusion_confidence(hypervectors, weights)
        }
        
    def _hierarchical_attention_fusion(self, 
                                     neural_features: List[torch.Tensor]) -> torch.Tensor:
        """Fuse neural features using hierarchical attention."""
        # Stack features
        stacked = torch.stack(neural_features, dim=1)  # [batch, modalities, features]
        
        # Apply attention across modalities
        attended, attention_weights = self.fusion_attention(
            stacked, stacked, stacked
        )
        
        # Global pooling
        fused = attended.mean(dim=1)
        
        return fused
        
    def _cross_modal_transformer_fusion(self, 
                                      neural_features: List[torch.Tensor]) -> torch.Tensor:
        """Fuse neural features using cross-modal transformer."""
        # Stack features
        stacked = torch.stack(neural_features, dim=1)  # [batch, modalities, features]
        
        # Apply cross-modal transformer
        transformed = self.cross_modal_transformer(stacked)
        
        # Global pooling
        fused = transformed.mean(dim=1)
        
        return fused
        
    def _compute_adaptive_weights(self, 
                                hypervectors: List[HyperVector],
                                context: Optional[Dict] = None) -> List[float]:
        """Compute adaptive weights for modality fusion."""
        weights = []
        
        for i, hv in enumerate(hypervectors):
            # Base weight from learned parameters
            base_weight = torch.softmax(self.modality_weights, dim=0)[i].item()
            
            # Adjust based on hypervector quality
            entropy = hv.entropy()
            sparsity = hv.sparsity()
            
            # Higher entropy and balanced sparsity = higher weight
            quality_factor = entropy * (1 - abs(sparsity - 0.5))
            
            # Context-based adjustment
            context_factor = 1.0
            if context and 'modality_importance' in context:
                modality_name = list(self.modality_configs.keys())[i]
                context_factor = context['modality_importance'].get(modality_name, 1.0)
                
            final_weight = base_weight * quality_factor * context_factor
            weights.append(final_weight)
            
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(weights)] * len(weights)
            
        return weights
        
    def _weighted_hdc_fusion(self, 
                           hypervectors: List[HyperVector], 
                           weights: List[float]) -> HyperVector:
        """Fuse hypervectors with adaptive weighting."""
        if not hypervectors:
            return HyperVector.zero(self.hdc_dimension)
            
        # Create weighted bundle
        weighted_sum = np.zeros(self.hdc_dimension, dtype=np.float32)
        
        for hv, weight in zip(hypervectors, weights):
            weighted_sum += hv.data * weight
            
        # Apply threshold
        result_data = np.where(weighted_sum > 0, 1, -1).astype(np.int8)
        
        return HyperVector(self.hdc_dimension, result_data)
        
    def _compute_attention_patterns(self, 
                                  neural_features: List[torch.Tensor],
                                  hypervectors: List[HyperVector]) -> Dict[str, float]:
        """Compute attention patterns for interpretability."""
        patterns = {}
        
        # Neural attention analysis
        if len(neural_features) > 1:
            # Compute pairwise similarities
            similarities = []
            for i in range(len(neural_features)):
                for j in range(i+1, len(neural_features)):
                    feat_i = neural_features[i].flatten()
                    feat_j = neural_features[j].flatten()
                    
                    # Cosine similarity
                    sim = torch.cosine_similarity(feat_i, feat_j, dim=0).item()
                    similarities.append(sim)
                    
                    mod_i = list(self.modality_configs.keys())[i]
                    mod_j = list(self.modality_configs.keys())[j]
                    patterns[f'{mod_i}_{mod_j}_neural_similarity'] = sim
                    
        # HDC attention analysis
        for i, hv in enumerate(hypervectors):
            mod_name = list(self.modality_configs.keys())[i]
            patterns[f'{mod_name}_entropy'] = hv.entropy()
            patterns[f'{mod_name}_sparsity'] = hv.sparsity()
            
        return patterns
        
    def _compute_fusion_confidence(self, 
                                 hypervectors: List[HyperVector],
                                 weights: List[float]) -> float:
        """Compute confidence score for the fusion result."""
        # Based on consistency across modalities and weight distribution
        if len(hypervectors) < 2:
            return 1.0
            
        # Inter-modality similarity
        similarities = []
        for i in range(len(hypervectors)):
            for j in range(i+1, len(hypervectors)):
                sim = hypervectors[i].similarity(hypervectors[j])
                similarities.append(abs(sim))  # High similarity = high confidence
                
        avg_similarity = np.mean(similarities) if similarities else 0.0
        
        # Weight distribution evenness (high entropy = balanced fusion)
        weight_entropy = -sum(w * np.log(w + 1e-8) for w in weights if w > 0)
        max_entropy = np.log(len(weights))
        normalized_entropy = weight_entropy / max_entropy if max_entropy > 0 else 0
        
        # Combine factors
        confidence = 0.6 * avg_similarity + 0.4 * normalized_entropy
        
        return max(0.0, min(1.0, confidence))
        
    def _update_metrics(self, 
                      fusion_time: float, 
                      modality_names: List[str], 
                      weights: List[float]):
        """Update performance metrics."""
        self.fusion_metrics['total_fusions'] += 1
        
        # Update average fusion time
        total = self.fusion_metrics['total_fusions']
        current_avg = self.fusion_metrics['avg_fusion_time']
        self.fusion_metrics['avg_fusion_time'] = (
            (current_avg * (total - 1) + fusion_time) / total
        )
        
        # Update modality contributions
        for mod_name, weight in zip(modality_names, weights):
            if mod_name not in self.fusion_metrics['modality_contributions']:
                self.fusion_metrics['modality_contributions'][mod_name] = []
            self.fusion_metrics['modality_contributions'][mod_name].append(weight)
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        summary = {
            'total_fusions': self.fusion_metrics['total_fusions'],
            'avg_fusion_time_ms': self.fusion_metrics['avg_fusion_time'] * 1000,
            'modality_stats': {}
        }
        
        for mod_name, contributions in self.fusion_metrics['modality_contributions'].items():
            summary['modality_stats'][mod_name] = {
                'avg_contribution': np.mean(contributions),
                'std_contribution': np.std(contributions),
                'max_contribution': np.max(contributions),
                'min_contribution': np.min(contributions)
            }
            
        return summary
        
    def save_model(self, path: str):
        """Save the trained model components."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save encoders
        for name, encoder in self.encoders.items():
            torch.save(encoder.state_dict(), save_path / f'encoder_{name}.pt')
            
        # Save fusion components
        if hasattr(self, 'fusion_attention'):
            torch.save(self.fusion_attention.state_dict(), 
                      save_path / 'fusion_attention.pt')
                      
        if hasattr(self, 'cross_modal_transformer'):
            torch.save(self.cross_modal_transformer.state_dict(), 
                      save_path / 'cross_modal_transformer.pt')
                      
        # Save modality weights
        torch.save(self.modality_weights, save_path / 'modality_weights.pt')
        
        # Save metrics
        import json
        with open(save_path / 'metrics.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            metrics_copy = {}
            for k, v in self.fusion_metrics.items():
                if isinstance(v, dict):
                    metrics_copy[k] = {
                        k2: v2.tolist() if isinstance(v2, np.ndarray) else v2
                        for k2, v2 in v.items()
                    }
                else:
                    metrics_copy[k] = v
            json.dump(metrics_copy, f, indent=2)
            
    def load_model(self, path: str):
        """Load previously trained model components."""
        load_path = Path(path)
        
        # Load encoders
        for name, encoder in self.encoders.items():
            encoder_path = load_path / f'encoder_{name}.pt'
            if encoder_path.exists():
                encoder.load_state_dict(torch.load(encoder_path))
                
        # Load fusion components
        fusion_attention_path = load_path / 'fusion_attention.pt'
        if hasattr(self, 'fusion_attention') and fusion_attention_path.exists():
            self.fusion_attention.load_state_dict(torch.load(fusion_attention_path))
            
        transformer_path = load_path / 'cross_modal_transformer.pt'
        if hasattr(self, 'cross_modal_transformer') and transformer_path.exists():
            self.cross_modal_transformer.load_state_dict(torch.load(transformer_path))
            
        # Load modality weights
        weights_path = load_path / 'modality_weights.pt'
        if weights_path.exists():
            self.modality_weights = torch.load(weights_path)
            
        # Load metrics
        metrics_path = load_path / 'metrics.json'
        if metrics_path.exists():
            import json
            with open(metrics_path, 'r') as f:
                self.fusion_metrics = json.load(f)


class UncertaintyAwareFusion(nn.Module):
    """Enhanced fusion with uncertainty quantification and adaptive processing."""
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int, 
                 num_modalities: int,
                 dropout_samples: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_modalities = num_modalities
        self.dropout_samples = dropout_samples
        
        # Bayesian layers for uncertainty estimation
        self.bayesian_fusion = nn.Sequential(
            nn.Linear(input_dim * num_modalities, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),  # High dropout for Monte Carlo sampling
            nn.Linear(1024, 512),
            nn.ReLU(), 
            nn.Dropout(0.5),
            nn.Linear(512, output_dim)
        )
        
        # Uncertainty quantification head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(output_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output uncertainty in [0,1]
        )
        
        # Quality assessment per modality
        self.quality_assessors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            ) for _ in range(num_modalities)
        ])
        
    def forward(self, 
                modality_features: List[torch.Tensor], 
                estimate_uncertainty: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.
        
        Args:
            modality_features: List of tensors from each modality
            estimate_uncertainty: Whether to perform uncertainty estimation
            
        Returns:
            Dictionary with fused features, uncertainty, and quality scores
        """
        batch_size = modality_features[0].shape[0]
        
        # Assess quality of each modality
        quality_scores = []
        for i, features in enumerate(modality_features):
            quality = self.quality_assessors[i](features)
            quality_scores.append(quality)
            
        # Concatenate modality features
        concatenated = torch.cat(modality_features, dim=1)
        
        if estimate_uncertainty and self.training:
            # Monte Carlo Dropout for uncertainty estimation
            outputs = []
            self.bayesian_fusion.train()  # Ensure dropout is active
            
            for _ in range(self.dropout_samples):
                output = self.bayesian_fusion(concatenated)
                outputs.append(output)
                
            # Statistics from multiple forward passes
            outputs = torch.stack(outputs, dim=0)  # [samples, batch, features]
            fused_mean = outputs.mean(dim=0)
            fused_std = outputs.std(dim=0)
            
            # Estimate epistemic uncertainty
            epistemic_uncertainty = fused_std.mean(dim=1, keepdim=True)
            
        else:
            # Single forward pass
            fused_mean = self.bayesian_fusion(concatenated)
            epistemic_uncertainty = torch.zeros(batch_size, 1)
            
        # Estimate aleatoric uncertainty
        aleatoric_uncertainty = self.uncertainty_head(fused_mean)
        
        # Total uncertainty (epistemic + aleatoric)
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        return {
            'fused_features': fused_mean,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'total_uncertainty': total_uncertainty,
            'quality_scores': torch.cat(quality_scores, dim=1)
        }


class NeuromorphicProcessor:
    """Neuromorphic processing for event-driven multi-modal fusion."""
    
    def __init__(self, 
                 dimension: int = 10000,
                 refractory_period: float = 0.001,
                 threshold: float = 1.0,
                 leak_rate: float = 0.95):
        self.dimension = dimension
        self.refractory_period = refractory_period
        self.threshold = threshold
        self.leak_rate = leak_rate
        
        # Neuromorphic state
        self.membrane_potential = np.zeros(dimension)
        self.last_spike_time = np.zeros(dimension)
        self.refractory_state = np.zeros(dimension, dtype=bool)
        
        # STDP (Spike-Timing Dependent Plasticity) parameters
        self.stdp_tau_pre = 0.020  # 20ms
        self.stdp_tau_post = 0.020 # 20ms
        self.stdp_a_plus = 0.1
        self.stdp_a_minus = 0.05
        
        # Adaptive weights for different modalities
        self.synaptic_weights = {}
        
        # Event buffer for temporal processing
        self.event_buffer = deque(maxlen=10000)
        
        # Learning state
        self.homeostatic_target = 10.0  # Target firing rate (Hz)
        self.homeostatic_gain = np.ones(dimension)
        
        self.logger = logging.getLogger(__name__)
        
    def register_modality(self, modality_name: str, input_size: int):
        """Register a new modality with random initial weights."""
        self.synaptic_weights[modality_name] = np.random.normal(
            0.0, 0.1, size=(self.dimension, input_size)
        )
        
    def process_event(self, 
                     modality_name: str, 
                     event_data: np.ndarray,
                     timestamp: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process an event from a specific modality using neuromorphic principles.
        
        Args:
            modality_name: Name of the input modality
            event_data: Event data (spikes or analog values)
            timestamp: Event timestamp
            
        Returns:
            Tuple of (output_spikes, processing_info)
        """
        if modality_name not in self.synaptic_weights:
            raise ValueError(f"Modality {modality_name} not registered")
            
        # Convert input to spikes if necessary
        if event_data.dtype != bool:
            input_spikes = self._analog_to_spikes(event_data, timestamp)
        else:
            input_spikes = event_data.astype(bool)
            
        # Apply synaptic weights
        weighted_input = np.dot(self.synaptic_weights[modality_name], input_spikes)
        
        # Update membrane potential
        dt = 0.001  # 1ms time step
        self._update_membrane_potential(weighted_input, dt, timestamp)
        
        # Generate output spikes
        output_spikes = self._generate_spikes(timestamp)
        
        # Apply STDP learning
        if input_spikes.any() and output_spikes.any():
            self._apply_stdp(modality_name, input_spikes, output_spikes, timestamp)
            
        # Homeostatic regulation
        self._apply_homeostasis(timestamp)
        
        # Store event for temporal analysis
        self.event_buffer.append({
            'modality': modality_name,
            'timestamp': timestamp,
            'input_spikes': input_spikes.copy(),
            'output_spikes': output_spikes.copy(),
            'membrane_potential': self.membrane_potential.copy()
        })
        
        # Processing statistics
        processing_info = {
            'input_spike_rate': np.sum(input_spikes) / len(input_spikes),
            'output_spike_rate': np.sum(output_spikes) / len(output_spikes),
            'avg_membrane_potential': np.mean(self.membrane_potential),
            'active_neurons': np.sum(output_spikes),
            'refractory_neurons': np.sum(self.refractory_state)
        }
        
        return output_spikes, processing_info
        
    def _analog_to_spikes(self, analog_data: np.ndarray, timestamp: float) -> np.ndarray:
        """Convert analog data to spike train using rate coding."""
        # Normalize to [0, 1]
        normalized = (analog_data - np.min(analog_data)) / (np.max(analog_data) - np.min(analog_data) + 1e-8)
        
        # Rate coding: higher values = higher spike probability
        spike_probability = normalized * 0.1  # Max 10% spike probability per time step
        random_values = np.random.uniform(0, 1, size=analog_data.shape)
        
        return random_values < spike_probability
        
    def _update_membrane_potential(self, weighted_input: np.ndarray, dt: float, timestamp: float):
        """Update membrane potential using leaky integrate-and-fire dynamics."""
        # Leak
        self.membrane_potential *= self.leak_rate
        
        # Integration (only for non-refractory neurons)
        active_mask = ~self.refractory_state
        self.membrane_potential[active_mask] += weighted_input[active_mask] * self.homeostatic_gain[active_mask]
        
        # Update refractory state
        time_since_spike = timestamp - self.last_spike_time
        self.refractory_state = time_since_spike < self.refractory_period
        
    def _generate_spikes(self, timestamp: float) -> np.ndarray:
        """Generate output spikes based on membrane potential and threshold."""
        # Neurons spike when potential exceeds threshold and not in refractory period
        spike_mask = (self.membrane_potential >= self.threshold) & (~self.refractory_state)
        
        # Reset spiked neurons
        self.membrane_potential[spike_mask] = 0.0
        self.last_spike_time[spike_mask] = timestamp
        
        return spike_mask
        
    def _apply_stdp(self, 
                   modality_name: str, 
                   input_spikes: np.ndarray, 
                   output_spikes: np.ndarray, 
                   timestamp: float):
        """Apply Spike-Timing Dependent Plasticity learning rule."""
        # Find pre-synaptic (input) and post-synaptic (output) spike indices
        pre_indices = np.where(input_spikes)[0]
        post_indices = np.where(output_spikes)[0]
        
        if len(pre_indices) == 0 or len(post_indices) == 0:
            return
            
        # For simplicity, we'll update weights based on coincident activity
        for post_idx in post_indices:
            for pre_idx in pre_indices:
                # Potentiation (strengthen synapses for coincident activity)
                self.synaptic_weights[modality_name][post_idx, pre_idx] += self.stdp_a_plus
                
                # Bound weights
                self.synaptic_weights[modality_name][post_idx, pre_idx] = np.clip(
                    self.synaptic_weights[modality_name][post_idx, pre_idx], -1.0, 1.0
                )
                
    def _apply_homeostasis(self, timestamp: float):
        """Apply homeostatic regulation to maintain target firing rates."""
        # Simple homeostatic scaling
        # In practice, this would track firing rates over longer windows
        current_activity = np.mean(self.membrane_potential)
        
        if len(self.event_buffer) > 100:  # Need history for homeostasis
            recent_events = list(self.event_buffer)[-100:]
            recent_spike_rates = [event['output_spike_rate'] for event in recent_events]
            avg_spike_rate = np.mean(recent_spike_rates)
            
            # Adjust homeostatic gain
            if avg_spike_rate < self.homeostatic_target * 0.8:
                self.homeostatic_gain *= 1.001  # Slight increase
            elif avg_spike_rate > self.homeostatic_target * 1.2:
                self.homeostatic_gain *= 0.999  # Slight decrease
                
            # Bound gains
            self.homeostatic_gain = np.clip(self.homeostatic_gain, 0.1, 2.0)
            
    def get_temporal_patterns(self, window_size: int = 1000) -> Dict[str, Any]:
        """Extract temporal patterns from recent processing history."""
        if len(self.event_buffer) < window_size:
            window_size = len(self.event_buffer)
            
        if window_size == 0:
            return {}
            
        recent_events = list(self.event_buffer)[-window_size:]
        
        # Analyze patterns
        modality_activity = {}
        temporal_correlations = {}
        
        for event in recent_events:
            modality = event['modality']
            if modality not in modality_activity:
                modality_activity[modality] = []
            modality_activity[modality].append(event['output_spike_rate'])
            
        # Compute statistics
        patterns = {
            'modality_stats': {},
            'cross_modal_correlations': {},
            'temporal_dynamics': {
                'avg_membrane_potential': np.mean([e['membrane_potential'] for e in recent_events], axis=0),
                'spike_rate_variance': np.var([e['output_spike_rate'] for e in recent_events])
            }
        }
        
        for modality, rates in modality_activity.items():
            patterns['modality_stats'][modality] = {
                'mean_spike_rate': np.mean(rates),
                'std_spike_rate': np.std(rates),
                'spike_rate_trend': np.polyfit(range(len(rates)), rates, 1)[0] if len(rates) > 1 else 0
            }
            
        return patterns
        
    def reset_state(self):
        """Reset neuromorphic processor state."""
        self.membrane_potential.fill(0.0)
        self.last_spike_time.fill(0.0)
        self.refractory_state.fill(False)
        self.event_buffer.clear()
        
        self.logger.info("Neuromorphic processor state reset")


class AdaptiveMultiModalFusion:
    """
    Enhanced multi-modal fusion with uncertainty quantification, 
    neuromorphic processing, and adaptive learning capabilities.
    """
    
    def __init__(self,
                 modality_configs: List[ModalityConfig],
                 hdc_dimension: int = 10000,
                 enable_neuromorphic: bool = True,
                 enable_uncertainty: bool = True,
                 learning_rate: float = 1e-4):
        
        self.modality_configs = {cfg.name: cfg for cfg in modality_configs}
        self.hdc_dimension = hdc_dimension
        self.enable_neuromorphic = enable_neuromorphic
        self.enable_uncertainty = enable_uncertainty
        
        # Initialize base fusion engine
        self.base_fusion = MultiModalFusionEngine(
            modality_configs, hdc_dimension, learning_rate=learning_rate
        )
        
        # Enhanced uncertainty-aware fusion
        if enable_uncertainty:
            self.uncertainty_fusion = UncertaintyAwareFusion(
                input_dim=512,  # Assuming 512-dim features from encoders
                output_dim=hdc_dimension,
                num_modalities=len(modality_configs)
            )
            
        # Neuromorphic processor
        if enable_neuromorphic:
            self.neuromorphic = NeuromorphicProcessor(hdc_dimension)
            for config in modality_configs:
                if config.neuromorphic_enabled:
                    self.neuromorphic.register_modality(config.name, config.dimension)
                    
        # Temporal buffers for each modality
        self.temporal_buffers = {
            cfg.name: deque(maxlen=int(cfg.temporal_window * 100))  # 100Hz sampling
            for cfg in modality_configs
        }
        
        # Adaptive quality monitoring
        self.quality_history = {cfg.name: deque(maxlen=1000) for cfg in modality_configs}
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=len(modality_configs))
        
        # Performance monitoring
        self.processing_stats = {
            'fusion_count': 0,
            'avg_uncertainty': 0.0,
            'quality_trends': {},
            'processing_times': deque(maxlen=1000)
        }
        
        self.logger = logging.getLogger(__name__)
        
    def process_multi_modal_data(self,
                                modality_data: Dict[str, np.ndarray],
                                timestamp: Optional[float] = None,
                                context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process multi-modal data with advanced fusion capabilities.
        
        Args:
            modality_data: Dictionary of modality name to data arrays
            timestamp: Optional timestamp for neuromorphic processing
            context: Additional context information
            
        Returns:
            Comprehensive fusion results with uncertainty and quality metrics
        """
        start_time = time.time()
        
        if timestamp is None:
            timestamp = start_time
            
        # Filter modalities based on quality thresholds
        filtered_data = self._filter_by_quality(modality_data)
        
        # Update temporal buffers
        self._update_temporal_buffers(filtered_data, timestamp)
        
        # Base fusion processing
        base_results = self.base_fusion.fuse_modalities(
            filtered_data, context, adaptive_weighting=True
        )
        
        enhanced_results = {
            'base_fusion': base_results,
            'timestamp': timestamp,
            'processing_time': 0.0,
            'quality_assessment': {},
            'uncertainty_analysis': {},
            'neuromorphic_analysis': {},
            'temporal_patterns': {}
        }
        
        # Enhanced uncertainty estimation
        if self.enable_uncertainty:
            uncertainty_results = self._estimate_uncertainty(
                modality_data, base_results['neural_features']
            )
            enhanced_results['uncertainty_analysis'] = uncertainty_results
            
        # Neuromorphic processing
        if self.enable_neuromorphic:
            neuromorphic_results = self._process_neuromorphic(
                filtered_data, timestamp
            )
            enhanced_results['neuromorphic_analysis'] = neuromorphic_results
            
        # Temporal pattern analysis
        temporal_results = self._analyze_temporal_patterns()
        enhanced_results['temporal_patterns'] = temporal_results
        
        # Quality assessment and adaptation
        quality_results = self._assess_and_adapt_quality(modality_data, enhanced_results)
        enhanced_results['quality_assessment'] = quality_results
        
        # Update performance statistics
        processing_time = time.time() - start_time
        enhanced_results['processing_time'] = processing_time
        self._update_performance_stats(enhanced_results, processing_time)
        
        return enhanced_results
        
    def _filter_by_quality(self, modality_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Filter modalities based on quality thresholds."""
        filtered = {}
        
        for modality_name, data in modality_data.items():
            config = self.modality_configs.get(modality_name)
            if not config:
                continue
                
            # Simple quality metrics
            data_quality = self._compute_data_quality(data)
            
            if data_quality >= config.quality_threshold:
                filtered[modality_name] = data
                
            # Update quality history
            self.quality_history[modality_name].append(data_quality)
            
        return filtered
        
    def _compute_data_quality(self, data: np.ndarray) -> float:
        """Compute quality score for data array."""
        # Simple quality metrics
        if data.size == 0:
            return 0.0
            
        # Check for NaN/Inf values
        if np.any(~np.isfinite(data)):
            return 0.0
            
        # Signal-to-noise ratio estimation
        signal_power = np.var(data)
        noise_estimate = np.var(np.diff(data.flatten())) / 2  # Rough noise estimate
        
        if noise_estimate == 0:
            snr = float('inf')
        else:
            snr = signal_power / noise_estimate
            
        # Normalize SNR to [0,1] quality score
        quality = min(1.0, snr / 10.0)  # Assume SNR > 10 is good quality
        
        return quality
        
    def _update_temporal_buffers(self, modality_data: Dict[str, np.ndarray], timestamp: float):
        """Update temporal buffers with new data."""
        for modality_name, data in modality_data.items():
            if modality_name in self.temporal_buffers:
                self.temporal_buffers[modality_name].append({
                    'data': data,
                    'timestamp': timestamp
                })
                
    def _estimate_uncertainty(self, 
                            modality_data: Dict[str, np.ndarray],
                            neural_features: torch.Tensor) -> Dict[str, Any]:
        """Enhanced uncertainty estimation."""
        if not self.enable_uncertainty:
            return {}
            
        # Prepare features for uncertainty estimation
        feature_list = []
        for modality_name in modality_data.keys():
            if modality_name in self.base_fusion.encoders:
                # Get neural features from base fusion encoders
                with torch.no_grad():
                    data_tensor = torch.from_numpy(modality_data[modality_name]).float()
                    if data_tensor.ndim == 2:
                        data_tensor = data_tensor.unsqueeze(0)
                    elif data_tensor.ndim == 1:
                        data_tensor = data_tensor.unsqueeze(0).unsqueeze(0)
                        
                    features, _ = self.base_fusion.encoders[modality_name](data_tensor)
                    feature_list.append(features)
                    
        if not feature_list:
            return {}
            
        # Run uncertainty-aware fusion
        uncertainty_results = self.uncertainty_fusion(feature_list, estimate_uncertainty=True)
        
        return {
            'epistemic_uncertainty': uncertainty_results['epistemic_uncertainty'].mean().item(),
            'aleatoric_uncertainty': uncertainty_results['aleatoric_uncertainty'].mean().item(),
            'total_uncertainty': uncertainty_results['total_uncertainty'].mean().item(),
            'quality_scores': uncertainty_results['quality_scores'].cpu().numpy(),
            'confidence': 1.0 - uncertainty_results['total_uncertainty'].mean().item()
        }
        
    def _process_neuromorphic(self,
                            modality_data: Dict[str, np.ndarray], 
                            timestamp: float) -> Dict[str, Any]:
        """Process data through neuromorphic processor."""
        if not self.enable_neuromorphic:
            return {}
            
        neuromorphic_results = {}
        
        for modality_name, data in modality_data.items():
            config = self.modality_configs.get(modality_name)
            if config and config.neuromorphic_enabled:
                try:
                    spikes, info = self.neuromorphic.process_event(
                        modality_name, data, timestamp
                    )
                    neuromorphic_results[modality_name] = {
                        'output_spikes': spikes,
                        'processing_info': info
                    }
                except Exception as e:
                    self.logger.error(f"Neuromorphic processing error for {modality_name}: {e}")
                    
        # Get temporal patterns
        if neuromorphic_results:
            temporal_patterns = self.neuromorphic.get_temporal_patterns()
            neuromorphic_results['temporal_patterns'] = temporal_patterns
            
        return neuromorphic_results
        
    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns across modalities."""
        patterns = {}
        
        for modality_name, buffer in self.temporal_buffers.items():
            if len(buffer) < 10:  # Need minimum history
                continue
                
            timestamps = [entry['timestamp'] for entry in buffer]
            
            # Compute temporal statistics
            if len(timestamps) > 1:
                intervals = np.diff(timestamps)
                patterns[modality_name] = {
                    'avg_interval': np.mean(intervals),
                    'interval_std': np.std(intervals),
                    'data_rate': len(buffer) / (timestamps[-1] - timestamps[0] + 1e-8),
                    'temporal_consistency': 1.0 / (1.0 + np.std(intervals))
                }
                
        return patterns
        
    def _assess_and_adapt_quality(self,
                                original_data: Dict[str, np.ndarray],
                                fusion_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess quality and adapt processing parameters."""
        quality_assessment = {}
        
        for modality_name in original_data.keys():
            history = list(self.quality_history[modality_name])
            
            if len(history) < 10:
                continue
                
            recent_quality = np.mean(history[-10:])  # Last 10 samples
            overall_quality = np.mean(history)
            quality_trend = np.polyfit(range(len(history)), history, 1)[0]
            
            quality_assessment[modality_name] = {
                'recent_quality': recent_quality,
                'overall_quality': overall_quality,
                'quality_trend': quality_trend,
                'quality_stability': 1.0 - np.std(history[-10:])
            }
            
            # Adaptive parameter adjustment
            config = self.modality_configs[modality_name]
            
            # Lower threshold if quality is consistently poor
            if recent_quality < 0.5 and quality_trend < 0:
                config.quality_threshold *= 0.95  # Slightly lower threshold
            elif recent_quality > 0.8 and quality_trend > 0:
                config.quality_threshold *= 1.05  # Raise threshold
                
            # Bound thresholds
            config.quality_threshold = np.clip(config.quality_threshold, 0.1, 0.9)
            
        return quality_assessment
        
    def _update_performance_stats(self, results: Dict[str, Any], processing_time: float):
        """Update performance statistics."""
        self.processing_stats['fusion_count'] += 1
        self.processing_stats['processing_times'].append(processing_time)
        
        # Update uncertainty tracking
        if 'uncertainty_analysis' in results and results['uncertainty_analysis']:
            uncertainty = results['uncertainty_analysis'].get('total_uncertainty', 0.0)
            count = self.processing_stats['fusion_count']
            current_avg = self.processing_stats['avg_uncertainty']
            self.processing_stats['avg_uncertainty'] = (
                (current_avg * (count - 1) + uncertainty) / count
            )
            
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status and performance metrics."""
        status = {
            'processing_stats': dict(self.processing_stats),
            'modality_status': {},
            'system_health': {},
            'recommendations': []
        }
        
        # Processing time statistics
        if self.processing_stats['processing_times']:
            times = list(self.processing_stats['processing_times'])
            status['processing_stats'].update({
                'avg_processing_time': np.mean(times),
                'std_processing_time': np.std(times),
                'max_processing_time': np.max(times),
                'min_processing_time': np.min(times)
            })
            
        # Modality status
        for modality_name, config in self.modality_configs.items():
            history = list(self.quality_history[modality_name])
            
            if history:
                status['modality_status'][modality_name] = {
                    'current_threshold': config.quality_threshold,
                    'avg_quality': np.mean(history),
                    'recent_quality': np.mean(history[-10:]) if len(history) >= 10 else np.mean(history),
                    'quality_trend': np.polyfit(range(len(history)), history, 1)[0] if len(history) > 1 else 0,
                    'samples_processed': len(history),
                    'neuromorphic_enabled': config.neuromorphic_enabled
                }
                
        # System health assessment
        avg_processing_time = status['processing_stats'].get('avg_processing_time', 0)
        avg_uncertainty = self.processing_stats['avg_uncertainty']
        
        health_score = 1.0
        if avg_processing_time > 0.1:  # Slow processing
            health_score -= 0.3
        if avg_uncertainty > 0.7:  # High uncertainty
            health_score -= 0.3
            
        status['system_health'] = {
            'overall_score': max(0.0, health_score),
            'processing_performance': 'good' if avg_processing_time < 0.05 else 'degraded',
            'fusion_reliability': 'high' if avg_uncertainty < 0.3 else 'moderate' if avg_uncertainty < 0.7 else 'low'
        }
        
        # Recommendations
        if avg_processing_time > 0.1:
            status['recommendations'].append("Consider reducing modality processing complexity")
        if avg_uncertainty > 0.7:
            status['recommendations'].append("Review data quality and sensor calibration")
            
        return status