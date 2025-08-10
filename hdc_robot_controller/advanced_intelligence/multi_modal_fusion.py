"""
Advanced Multi-Modal Fusion Engine with Transformer Architecture

Combines HDC with state-of-the-art transformer models for superior
multi-modal understanding and reasoning capabilities.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time
from pathlib import Path

from ..core.hypervector import HyperVector
from ..core.operations import HDCOperations


@dataclass
class ModalityConfig:
    """Configuration for each modality."""
    name: str
    dimension: int
    encoder_type: str  # 'transformer', 'cnn', 'rnn', 'hdc'
    attention_heads: int = 8
    hidden_dim: int = 512
    dropout: float = 0.1
    

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
            raise ValueError(f"Unknown modality: {modality_name}")\n            
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