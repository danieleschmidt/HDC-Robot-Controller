"""
Neural-HDC Hybrid Architecture

Seamlessly integrates neural networks with hyperdimensional computing
for enhanced learning, reasoning, and adaptation capabilities.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import time
from pathlib import Path

from ..core.hypervector import HyperVector
from ..core.operations import HDCOperations


@dataclass
class HybridConfig:
    """Configuration for Neural-HDC hybrid architecture."""
    neural_hidden_dim: int = 512
    hdc_dimension: int = 10000
    fusion_strategy: str = "cross_attention"  # "concat", "cross_attention", "gating"
    learning_rate: float = 1e-4
    hdc_learning_rate: float = 0.01
    enable_bidirectional: bool = True
    enable_adaptive_fusion: bool = True
    dropout_rate: float = 0.1


class NeuralToHDCBridge(nn.Module):
    """Bridge module to convert neural representations to HDC."""
    
    def __init__(self, neural_dim: int, hdc_dim: int, bridge_type: str = "learned"):
        super().__init__()
        self.neural_dim = neural_dim
        self.hdc_dim = hdc_dim
        self.bridge_type = bridge_type
        
        if bridge_type == "learned":
            # Learnable projection matrix
            self.projection = nn.Sequential(
                nn.Linear(neural_dim, hdc_dim),
                nn.Tanh(),
                nn.Dropout(0.1)
            )
            
        elif bridge_type == "random_fixed":
            # Fixed random projection (Johnson-Lindenstrauss)
            self.register_buffer('random_matrix', 
                               torch.randn(neural_dim, hdc_dim) / np.sqrt(neural_dim))
            
        elif bridge_type == "hdc_encoder":
            # HDC-based encoding
            self.hdc_encoders = nn.ModuleDict({
                f'encoder_{i}': nn.Linear(1, hdc_dim // neural_dim)
                for i in range(neural_dim)
            })
            
    def forward(self, neural_features: torch.Tensor) -> torch.Tensor:
        """Convert neural features to HDC representation."""
        batch_size = neural_features.shape[0]
        
        if self.bridge_type == "learned":
            hdc_features = self.projection(neural_features)
            
        elif self.bridge_type == "random_fixed":
            hdc_features = torch.matmul(neural_features, self.random_matrix)
            
        elif self.bridge_type == "hdc_encoder":
            hdc_parts = []
            for i in range(self.neural_dim):
                feature_val = neural_features[:, i:i+1]
                encoded = self.hdc_encoders[f'encoder_{i}'](feature_val)
                hdc_parts.append(encoded)
            hdc_features = torch.cat(hdc_parts, dim=1)
            
        # Apply bipolar activation
        return torch.sign(hdc_features)


class HDCToNeuralBridge(nn.Module):
    """Bridge module to convert HDC representations to neural features."""
    
    def __init__(self, hdc_dim: int, neural_dim: int, bridge_type: str = "learned"):
        super().__init__()
        self.hdc_dim = hdc_dim
        self.neural_dim = neural_dim
        self.bridge_type = bridge_type
        
        if bridge_type == "learned":
            # Learnable projection with attention
            self.attention = nn.MultiheadAttention(
                embed_dim=hdc_dim,
                num_heads=16,
                batch_first=True
            )
            
            self.projection = nn.Sequential(
                nn.Linear(hdc_dim, neural_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(neural_dim * 2, neural_dim)
            )
            
        elif bridge_type == "statistical":
            # Statistical feature extraction from HDC
            self.feature_extractors = nn.ModuleDict({
                'sparsity': nn.Linear(1, neural_dim // 4),
                'entropy': nn.Linear(1, neural_dim // 4),
                'clustering': nn.Linear(10, neural_dim // 4),  # Top-10 clusters
                'similarity': nn.Linear(1, neural_dim // 4)
            })
            
    def forward(self, hdc_tensor: torch.Tensor) -> torch.Tensor:
        """Convert HDC representation to neural features."""
        batch_size = hdc_tensor.shape[0]
        
        if self.bridge_type == "learned":
            # Apply attention to HDC features
            hdc_attended, _ = self.attention(
                hdc_tensor.unsqueeze(1),
                hdc_tensor.unsqueeze(1), 
                hdc_tensor.unsqueeze(1)
            )
            hdc_pooled = hdc_attended.squeeze(1)
            
            # Project to neural dimension
            neural_features = self.projection(hdc_pooled)
            
        elif self.bridge_type == "statistical":
            # Extract statistical features
            neural_parts = []
            
            # Sparsity
            sparsity = torch.mean((hdc_tensor > 0).float(), dim=1, keepdim=True)
            sparsity_feat = self.feature_extractors['sparsity'](sparsity)
            neural_parts.append(sparsity_feat)
            
            # Entropy (approximation)
            p_pos = torch.clamp(sparsity, 1e-8, 1-1e-8)
            p_neg = 1 - p_pos
            entropy = -(p_pos * torch.log(p_pos) + p_neg * torch.log(p_neg))
            entropy_feat = self.feature_extractors['entropy'](entropy)
            neural_parts.append(entropy_feat)
            
            # Clustering features (simplified)
            clustered = F.adaptive_avg_pool1d(
                hdc_tensor.unsqueeze(1),
                output_size=10
            ).squeeze(1)
            cluster_feat = self.feature_extractors['clustering'](clustered)
            neural_parts.append(cluster_feat)
            
            # Self-similarity
            self_sim = torch.cosine_similarity(
                hdc_tensor, 
                torch.roll(hdc_tensor, 1, dims=1),
                dim=1
            ).unsqueeze(1)
            sim_feat = self.feature_extractors['similarity'](self_sim)
            neural_parts.append(sim_feat)
            
            neural_features = torch.cat(neural_parts, dim=1)
            
        return neural_features


class CrossAttentionFusion(nn.Module):
    """Cross-attention mechanism for neural-HDC fusion."""
    
    def __init__(self, neural_dim: int, hdc_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.neural_dim = neural_dim
        self.hdc_dim = hdc_dim
        self.hidden_dim = hidden_dim
        
        # Project both modalities to common dimension
        self.neural_projector = nn.Linear(neural_dim, hidden_dim)
        self.hdc_projector = nn.Linear(hdc_dim, hidden_dim)
        
        # Cross-attention layers
        self.neural_to_hdc_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        self.hdc_to_neural_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Fusion layers
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, neural_features: torch.Tensor, hdc_features: torch.Tensor) -> torch.Tensor:
        """Perform cross-attention fusion."""
        batch_size = neural_features.shape[0]
        
        # Project to common space
        neural_proj = self.neural_projector(neural_features).unsqueeze(1)  # [B, 1, H]
        hdc_proj = self.hdc_projector(hdc_features).unsqueeze(1)  # [B, 1, H]
        
        # Cross-attention
        neural_attended, _ = self.neural_to_hdc_attention(
            neural_proj, hdc_proj, hdc_proj
        )
        
        hdc_attended, _ = self.hdc_to_neural_attention(
            hdc_proj, neural_proj, neural_proj
        )
        
        # Combine attended features
        fused = torch.cat([
            neural_attended.squeeze(1),
            hdc_attended.squeeze(1)
        ], dim=1)
        
        # Final fusion
        output = self.fusion_mlp(fused)
        
        return output


class AdaptiveGating(nn.Module):
    """Adaptive gating mechanism for neural-HDC fusion."""
    
    def __init__(self, neural_dim: int, hdc_dim: int):
        super().__init__()
        self.neural_dim = neural_dim
        self.hdc_dim = hdc_dim
        
        # Gating networks
        self.neural_gate = nn.Sequential(
            nn.Linear(neural_dim + hdc_dim, 128),
            nn.ReLU(),
            nn.Linear(128, neural_dim),
            nn.Sigmoid()
        )
        
        self.hdc_gate = nn.Sequential(
            nn.Linear(neural_dim + hdc_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hdc_dim),
            nn.Sigmoid()
        )
        
        # Combination weights
        self.combination_weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        
    def forward(self, neural_features: torch.Tensor, hdc_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply adaptive gating."""
        # Concatenate features for gate computation
        combined = torch.cat([neural_features, hdc_features], dim=1)
        
        # Compute gates
        neural_gate_weights = self.neural_gate(combined)
        hdc_gate_weights = self.hdc_gate(combined)
        
        # Apply gates
        gated_neural = neural_features * neural_gate_weights
        gated_hdc = hdc_features * hdc_gate_weights
        
        # Compute combination weights
        weights = F.softmax(self.combination_weights, dim=0)
        
        return {
            'gated_neural': gated_neural,
            'gated_hdc': gated_hdc,
            'neural_gate_weights': neural_gate_weights,
            'hdc_gate_weights': hdc_gate_weights,
            'combination_weights': weights
        }


class NeuralHDCHybrid(nn.Module):
    """
    Hybrid Neural-HDC Architecture.
    
    Combines the representational power of neural networks with the
    efficiency and interpretability of hyperdimensional computing.
    """
    
    def __init__(self, config: HybridConfig):
        super().__init__()
        self.config = config
        
        # Neural network component
        self.neural_encoder = nn.Sequential(
            nn.Linear(config.neural_hidden_dim, config.neural_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.neural_hidden_dim, config.neural_hidden_dim)
        )
        
        # HDC component
        self.hdc_ops = HDCOperations()
        
        # Bridge modules
        self.neural_to_hdc = NeuralToHDCBridge(
            config.neural_hidden_dim, 
            config.hdc_dimension,
            bridge_type="learned"
        )
        
        if config.enable_bidirectional:
            self.hdc_to_neural = HDCToNeuralBridge(
                config.hdc_dimension,
                config.neural_hidden_dim,
                bridge_type="learned"
            )
        
        # Fusion mechanism
        if config.fusion_strategy == "cross_attention":
            self.fusion = CrossAttentionFusion(
                config.neural_hidden_dim,
                config.hdc_dimension
            )
        elif config.fusion_strategy == "gating" and config.enable_adaptive_fusion:
            self.fusion = AdaptiveGating(
                config.neural_hidden_dim,
                config.hdc_dimension
            )
        
        # Output layers
        self.output_projector = nn.Sequential(
            nn.Linear(512, config.neural_hidden_dim),  # Assuming fusion output is 512
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # HDC memory for associative operations
        self.hdc_memory = {}
        self.neural_memory = {}
        
        # Performance tracking
        self.hybrid_metrics = {
            'neural_forward_passes': 0,
            'hdc_operations': 0,
            'fusion_operations': 0,
            'memory_retrievals': 0,
            'adaptation_updates': 0
        }
        
    def forward(self, input_data: torch.Tensor, mode: str = "hybrid") -> Dict[str, torch.Tensor]:
        """
        Forward pass with multiple operation modes.
        
        Args:
            input_data: Input tensor
            mode: Operation mode ("neural", "hdc", "hybrid")
            
        Returns:
            Dictionary with outputs from different components
        """
        batch_size = input_data.shape[0]
        
        outputs = {}
        
        if mode in ["neural", "hybrid"]:
            # Neural processing
            neural_features = self.neural_encoder(input_data)
            outputs['neural_features'] = neural_features
            self.hybrid_metrics['neural_forward_passes'] += 1
            
        if mode in ["hdc", "hybrid"]:
            # Convert to HDC
            if mode == "hdc":
                # Direct HDC processing
                hdc_tensor = self.neural_to_hdc(input_data)
            else:
                # Use neural features for HDC conversion
                hdc_tensor = self.neural_to_hdc(neural_features)
                
            outputs['hdc_tensor'] = hdc_tensor
            
            # Convert to HyperVector for HDC operations
            hdc_vectors = []
            for i in range(batch_size):
                hv_data = hdc_tensor[i].detach().cpu().numpy().astype(np.int8)
                hv = HyperVector(self.config.hdc_dimension, hv_data)
                hdc_vectors.append(hv)
                
            outputs['hdc_vectors'] = hdc_vectors
            self.hybrid_metrics['hdc_operations'] += 1
            
        if mode == "hybrid":
            # Bidirectional processing
            if self.config.enable_bidirectional:
                # Convert HDC back to neural
                neural_from_hdc = self.hdc_to_neural(hdc_tensor)
                outputs['neural_from_hdc'] = neural_from_hdc
                
            # Fusion
            if self.config.fusion_strategy == "concat":
                if self.config.enable_bidirectional:
                    fused = torch.cat([neural_features, neural_from_hdc], dim=1)
                else:
                    # Downsample HDC for concatenation
                    hdc_downsampled = F.adaptive_avg_pool1d(
                        hdc_tensor.unsqueeze(1),
                        output_size=self.config.neural_hidden_dim
                    ).squeeze(1)
                    fused = torch.cat([neural_features, hdc_downsampled], dim=1)
                    
            elif self.config.fusion_strategy == "cross_attention":
                fused = self.fusion(neural_features, hdc_tensor)
                
            elif self.config.fusion_strategy == "gating":
                gating_results = self.fusion(neural_features, hdc_tensor)
                # Combine gated features
                weights = gating_results['combination_weights']
                fused = (weights[0] * gating_results['gated_neural'] + 
                        weights[1] * gating_results['gated_hdc'])
                outputs['gating_info'] = gating_results
                
            outputs['fused_features'] = fused
            
            # Final output
            final_output = self.output_projector(fused)
            outputs['final_output'] = final_output
            
            self.hybrid_metrics['fusion_operations'] += 1
            
        return outputs
        
    def learn_association(self, 
                         input_data: torch.Tensor, 
                         target_data: torch.Tensor,
                         association_name: str,
                         learning_mode: str = "both") -> Dict[str, Any]:
        """
        Learn associations in both neural and HDC domains.
        
        Args:
            input_data: Input patterns
            target_data: Target patterns  
            association_name: Name for the association
            learning_mode: "neural", "hdc", or "both"
            
        Returns:
            Learning statistics and stored associations
        """
        learning_stats = {'neural': {}, 'hdc': {}}
        
        if learning_mode in ["neural", "both"]:
            # Neural association learning
            with torch.no_grad():
                input_features = self.neural_encoder(input_data)
                
                # Store in neural memory
                self.neural_memory[association_name] = {
                    'input_features': input_features.clone(),
                    'target_data': target_data.clone(),
                    'creation_time': time.time()
                }
                
                learning_stats['neural'] = {
                    'stored_patterns': input_features.shape[0],
                    'feature_dimension': input_features.shape[1]
                }
                
        if learning_mode in ["hdc", "both"]:
            # HDC association learning
            with torch.no_grad():
                # Convert inputs to HDC
                hdc_tensors = self.neural_to_hdc(input_data)
                
                # Create HyperVectors
                input_hvs = []
                target_hvs = []
                
                for i in range(hdc_tensors.shape[0]):
                    input_hv_data = hdc_tensors[i].detach().cpu().numpy().astype(np.int8)
                    input_hv = HyperVector(self.config.hdc_dimension, input_hv_data)
                    input_hvs.append(input_hv)
                    
                    # Convert target to HDC as well
                    target_hdc = self.neural_to_hdc(target_data[i:i+1])
                    target_hv_data = target_hdc[0].detach().cpu().numpy().astype(np.int8)
                    target_hv = HyperVector(self.config.hdc_dimension, target_hv_data)
                    target_hvs.append(target_hv)
                    
                # Create association using binding
                association_hv = HyperVector.zero(self.config.hdc_dimension)
                for input_hv, target_hv in zip(input_hvs, target_hvs):
                    bound_pair = input_hv.bind(target_hv)
                    association_hv = association_hv.bundle(bound_pair)
                    
                # Store HDC association
                self.hdc_memory[association_name] = {
                    'association_hv': association_hv,
                    'input_hvs': input_hvs,
                    'target_hvs': target_hvs,
                    'creation_time': time.time()
                }
                
                learning_stats['hdc'] = {
                    'stored_patterns': len(input_hvs),
                    'hdc_dimension': self.config.hdc_dimension,
                    'association_sparsity': association_hv.sparsity(),
                    'association_entropy': association_hv.entropy()
                }
                
        self.hybrid_metrics['adaptation_updates'] += 1
        
        return learning_stats
        
    def retrieve_association(self, 
                           query_data: torch.Tensor, 
                           association_name: str,
                           retrieval_mode: str = "both",
                           top_k: int = 3) -> Dict[str, Any]:
        """
        Retrieve associations using both neural and HDC mechanisms.
        
        Args:
            query_data: Query pattern
            association_name: Name of association to query
            retrieval_mode: "neural", "hdc", or "both"
            top_k: Number of top matches to return
            
        Returns:
            Retrieved associations and similarity scores
        """
        retrieval_results = {'neural': {}, 'hdc': {}}
        
        if retrieval_mode in ["neural", "both"] and association_name in self.neural_memory:
            # Neural retrieval
            with torch.no_grad():
                query_features = self.neural_encoder(query_data)
                stored_data = self.neural_memory[association_name]
                
                # Compute similarities
                stored_features = stored_data['input_features']
                similarities = F.cosine_similarity(
                    query_features.unsqueeze(1),
                    stored_features.unsqueeze(0),
                    dim=2
                )
                
                # Get top-k matches
                top_similarities, top_indices = torch.topk(similarities, 
                                                         min(top_k, similarities.shape[1]),
                                                         dim=1)
                
                retrieval_results['neural'] = {
                    'similarities': top_similarities,
                    'indices': top_indices,
                    'retrieved_targets': stored_data['target_data'][top_indices[0]]
                }
                
        if retrieval_mode in ["hdc", "both"] and association_name in self.hdc_memory:
            # HDC retrieval
            with torch.no_grad():
                # Convert query to HDC
                query_hdc = self.neural_to_hdc(query_data)
                query_hv_data = query_hdc[0].detach().cpu().numpy().astype(np.int8)
                query_hv = HyperVector(self.config.hdc_dimension, query_hv_data)
                
                stored_data = self.hdc_memory[association_name]
                association_hv = stored_data['association_hv']
                
                # Unbind to retrieve targets
                unbound = query_hv.bind(association_hv)  # Should be similar to targets
                
                # Compare with stored targets
                similarities = []
                for i, target_hv in enumerate(stored_data['target_hvs']):
                    sim = unbound.similarity(target_hv)
                    similarities.append((sim, i, target_hv))
                    
                # Sort by similarity
                similarities.sort(key=lambda x: abs(x[0]), reverse=True)
                top_similarities = similarities[:top_k]
                
                retrieval_results['hdc'] = {
                    'similarities': [s[0] for s in top_similarities],
                    'indices': [s[1] for s in top_similarities],
                    'retrieved_hvs': [s[2] for s in top_similarities],
                    'unbound_hv': unbound
                }
                
        self.hybrid_metrics['memory_retrievals'] += 1
        
        return retrieval_results
        
    def adapt_online(self, 
                    input_data: torch.Tensor, 
                    target_data: torch.Tensor,
                    adaptation_rate: float = 0.01) -> Dict[str, float]:
        """
        Perform online adaptation using both neural and HDC mechanisms.
        
        Args:
            input_data: New input pattern
            target_data: New target pattern
            adaptation_rate: Rate of adaptation
            
        Returns:
            Adaptation statistics
        """
        adaptation_stats = {}
        
        # Neural adaptation (gradient-based)
        self.train()
        outputs = self.forward(input_data, mode="hybrid")
        
        # Compute loss (assuming regression for now)
        loss = F.mse_loss(outputs['final_output'], target_data)
        
        # Gradient step
        loss.backward()
        
        # Apply gradients with adaptation rate
        with torch.no_grad():
            for param in self.parameters():
                if param.grad is not None:
                    param.data -= adaptation_rate * param.grad
                    param.grad.zero_()
                    
        adaptation_stats['neural_loss'] = loss.item()
        
        # HDC adaptation (associative)
        with torch.no_grad():
            # Convert to HDC
            input_hdc = self.neural_to_hdc(input_data)
            target_hdc = self.neural_to_hdc(target_data)
            
            # Create association
            input_hv_data = input_hdc[0].detach().cpu().numpy().astype(np.int8)
            target_hv_data = target_hdc[0].detach().cpu().numpy().astype(np.int8)
            
            input_hv = HyperVector(self.config.hdc_dimension, input_hv_data)
            target_hv = HyperVector(self.config.hdc_dimension, target_hv_data)
            
            # Bind input-target association
            new_association = input_hv.bind(target_hv)
            
            # Update global HDC memory (if exists)
            if 'global_adaptation' in self.hdc_memory:
                current_association = self.hdc_memory['global_adaptation']['association_hv']
                updated_association = current_association.bundle(new_association)
                self.hdc_memory['global_adaptation']['association_hv'] = updated_association
            else:
                self.hdc_memory['global_adaptation'] = {
                    'association_hv': new_association,
                    'update_count': 1,
                    'creation_time': time.time()
                }
                
            adaptation_stats['hdc_sparsity'] = new_association.sparsity()
            adaptation_stats['hdc_entropy'] = new_association.entropy()
            
        self.eval()
        self.hybrid_metrics['adaptation_updates'] += 1
        
        return adaptation_stats
        
    def get_interpretability_analysis(self) -> Dict[str, Any]:
        """
        Analyze the hybrid model for interpretability.
        
        Returns:
            Comprehensive interpretability analysis
        """
        analysis = {
            'neural_analysis': {},
            'hdc_analysis': {},
            'fusion_analysis': {},
            'memory_analysis': {}
        }
        
        # Neural component analysis
        with torch.no_grad():
            # Analyze neural weights
            neural_weights = []
            for name, param in self.neural_encoder.named_parameters():
                if 'weight' in name:
                    weight_stats = {
                        'mean': param.mean().item(),
                        'std': param.std().item(),
                        'min': param.min().item(),
                        'max': param.max().item(),
                        'sparsity': (param.abs() < 1e-6).float().mean().item()
                    }
                    neural_weights.append((name, weight_stats))
                    
            analysis['neural_analysis']['weight_statistics'] = neural_weights
            
            # Bridge analysis
            if hasattr(self.neural_to_hdc, 'projection'):
                bridge_weight = self.neural_to_hdc.projection[0].weight
                analysis['neural_analysis']['bridge_statistics'] = {
                    'effective_rank': torch.linalg.matrix_rank(bridge_weight).item(),
                    'condition_number': torch.linalg.cond(bridge_weight).item(),
                    'weight_distribution': {
                        'mean': bridge_weight.mean().item(),
                        'std': bridge_weight.std().item()
                    }
                }
                
        # HDC analysis
        if self.hdc_memory:
            hdc_stats = {}
            for name, data in self.hdc_memory.items():
                if 'association_hv' in data:
                    hv = data['association_hv']
                    hdc_stats[name] = {
                        'sparsity': hv.sparsity(),
                        'entropy': hv.entropy(),
                        'active_dimensions': len(hv.get_active_dimensions()),
                        'age_seconds': time.time() - data['creation_time']
                    }
                    
            analysis['hdc_analysis']['memory_statistics'] = hdc_stats
            
            # Analyze HDC similarity structure
            if len(self.hdc_memory) > 1:
                similarities = {}
                memory_items = list(self.hdc_memory.items())
                for i, (name1, data1) in enumerate(memory_items):
                    for name2, data2 in memory_items[i+1:]:
                        if 'association_hv' in data1 and 'association_hv' in data2:
                            sim = data1['association_hv'].similarity(data2['association_hv'])
                            similarities[f'{name1}_{name2}'] = sim
                            
                analysis['hdc_analysis']['inter_memory_similarities'] = similarities
                
        # Fusion analysis
        if hasattr(self, 'fusion'):
            if hasattr(self.fusion, 'combination_weights'):
                weights = self.fusion.combination_weights.detach().cpu()
                analysis['fusion_analysis']['combination_weights'] = {
                    'neural_weight': weights[0].item(),
                    'hdc_weight': weights[1].item(),
                    'balance_ratio': (weights[0] / weights[1]).item()
                }
                
        # Memory analysis
        analysis['memory_analysis'] = {
            'neural_memory_size': len(self.neural_memory),
            'hdc_memory_size': len(self.hdc_memory),
            'total_memory_items': len(self.neural_memory) + len(self.hdc_memory)
        }
        
        return analysis
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        return {
            'hybrid_metrics': self.hybrid_metrics,
            'memory_stats': {
                'neural_memory_items': len(self.neural_memory),
                'hdc_memory_items': len(self.hdc_memory)
            },
            'model_config': {
                'neural_dim': self.config.neural_hidden_dim,
                'hdc_dim': self.config.hdc_dimension,
                'fusion_strategy': self.config.fusion_strategy,
                'bidirectional': self.config.enable_bidirectional,
                'adaptive_fusion': self.config.enable_adaptive_fusion
            }
        }
        
    def save_hybrid_model(self, path: str):
        """Save the complete hybrid model."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save neural components
        torch.save(self.state_dict(), save_path / 'hybrid_model.pt')
        
        # Save HDC memory
        import pickle
        with open(save_path / 'hdc_memory.pkl', 'wb') as f:
            pickle.dump(self.hdc_memory, f)
            
        with open(save_path / 'neural_memory.pkl', 'wb') as f:
            pickle.dump(self.neural_memory, f)
            
        # Save configuration
        import json
        config_dict = {
            'neural_hidden_dim': self.config.neural_hidden_dim,
            'hdc_dimension': self.config.hdc_dimension,
            'fusion_strategy': self.config.fusion_strategy,
            'learning_rate': self.config.learning_rate,
            'hdc_learning_rate': self.config.hdc_learning_rate,
            'enable_bidirectional': self.config.enable_bidirectional,
            'enable_adaptive_fusion': self.config.enable_adaptive_fusion,
            'dropout_rate': self.config.dropout_rate
        }
        
        with open(save_path / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
            
        # Save metrics
        with open(save_path / 'metrics.json', 'w') as f:
            json.dump(self.hybrid_metrics, f, indent=2)
            
    def load_hybrid_model(self, path: str):
        """Load the complete hybrid model."""
        load_path = Path(path)
        
        # Load neural components
        model_path = load_path / 'hybrid_model.pt'
        if model_path.exists():
            self.load_state_dict(torch.load(model_path))
            
        # Load HDC memory
        import pickle
        hdc_memory_path = load_path / 'hdc_memory.pkl'
        if hdc_memory_path.exists():
            with open(hdc_memory_path, 'rb') as f:
                self.hdc_memory = pickle.load(f)
                
        neural_memory_path = load_path / 'neural_memory.pkl'
        if neural_memory_path.exists():
            with open(neural_memory_path, 'rb') as f:
                self.neural_memory = pickle.load(f)
                
        # Load metrics
        metrics_path = load_path / 'metrics.json'
        if metrics_path.exists():
            import json
            with open(metrics_path, 'r') as f:
                self.hybrid_metrics = json.load(f)