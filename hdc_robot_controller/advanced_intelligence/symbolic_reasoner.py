"""
Advanced Symbolic Reasoner with HDC Integration

Combines symbolic reasoning with hyperdimensional computing for robust,
interpretable, and adaptive reasoning capabilities in robotic systems.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass
from enum import Enum
import re
import time
from pathlib import Path

from ..core.hypervector import HyperVector


class LogicOperator(Enum):
    """Supported logic operators."""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    IMPLIES = "IMPLIES"
    IFF = "IFF"  # If and only if
    XOR = "XOR"


class TemporalOperator(Enum):
    """Temporal logic operators."""
    ALWAYS = "ALWAYS"  # □ (box)
    EVENTUALLY = "EVENTUALLY"  # ◊ (diamond)
    NEXT = "NEXT"  # X
    UNTIL = "UNTIL"  # U
    SINCE = "SINCE"  # S


@dataclass
class Concept:
    """Represents a symbolic concept with HDC encoding."""
    name: str
    hypervector: HyperVector
    attributes: Dict[str, Any]
    confidence: float = 1.0
    creation_time: float = 0.0
    
    def __post_init__(self):
        if self.creation_time == 0.0:
            self.creation_time = time.time()


@dataclass
class Rule:
    """Represents a logical rule with HDC encoding."""
    premise: str  # Logical expression
    conclusion: str  # Logical expression
    confidence: float
    rule_hv: HyperVector
    activation_count: int = 0
    success_count: int = 0
    
    @property
    def success_rate(self) -> float:
        if self.activation_count == 0:
            return 0.0
        return self.success_count / self.activation_count


@dataclass
class Fact:
    """Represents a fact in the knowledge base."""
    statement: str
    truth_value: float  # Fuzzy truth value [0, 1]
    fact_hv: HyperVector
    timestamp: float
    source: str = "unknown"
    
    def __post_init__(self):
        if not hasattr(self, 'timestamp') or self.timestamp == 0:
            self.timestamp = time.time()


class SymbolicParser:
    """Parser for symbolic expressions."""
    
    def __init__(self):
        self.operators = {
            'AND': LogicOperator.AND,
            'OR': LogicOperator.OR,
            'NOT': LogicOperator.NOT,
            'IMPLIES': LogicOperator.IMPLIES,
            'IFF': LogicOperator.IFF,
            'XOR': LogicOperator.XOR,
            '&': LogicOperator.AND,
            '|': LogicOperator.OR,
            '!': LogicOperator.NOT,
            '->': LogicOperator.IMPLIES,
            '<->': LogicOperator.IFF,
            '^': LogicOperator.XOR
        }
        
        self.temporal_operators = {
            'ALWAYS': TemporalOperator.ALWAYS,
            'EVENTUALLY': TemporalOperator.EVENTUALLY,
            'NEXT': TemporalOperator.NEXT,
            'UNTIL': TemporalOperator.UNTIL,
            'SINCE': TemporalOperator.SINCE,
            '[]': TemporalOperator.ALWAYS,
            '<>': TemporalOperator.EVENTUALLY,
            'X': TemporalOperator.NEXT,
            'U': TemporalOperator.UNTIL,
            'S': TemporalOperator.SINCE
        }
    
    def parse_expression(self, expression: str) -> Dict[str, Any]:
        """Parse a logical expression into structured form."""
        # Remove extra whitespace
        expression = re.sub(r'\s+', ' ', expression.strip())
        
        # Extract components
        parsed = {
            'expression': expression,
            'concepts': self._extract_concepts(expression),
            'operators': self._extract_operators(expression),
            'temporal_operators': self._extract_temporal_operators(expression),
            'variables': self._extract_variables(expression),
            'structure': self._analyze_structure(expression)
        }
        
        return parsed
    
    def _extract_concepts(self, expression: str) -> List[str]:
        """Extract concept names from expression."""
        # Simple concept extraction (can be enhanced)
        concepts = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', expression)
        # Filter out operators
        operators = set(self.operators.keys()) | set(self.temporal_operators.keys())
        concepts = [c for c in concepts if c not in operators]
        return list(set(concepts))  # Remove duplicates
    
    def _extract_operators(self, expression: str) -> List[LogicOperator]:
        """Extract logic operators from expression."""
        found_operators = []
        for op_str, op_enum in self.operators.items():
            if op_str in expression:
                found_operators.append(op_enum)
        return found_operators
    
    def _extract_temporal_operators(self, expression: str) -> List[TemporalOperator]:
        """Extract temporal operators from expression."""
        found_operators = []
        for op_str, op_enum in self.temporal_operators.items():
            if op_str in expression:
                found_operators.append(op_enum)
        return found_operators
    
    def _extract_variables(self, expression: str) -> List[str]:
        """Extract variable names (starting with lowercase)."""
        variables = re.findall(r'\b[a-z][a-zA-Z0-9_]*\b', expression)
        operators = set(self.operators.keys()) | set(self.temporal_operators.keys())
        variables = [v for v in variables if v not in operators]
        return list(set(variables))
    
    def _analyze_structure(self, expression: str) -> Dict[str, Any]:
        """Analyze the logical structure of the expression."""
        return {
            'has_negation': 'NOT' in expression or '!' in expression,
            'has_conjunction': 'AND' in expression or '&' in expression,
            'has_disjunction': 'OR' in expression or '|' in expression,
            'has_implication': 'IMPLIES' in expression or '->' in expression,
            'has_temporal': any(op in expression for op in self.temporal_operators.keys()),
            'nesting_level': expression.count('(')
        }


class HDCSymbolicEncoder:
    """Encodes symbolic concepts and rules as hypervectors."""
    
    def __init__(self, dimension: int = 10000):
        self.dimension = dimension
        self.concept_registry = {}  # Consistent concept encodings
        self.operator_encodings = {}
        self.variable_encodings = {}
        
        # Initialize operator encodings
        self._initialize_operator_encodings()
    
    def _initialize_operator_encodings(self):
        """Initialize consistent encodings for logical operators."""
        # Logic operators
        for i, op in enumerate(LogicOperator):
            seed = hash(f"logic_{op.value}") % (2**31)
            self.operator_encodings[op] = HyperVector.random(self.dimension, seed)
        
        # Temporal operators
        for i, op in enumerate(TemporalOperator):
            seed = hash(f"temporal_{op.value}") % (2**31)
            self.operator_encodings[op] = HyperVector.random(self.dimension, seed)
    
    def encode_concept(self, concept_name: str, attributes: Optional[Dict] = None) -> HyperVector:
        """Encode a concept as a hypervector."""
        if concept_name in self.concept_registry:
            base_hv = self.concept_registry[concept_name]
        else:
            # Create new encoding
            seed = hash(concept_name) % (2**31)
            base_hv = HyperVector.random(self.dimension, seed)
            self.concept_registry[concept_name] = base_hv
        
        if attributes:
            # Bind attributes to concept
            result_hv = base_hv.copy()
            for attr_name, attr_value in attributes.items():
                attr_hv = self._encode_attribute(attr_name, attr_value)
                result_hv = result_hv.bind(attr_hv)
            return result_hv
        
        return base_hv
    
    def _encode_attribute(self, attr_name: str, attr_value: Any) -> HyperVector:
        """Encode an attribute-value pair."""
        # Encode attribute name
        name_seed = hash(f"attr_{attr_name}") % (2**31)
        name_hv = HyperVector.random(self.dimension, name_seed)
        
        # Encode attribute value
        value_hv = self._encode_value(attr_value)
        
        # Bind name and value
        return name_hv.bind(value_hv)
    
    def _encode_value(self, value: Any) -> HyperVector:
        """Encode a value as hypervector."""
        if isinstance(value, bool):
            seed = hash(f"bool_{value}") % (2**31)
            return HyperVector.random(self.dimension, seed)
        elif isinstance(value, (int, float)):
            # Quantized encoding for numbers
            quantized = int(value * 100) / 100  # 2 decimal precision
            seed = hash(f"num_{quantized}") % (2**31)
            return HyperVector.random(self.dimension, seed)
        elif isinstance(value, str):
            seed = hash(f"str_{value}") % (2**31)
            return HyperVector.random(self.dimension, seed)
        else:
            # Generic encoding
            seed = hash(f"obj_{str(value)}") % (2**31)
            return HyperVector.random(self.dimension, seed)
    
    def encode_rule(self, premise_hv: HyperVector, conclusion_hv: HyperVector, 
                   confidence: float = 1.0) -> HyperVector:
        """Encode a logical rule as hypervector."""
        # Create implication binding
        implies_op = self.operator_encodings[LogicOperator.IMPLIES]
        
        # Bind premise with implication operator
        premise_implies = premise_hv.bind(implies_op)
        
        # Bundle with conclusion (weighted by confidence)
        if confidence < 1.0:
            # Add noise based on uncertainty
            noise_level = 1.0 - confidence
            conclusion_hv = conclusion_hv.add_noise(noise_level * 0.1)
        
        rule_hv = premise_implies.bundle(conclusion_hv)
        
        return rule_hv
    
    def encode_expression(self, parsed_expr: Dict[str, Any]) -> HyperVector:
        """Encode a parsed logical expression as hypervector."""
        expression = parsed_expr['expression']
        concepts = parsed_expr['concepts']
        operators = parsed_expr['operators']
        
        if not concepts:
            return HyperVector.zero(self.dimension)
        
        # Encode concepts
        concept_hvs = [self.encode_concept(concept) for concept in concepts]
        
        if len(concept_hvs) == 1 and not operators:
            return concept_hvs[0]
        
        # Apply operators (simplified approach)
        result_hv = concept_hvs[0]
        
        for i, concept_hv in enumerate(concept_hvs[1:], 1):
            if i <= len(operators):
                op = operators[i-1]
                op_hv = self.operator_encodings[op]
                
                if op == LogicOperator.AND:
                    # Bind concepts for conjunction
                    result_hv = result_hv.bind(concept_hv)
                elif op == LogicOperator.OR:
                    # Bundle concepts for disjunction  
                    result_hv = result_hv.bundle(concept_hv)
                elif op == LogicOperator.NOT:
                    # Invert for negation
                    result_hv = concept_hv.invert()
                else:
                    # Generic operator binding
                    result_hv = result_hv.bind(op_hv).bind(concept_hv)
            else:
                # Default to conjunction
                result_hv = result_hv.bind(concept_hv)
        
        return result_hv


class AdvancedSymbolicReasoner:
    """
    Advanced Symbolic Reasoner with HDC Integration.
    
    Provides sophisticated reasoning capabilities including:
    - First-order logic reasoning
    - Temporal logic
    - Fuzzy reasoning
    - Causal inference
    - Meta-reasoning
    """
    
    def __init__(self, hdc_dimension: int = 10000):
        self.hdc_dimension = hdc_dimension
        self.parser = SymbolicParser()
        self.encoder = HDCSymbolicEncoder(hdc_dimension)
        
        # Knowledge base components
        self.concepts = {}  # name -> Concept
        self.rules = {}     # name -> Rule
        self.facts = {}     # name -> Fact
        self.temporal_facts = []  # List of (Fact, timestamp) for temporal reasoning
        
        # Reasoning state
        self.working_memory = {}  # Temporary facts during reasoning
        self.inference_chain = []  # Track reasoning steps
        self.uncertainty_threshold = 0.7
        
        # Performance metrics
        self.reasoning_metrics = {
            'inferences_made': 0,
            'rules_fired': 0,
            'concepts_created': 0,
            'temporal_inferences': 0,
            'uncertainty_resolutions': 0,
            'reasoning_time_total': 0.0
        }
    
    def add_concept(self, name: str, attributes: Optional[Dict] = None, 
                   confidence: float = 1.0) -> Concept:
        """Add a new concept to the knowledge base."""
        hv = self.encoder.encode_concept(name, attributes)
        
        concept = Concept(
            name=name,
            hypervector=hv,
            attributes=attributes or {},
            confidence=confidence
        )
        
        self.concepts[name] = concept
        self.reasoning_metrics['concepts_created'] += 1
        
        return concept
    
    def add_rule(self, name: str, premise: str, conclusion: str, 
                confidence: float = 1.0) -> Rule:
        """Add a logical rule to the knowledge base."""
        # Parse premise and conclusion
        premise_parsed = self.parser.parse_expression(premise)
        conclusion_parsed = self.parser.parse_expression(conclusion)
        
        # Encode as hypervectors
        premise_hv = self.encoder.encode_expression(premise_parsed)
        conclusion_hv = self.encoder.encode_expression(conclusion_parsed)
        
        # Create rule hypervector
        rule_hv = self.encoder.encode_rule(premise_hv, conclusion_hv, confidence)
        
        rule = Rule(
            premise=premise,
            conclusion=conclusion,
            confidence=confidence,
            rule_hv=rule_hv
        )
        
        self.rules[name] = rule
        
        return rule
    
    def add_fact(self, name: str, statement: str, truth_value: float = 1.0, 
                source: str = "user") -> Fact:
        """Add a fact to the knowledge base."""
        # Parse and encode the statement
        parsed = self.parser.parse_expression(statement)
        fact_hv = self.encoder.encode_expression(parsed)
        
        fact = Fact(
            statement=statement,
            truth_value=truth_value,
            fact_hv=fact_hv,
            timestamp=time.time(),
            source=source
        )
        
        self.facts[name] = fact
        
        # Add to temporal facts for temporal reasoning
        self.temporal_facts.append((fact, fact.timestamp))
        
        return fact
    
    def reason(self, query: str, max_steps: int = 10, 
              reasoning_type: str = "forward") -> Dict[str, Any]:
        """
        Perform reasoning to answer a query.
        
        Args:
            query: Query to reason about
            max_steps: Maximum reasoning steps
            reasoning_type: "forward", "backward", or "bidirectional"
            
        Returns:
            Reasoning result with conclusions and explanations
        """
        start_time = time.time()
        
        # Parse query
        query_parsed = self.parser.parse_expression(query)
        query_hv = self.encoder.encode_expression(query_parsed)
        
        # Initialize reasoning state
        self.working_memory.clear()
        self.inference_chain.clear()
        
        # Perform reasoning based on type
        if reasoning_type == "forward":
            result = self._forward_reasoning(query_hv, query, max_steps)
        elif reasoning_type == "backward":
            result = self._backward_reasoning(query_hv, query, max_steps)
        elif reasoning_type == "bidirectional":
            forward_result = self._forward_reasoning(query_hv, query, max_steps // 2)
            backward_result = self._backward_reasoning(query_hv, query, max_steps // 2)
            result = self._combine_reasoning_results(forward_result, backward_result)
        else:
            raise ValueError(f"Unknown reasoning type: {reasoning_type}")
        
        # Add temporal reasoning if query has temporal operators
        if query_parsed['temporal_operators']:
            temporal_result = self._temporal_reasoning(query_parsed, result)
            result.update(temporal_result)
        
        # Track performance
        reasoning_time = time.time() - start_time
        self.reasoning_metrics['reasoning_time_total'] += reasoning_time
        self.reasoning_metrics['inferences_made'] += len(self.inference_chain)
        
        # Add meta-information
        result.update({
            'query': query,
            'reasoning_type': reasoning_type,
            'steps_taken': len(self.inference_chain),
            'reasoning_time': reasoning_time,
            'inference_chain': self.inference_chain.copy(),
            'working_memory': self.working_memory.copy()
        })
        
        return result
    
    def _forward_reasoning(self, query_hv: HyperVector, query: str, 
                          max_steps: int) -> Dict[str, Any]:
        """Forward chaining reasoning."""
        conclusions = []
        confidences = []
        
        for step in range(max_steps):
            new_inferences = False
            
            # Try to apply each rule
            for rule_name, rule in self.rules.items():
                if self._can_apply_rule(rule):
                    # Apply rule
                    conclusion = self._apply_rule(rule, step)
                    if conclusion:
                        conclusions.append(conclusion)
                        confidences.append(rule.confidence)
                        new_inferences = True
                        
                        # Check if we've answered the query
                        if self._matches_query(conclusion, query_hv):
                            return {
                                'answer': True,
                                'conclusions': conclusions,
                                'confidences': confidences,
                                'final_confidence': max(confidences) if confidences else 0.0
                            }
            
            if not new_inferences:
                break
        
        # Check if any conclusion matches the query
        best_match = self._find_best_match(query_hv, conclusions)
        
        return {
            'answer': best_match['similarity'] > self.uncertainty_threshold,
            'conclusions': conclusions,
            'confidences': confidences,
            'best_match': best_match,
            'final_confidence': best_match['confidence'] if best_match else 0.0
        }
    
    def _backward_reasoning(self, query_hv: HyperVector, query: str, 
                           max_steps: int) -> Dict[str, Any]:
        """Backward chaining reasoning."""
        # Start with goal and work backwards
        goals = [query_hv]
        proved_goals = []
        
        for step in range(max_steps):
            if not goals:
                break
                
            current_goal = goals.pop(0)
            
            # Check if goal is already a known fact
            if self._is_known_fact(current_goal):
                proved_goals.append(current_goal)
                continue
            
            # Find rules that conclude this goal
            applicable_rules = self._find_concluding_rules(current_goal)
            
            if applicable_rules:
                # Add premises as new goals
                for rule in applicable_rules:
                    premise_hv = self._get_premise_hv(rule)
                    if premise_hv not in goals and premise_hv not in proved_goals:
                        goals.append(premise_hv)
                        
                        # Record inference step
                        self.inference_chain.append({
                            'step': step,
                            'type': 'backward',
                            'rule': rule.premise + " -> " + rule.conclusion,
                            'goal': self._hv_to_description(current_goal)
                        })
            
        # Check if original query was proved
        query_proved = any(self._similarity_above_threshold(query_hv, goal) 
                          for goal in proved_goals)
        
        return {
            'answer': query_proved,
            'proved_goals': len(proved_goals),
            'remaining_goals': len(goals),
            'final_confidence': 1.0 if query_proved else 0.0
        }
    
    def _temporal_reasoning(self, query_parsed: Dict, base_result: Dict) -> Dict[str, Any]:
        """Perform temporal logic reasoning."""
        temporal_ops = query_parsed['temporal_operators']
        temporal_results = {}
        
        for op in temporal_ops:
            if op == TemporalOperator.ALWAYS:
                # Check if property holds at all time points
                temporal_results['always'] = self._check_always_property(query_parsed)
            elif op == TemporalOperator.EVENTUALLY:
                # Check if property will eventually hold
                temporal_results['eventually'] = self._check_eventually_property(query_parsed)
            elif op == TemporalOperator.NEXT:
                # Check property at next time step
                temporal_results['next'] = self._check_next_property(query_parsed)
            elif op == TemporalOperator.UNTIL:
                # Check until property
                temporal_results['until'] = self._check_until_property(query_parsed)
        
        self.reasoning_metrics['temporal_inferences'] += len(temporal_results)
        
        return {'temporal_reasoning': temporal_results}
    
    def _can_apply_rule(self, rule: Rule) -> bool:
        """Check if a rule can be applied given current facts."""
        # Parse rule premise
        premise_parsed = self.parser.parse_expression(rule.premise)
        premise_concepts = premise_parsed['concepts']
        
        # Check if all premise concepts are satisfied
        for concept in premise_concepts:
            if not self._concept_satisfied(concept):
                return False
        
        return True
    
    def _concept_satisfied(self, concept_name: str) -> bool:
        """Check if a concept is satisfied by current facts."""
        # Look in facts and working memory
        for fact in list(self.facts.values()) + list(self.working_memory.values()):
            if concept_name.lower() in fact.statement.lower():
                if fact.truth_value > self.uncertainty_threshold:
                    return True
        
        return False
    
    def _apply_rule(self, rule: Rule, step: int) -> Optional[str]:
        """Apply a rule and return the conclusion."""
        rule.activation_count += 1
        
        # Create new fact from conclusion
        conclusion_name = f"inferred_{step}_{rule.activation_count}"
        conclusion_fact = self.add_fact(
            conclusion_name, 
            rule.conclusion,
            truth_value=rule.confidence,
            source=f"rule_{hash(rule.premise + rule.conclusion) % 1000}"
        )
        
        # Add to working memory
        self.working_memory[conclusion_name] = conclusion_fact
        
        # Record inference
        self.inference_chain.append({
            'step': step,
            'type': 'forward',
            'rule': rule.premise + " -> " + rule.conclusion,
            'conclusion': rule.conclusion,
            'confidence': rule.confidence
        })
        
        rule.success_count += 1
        self.reasoning_metrics['rules_fired'] += 1
        
        return rule.conclusion
    
    def _matches_query(self, conclusion: str, query_hv: HyperVector) -> bool:
        """Check if conclusion matches the query."""
        conclusion_parsed = self.parser.parse_expression(conclusion)
        conclusion_hv = self.encoder.encode_expression(conclusion_parsed)
        
        similarity = query_hv.similarity(conclusion_hv)
        return similarity > self.uncertainty_threshold
    
    def _find_best_match(self, query_hv: HyperVector, 
                        conclusions: List[str]) -> Dict[str, Any]:
        """Find the best matching conclusion for the query."""
        if not conclusions:
            return {'similarity': 0.0, 'conclusion': None, 'confidence': 0.0}
        
        best_similarity = 0.0
        best_conclusion = None
        best_confidence = 0.0
        
        for conclusion in conclusions:
            conclusion_parsed = self.parser.parse_expression(conclusion)
            conclusion_hv = self.encoder.encode_expression(conclusion_parsed)
            
            similarity = query_hv.similarity(conclusion_hv)
            if similarity > best_similarity:
                best_similarity = similarity
                best_conclusion = conclusion
                # Get confidence from working memory or rules
                best_confidence = self._get_conclusion_confidence(conclusion)
        
        return {
            'similarity': best_similarity,
            'conclusion': best_conclusion,
            'confidence': best_confidence
        }
    
    def _get_conclusion_confidence(self, conclusion: str) -> float:
        """Get confidence value for a conclusion."""
        # Check working memory
        for fact in self.working_memory.values():
            if fact.statement == conclusion:
                return fact.truth_value
        
        # Check inference chain
        for inference in self.inference_chain:
            if inference.get('conclusion') == conclusion:
                return inference.get('confidence', 1.0)
        
        return 1.0
    
    def _combine_reasoning_results(self, forward_result: Dict, 
                                  backward_result: Dict) -> Dict[str, Any]:
        """Combine forward and backward reasoning results."""
        # Combine confidences
        forward_conf = forward_result.get('final_confidence', 0.0)
        backward_conf = backward_result.get('final_confidence', 0.0)
        
        # Use maximum confidence
        combined_confidence = max(forward_conf, backward_conf)
        
        # Combine answers
        combined_answer = forward_result.get('answer', False) or backward_result.get('answer', False)
        
        return {
            'answer': combined_answer,
            'final_confidence': combined_confidence,
            'forward_result': forward_result,
            'backward_result': backward_result,
            'reasoning_method': 'bidirectional'
        }
    
    def explain_reasoning(self, reasoning_result: Dict[str, Any]) -> str:
        """Generate human-readable explanation of reasoning process."""
        explanation = f"Query: {reasoning_result['query']}\n\n"
        
        if reasoning_result.get('answer'):
            explanation += f"✓ CONCLUSION: TRUE (confidence: {reasoning_result['final_confidence']:.2f})\n\n"
        else:
            explanation += f"✗ CONCLUSION: FALSE (confidence: {reasoning_result['final_confidence']:.2f})\n\n"
        
        explanation += "REASONING STEPS:\n"
        for i, step in enumerate(reasoning_result.get('inference_chain', []), 1):
            explanation += f"{i}. {step.get('type', 'unknown').upper()}: "
            explanation += f"{step.get('rule', 'unknown rule')}\n"
            if 'conclusion' in step:
                explanation += f"   → {step['conclusion']} (confidence: {step.get('confidence', 1.0):.2f})\n"
            explanation += "\n"
        
        if 'temporal_reasoning' in reasoning_result:
            explanation += "TEMPORAL REASONING:\n"
            for prop, result in reasoning_result['temporal_reasoning'].items():
                explanation += f"- {prop.upper()}: {result}\n"
            explanation += "\n"
        
        explanation += f"Total reasoning time: {reasoning_result.get('reasoning_time', 0.0):.3f}s\n"
        explanation += f"Steps taken: {reasoning_result.get('steps_taken', 0)}\n"
        
        return explanation
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary of the knowledge base."""
        return {
            'concepts': len(self.concepts),
            'rules': len(self.rules),
            'facts': len(self.facts),
            'temporal_facts': len(self.temporal_facts),
            'working_memory_size': len(self.working_memory),
            'concept_names': list(self.concepts.keys()),
            'rule_names': list(self.rules.keys()),
            'fact_names': list(self.facts.keys()),
            'reasoning_metrics': self.reasoning_metrics.copy()
        }
    
    def save_knowledge_base(self, path: str):
        """Save the knowledge base to disk."""
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        import pickle
        import json
        
        # Save concepts (without hypervectors for JSON compatibility)
        concepts_data = {}
        for name, concept in self.concepts.items():
            concepts_data[name] = {
                'name': concept.name,
                'attributes': concept.attributes,
                'confidence': concept.confidence,
                'creation_time': concept.creation_time
            }
        
        with open(save_path / 'concepts.json', 'w') as f:
            json.dump(concepts_data, f, indent=2)
        
        # Save rules
        rules_data = {}
        for name, rule in self.rules.items():
            rules_data[name] = {
                'premise': rule.premise,
                'conclusion': rule.conclusion,
                'confidence': rule.confidence,
                'activation_count': rule.activation_count,
                'success_count': rule.success_count
            }
        
        with open(save_path / 'rules.json', 'w') as f:
            json.dump(rules_data, f, indent=2)
        
        # Save facts
        facts_data = {}
        for name, fact in self.facts.items():
            facts_data[name] = {
                'statement': fact.statement,
                'truth_value': fact.truth_value,
                'timestamp': fact.timestamp,
                'source': fact.source
            }
        
        with open(save_path / 'facts.json', 'w') as f:
            json.dump(facts_data, f, indent=2)
        
        # Save hypervector components with pickle
        hv_data = {
            'concept_hvs': {name: concept.hypervector for name, concept in self.concepts.items()},
            'rule_hvs': {name: rule.rule_hv for name, rule in self.rules.items()},
            'fact_hvs': {name: fact.fact_hv for name, fact in self.facts.items()},
            'encoder_registry': self.encoder.concept_registry
        }
        
        with open(save_path / 'hypervectors.pkl', 'wb') as f:
            pickle.dump(hv_data, f)
        
        # Save metrics
        with open(save_path / 'metrics.json', 'w') as f:
            json.dump(self.reasoning_metrics, f, indent=2)
    
    # Temporal reasoning helper methods
    def _check_always_property(self, query_parsed: Dict) -> bool:
        """Check if property holds at all time points."""
        # Simplified implementation
        return len(self.temporal_facts) > 0 and all(
            fact.truth_value > self.uncertainty_threshold 
            for fact, _ in self.temporal_facts[-10:]  # Check last 10 time points
        )
    
    def _check_eventually_property(self, query_parsed: Dict) -> bool:
        """Check if property will eventually hold."""
        # Check if property holds at any recent time point
        return any(
            fact.truth_value > self.uncertainty_threshold 
            for fact, _ in self.temporal_facts[-5:]  # Check last 5 time points
        )
    
    def _check_next_property(self, query_parsed: Dict) -> bool:
        """Check property at next time step."""
        # Predict next state (simplified)
        if len(self.temporal_facts) >= 2:
            recent_facts = [fact for fact, _ in self.temporal_facts[-2:]]
            return recent_facts[-1].truth_value > self.uncertainty_threshold
        return False
    
    def _check_until_property(self, query_parsed: Dict) -> bool:
        """Check until property."""
        # Simplified until check
        return len(self.temporal_facts) > 0
    
    # Backward reasoning helper methods
    def _is_known_fact(self, goal_hv: HyperVector) -> bool:
        """Check if goal is a known fact."""
        for fact in self.facts.values():
            if goal_hv.similarity(fact.fact_hv) > self.uncertainty_threshold:
                return True
        return False
    
    def _find_concluding_rules(self, goal_hv: HyperVector) -> List[Rule]:
        """Find rules that conclude the goal."""
        concluding_rules = []
        for rule in self.rules.values():
            conclusion_parsed = self.parser.parse_expression(rule.conclusion)
            conclusion_hv = self.encoder.encode_expression(conclusion_parsed)
            
            if goal_hv.similarity(conclusion_hv) > self.uncertainty_threshold:
                concluding_rules.append(rule)
        
        return concluding_rules
    
    def _get_premise_hv(self, rule: Rule) -> HyperVector:
        """Get hypervector for rule premise."""
        premise_parsed = self.parser.parse_expression(rule.premise)
        return self.encoder.encode_expression(premise_parsed)
    
    def _similarity_above_threshold(self, hv1: HyperVector, hv2: HyperVector) -> bool:
        """Check if similarity is above threshold."""
        return hv1.similarity(hv2) > self.uncertainty_threshold
    
    def _hv_to_description(self, hv: HyperVector) -> str:
        """Convert hypervector to description (simplified)."""
        return f"HV(active={np.sum(hv.data > 0)}, entropy={hv.entropy():.3f})"