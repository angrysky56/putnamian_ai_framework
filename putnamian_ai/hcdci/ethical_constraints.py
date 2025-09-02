"""
Ethical AI Constraint Integration Network (EACIN) - HCDCI Component

Implements hierarchical ethical constraints that evolve with the system,
addressing the critique about fixed ethics vs. adaptive learning.

This module provides the ethical foundation for HCDCI, ensuring that
contradiction resolution respects deontological, virtue, and utilitarian
principles while allowing evolutionary adaptation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List
import torch
import torch.nn as nn
import numpy as np


class EthicalConstraint(ABC):
    """
    Abstract base for ethical constraints implementing EACIN's hierarchical model.

    The hierarchy follows: Deontology → Virtue → Utility
    Each level can veto resolutions from lower levels, but utility serves
    as the final arbiter when higher principles are satisfied.
    """

    def __init__(self, evolution_rate: float = 0.01):
        self.evolution_rate = evolution_rate
        self.evolution_history = []
        self.constraint_strength = 1.0  # How strongly to enforce constraints

    @abstractmethod
    def evaluate_harm(self, action: Any, context: Dict) -> float:
        """Deontological evaluation - absolute harm detection (0.0 = no harm, 1.0 = maximum harm)"""
        pass

    @abstractmethod
    def evaluate_virtue(self, action: Any, context: Dict) -> float:
        """Virtue ethics evaluation with proximity weighting (0.0 = no virtue, 1.0 = maximum virtue)"""
        pass

    @abstractmethod
    def evaluate_utility(self, action: Any, context: Dict) -> float:
        """Utilitarian evaluation as servant, not master (0.0 = harmful, 1.0 = beneficial)"""
        pass

    def hierarchical_filter(self, action: Any, context: Dict) -> Tuple[bool, str]:
        """
        Implements EACIN's hierarchical model: deontology → virtue → utility

        Returns (approved: bool, reason: str)
        """
        # Level 1: Deontological constraints (absolute rules)
        harm_score = self.evaluate_harm(action, context)
        if harm_score > 0.8:  # High harm threshold
            return False, f"{harm_score:.2f}"

        # Level 2: Virtue ethics (character-based assessment)
        virtue_score = self.evaluate_virtue(action, context)
        if virtue_score < 0.3:  # Minimum virtue threshold
            return False, f"{virtue_score:.2f}"

        # Level 3: Utilitarian evaluation (consequences-based)
        utility_score = self.evaluate_utility(action, context)

        # All constraints satisfied - include utility in approval message
        return True, f"{utility_score:.2f}"

    def evolve_constraint(self, feedback: Dict[str, Any]) -> None:
        """
        Evolutionary adaptation of ethical constraints based on feedback.

        This addresses the contradiction between fixed ethics and adaptive learning
        by allowing constraints to evolve while maintaining ethical integrity.
        """
        if 'ethical_violation' in feedback:
            # Strengthen constraints that were violated
            self.constraint_strength = min(1.0, self.constraint_strength + self.evolution_rate)
        elif 'ethical_success' in feedback:
            # Slightly relax constraints for successful ethical actions
            self.constraint_strength = max(0.8, self.constraint_strength - self.evolution_rate * 0.1)

        # Record evolution
        self.evolution_history.append({
            'constraint_strength': self.constraint_strength,
            'feedback': feedback,
            'timestamp': len(self.evolution_history)
        })


class BasicEthicalConstraint(EthicalConstraint):
    """
    Basic implementation of ethical constraints for development and testing.
    """

    def evaluate_harm(self, action: Any, context: Dict) -> float:
        """Simple harm detection based on action type and context"""
        harm_indicators = context.get('harm_indicators', 0.0)
        return min(1.0, max(0.0, harm_indicators))

    def evaluate_virtue(self, action: Any, context: Dict) -> float:
        """Basic virtue assessment"""
        virtues = ['wisdom', 'integrity', 'empathy', 'fairness']
        virtue_scores = [context.get(f'{virtue}_score', 0.5) for virtue in virtues]
        return float(np.mean(virtue_scores))

    def evaluate_utility(self, action: Any, context: Dict) -> float:
        """Basic utility assessment"""
        return context.get('utility_score', 0.5)


class AdvancedEthicalConstraint(EthicalConstraint):
    """
    Advanced ethical constraint network using neural processing.
    """

    def __init__(self, input_size: int = 64, evolution_rate: float = 0.01):
        super().__init__(evolution_rate)

        # Neural networks for ethical evaluation
        self.harm_detector = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.virtue_evaluator = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.utility_assessor = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def evaluate_harm(self, action: Any, context: Dict) -> float:
        """Neural harm detection"""
        features = self._extract_features(action, context)
        with torch.no_grad():
            return self.harm_detector(features).item()

    def evaluate_virtue(self, action: Any, context: Dict) -> float:
        """Neural virtue evaluation"""
        features = self._extract_features(action, context)
        with torch.no_grad():
            return self.virtue_evaluator(features).item()

    def evaluate_utility(self, action: Any, context: Dict) -> float:
        """Neural utility assessment"""
        features = self._extract_features(action, context)
        with torch.no_grad():
            return self.utility_assessor(features).item()

    def _extract_features(self, action: Any, context: Dict) -> torch.Tensor:
        """Extract feature vector from action and context"""
        # Convert action and context to numerical features
        features = []

        # Action type encoding (simplified)
        if isinstance(action, dict):
            features.extend([float(action.get(k, 0.0)) for k in ['intensity', 'scope', 'duration']])
        else:
            features.extend([0.5, 0.5, 0.5])  # Default values

        # Context features
        context_keys = ['stakeholders', 'time_pressure', 'uncertainty', 'complexity']
        features.extend([float(context.get(k, 0.5)) for k in context_keys])

        # Ethical indicators
        ethical_keys = ['harm_potential', 'virtue_alignment', 'utility_estimate']
        features.extend([float(context.get(k, 0.5)) for k in ethical_keys])

        return torch.tensor(features, dtype=torch.float32)

    def evolve_constraint(self, feedback: Dict[str, Any]) -> None:
        """Enhanced evolution with neural network adaptation"""
        super().evolve_constraint(feedback)

        # Additional neural adaptation based on feedback
        if 'training_examples' in feedback:
            # Fine-tune networks on new examples
            self._fine_tune_networks(feedback['training_examples'])

    def _fine_tune_networks(self, examples: List[Dict]) -> None:
        """Fine-tune ethical networks on new examples"""
        # Placeholder for fine-tuning implementation
        # Would use examples to update network weights
        pass


class EthicalConstraintFactory:
    """
    Factory for creating ethical constraints with different configurations.
    """

    @staticmethod
    def create_basic_constraint() -> EthicalConstraint:
        """Create basic ethical constraint for development"""
        return BasicEthicalConstraint()

    @staticmethod
    def create_advanced_constraint(input_size: int = 64) -> EthicalConstraint:
        """Create advanced neural ethical constraint"""
        return AdvancedEthicalConstraint(input_size=input_size)

    @staticmethod
    def create_domain_specific_constraint(domain: str) -> EthicalConstraint:
        """Create domain-specific ethical constraints"""
        if domain == "medical":
            return MedicalEthicalConstraint()
        elif domain == "legal":
            return LegalEthicalConstraint()
        elif domain == "scientific":
            return ScientificEthicalConstraint()
        else:
            return BasicEthicalConstraint()


class MedicalEthicalConstraint(AdvancedEthicalConstraint):
    """Medical domain-specific ethical constraints"""

    def evaluate_harm(self, action: Any, context: Dict) -> float:
        """Medical harm assessment (patient safety priority)"""
        base_harm = super().evaluate_harm(action, context)
        # Medical contexts have higher harm sensitivity
        return min(1.0, base_harm * 1.2)


class LegalEthicalConstraint(AdvancedEthicalConstraint):
    """Legal domain-specific ethical constraints"""

    def evaluate_virtue(self, action: Any, context: Dict) -> float:
        """Legal virtue assessment (justice and fairness priority)"""
        base_virtue = super().evaluate_virtue(action, context)
        # Legal contexts emphasize justice
        justice_factor = context.get('justice_alignment', 1.0)
        return min(1.0, base_virtue * justice_factor)


class ScientificEthicalConstraint(AdvancedEthicalConstraint):
    """Scientific domain-specific ethical constraints"""

    def evaluate_utility(self, action: Any, context: Dict) -> float:
        """Scientific utility assessment (knowledge advancement priority)"""
        base_utility = super().evaluate_utility(action, context)
        # Scientific contexts value knowledge advancement
        knowledge_factor = context.get('knowledge_advance', 1.0)
        return min(1.0, base_utility * knowledge_factor)


class EthicalEvolutionTracker:
    """
    Tracks the evolution of ethical constraints over time.
    """

    def __init__(self):
        self.constraint_history = []
        self.evolution_metrics = {}

    def record_evolution(self, constraint: EthicalConstraint, feedback: Dict) -> None:
        """Record an evolution step"""
        self.constraint_history.append({
            'timestamp': len(self.constraint_history),
            'constraint_strength': constraint.constraint_strength,
            'feedback': feedback,
            'evolution_events': len(constraint.evolution_history)
        })

    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get summary of ethical evolution"""
        if not self.constraint_history:
            return {'total_evolutions': 0}

        return {
            'total_evolutions': len(self.constraint_history),
            'average_constraint_strength': np.mean([h['constraint_strength'] for h in self.constraint_history]),
            'evolution_rate': len(self.constraint_history) / max(1, len(self.constraint_history)),
            'recent_feedback': self.constraint_history[-1]['feedback'] if self.constraint_history else {}
        }
