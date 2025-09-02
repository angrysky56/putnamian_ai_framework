"""
Core data structures and enums for CDNE framework.

Defines fundamental types and categories for contradiction detection,
resolution strategies, and architectural modification patterns.
"""

from enum import Enum, auto
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import numpy as np


class ConflictType(Enum):
    """Types of logical conflicts that can arise in neural pathways."""

    PATHWAY_DEADLOCK = auto()      # Two pathways produce contradictory outputs
    RESOURCE_COMPETITION = auto()   # Multiple pathways compete for same resources
    GOAL_CONTRADICTION = auto()     # System goals conflict with each other
    TEMPORAL_INCONSISTENCY = auto() # Outputs vary incorrectly across time
    LOGICAL_PARADOX = auto()       # Self-referential contradictions


class ResolutionStrategy(Enum):
    """Strategies for resolving detected contradictions."""

    VARIABLE_INTRODUCTION = auto()  # Add new environmental/internal variables
    PATHWAY_SYNTHESIS = auto()      # Combine competing pathways into higher-order rule
    CONTEXT_PARTITIONING = auto()   # Separate contexts where different rules apply
    TEMPORAL_SEQUENCING = auto()    # Resolve through time-based activation patterns
    META_RULE_CREATION = auto()     # Create rules that govern when other rules apply


class ConflictState(BaseModel):
    """Represents a detected contradiction within the neural network."""

    conflict_type: ConflictType
    pathway_ids: List[str]
    input_state: Dict[str, float]
    conflicting_outputs: Dict[str, Any]
    confidence_score: float
    environmental_context: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True


class ResolutionCandidate(BaseModel):
    """Represents a potential solution to a detected conflict."""

    strategy: ResolutionStrategy
    new_variables: List[str]
    architectural_modifications: Dict[str, Any]
    predicted_effectiveness: float
    computational_cost: float

    class Config:
        arbitrary_types_allowed = True


class ArchitecturalGenome(BaseModel):
    """Represents the evolving structure of a CDNE network."""

    pathways: Dict[str, Dict[str, Any]]
    connections: List[tuple[str, str, float]]  # (from, to, weight)
    variable_space: Dict[str, Any]
    resolution_history: List[ResolutionCandidate]
    generation: int = 0

    class Config:
        arbitrary_types_allowed = True

    def mutate(self, mutation_rate: float = 0.1) -> "ArchitecturalGenome":
        """Create a mutated version of this genome for variation."""
        # Use shallow copy to avoid deepcopy issues with potential tensor content
        mutated = self.copy(deep=False)
        # Implementation would include pathway modification logic
        return mutated

    def crossover(self, other: "ArchitecturalGenome") -> "ArchitecturalGenome":
        """Combine with another genome to create hybrid architecture."""
        # Use shallow copy to avoid deepcopy issues
        hybrid = self.copy(deep=False)
        # Implementation would include pathway combination logic
        return hybrid
