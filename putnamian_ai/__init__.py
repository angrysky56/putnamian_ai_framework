"""
Contradiction-Driven Neural Evolution (CDNE) Framework

Core module implementing Peter Putnam's insights about brain function
as computational principles for AI systems.

Key Concepts:
- Intelligence emerges through architectural self-modification
- Contradictions drive neural pathway evolution  
- Environmental variables resolve logical deadlocks
- Inductive reasoning through contradiction synthesis
"""

from .core import ConflictDetectionEngine, VariableSpaceExplorer, ArchitecturalModifier
from .networks import CDNENetwork
from .utils import ConflictType, ResolutionStrategy

__version__ = "0.1.0"

__all__ = [
    "ConflictDetectionEngine",
    "VariableSpaceExplorer", 
    "ArchitecturalModifier",
    "CDNENetwork",
    "ConflictType",
    "ResolutionStrategy",
]
