"""
Classes implementing Putnam's contradiction-driven neural evolution principles.

These classes translate Putnam's theoretical insights about brain function into
computational mechanisms for AI systems that evolve through conflict resolution.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
import random

from .utils import (
    ConflictType, ResolutionStrategy, ConflictState,
    ResolutionCandidate, ArchitecturalGenome
)


class ConflictDetectionEngine:
    """
    Detects logical contradictions within neural pathways.

    Based on Putnam's insight that intelligence emerges when systems encounter
    unresolvable conflicts between equally-valid rules, forcing architectural innovation.
    """

    def __init__(self, threshold: float = 0.1, temporal_window: int = 10):
        self.threshold = threshold
        self.temporal_window = temporal_window
        self.conflict_history: List[ConflictState] = []
        self.pathway_outputs: defaultdict[str, List[Any]] = defaultdict(list)

        # Add configurable parameters for better flexibility
        self.temporal_variance_multiplier = 2.0  # Multiplier for temporal threshold
        self.max_history_length = 1000  # Limit memory usage
        self.enable_detailed_logging = False

    def detect_conflicts(
        self,
        pathway_outputs: Dict[str, Any],
        input_state: Dict[str, float],
        environmental_context: Optional[Dict[str, Any]] = None
    ) -> List[ConflictState]:
        """
        Identify contradictions between neural pathway outputs.

        Args:
            pathway_outputs: Current outputs from each neural pathway
            input_state: The input that produced these outputs
            environmental_context: Additional contextual information

        Returns:
            List of detected conflicts requiring resolution
        """
        conflicts = []

        # Store outputs for temporal analysis
        for pathway_id, output in pathway_outputs.items():
            self.pathway_outputs[pathway_id].append(output)

        # Detect pathway deadlocks (Putnam's primary insight)
        deadlock_conflicts = self._detect_pathway_deadlocks(
            pathway_outputs, input_state, environmental_context
        )
        conflicts.extend(deadlock_conflicts)

        # Detect temporal inconsistencies
        temporal_conflicts = self._detect_temporal_inconsistencies(
            pathway_outputs, input_state
        )
        conflicts.extend(temporal_conflicts)

        # Store in history with memory management
        self.conflict_history.extend(conflicts)

        # Limit memory usage by pruning old history
        if len(self.conflict_history) > self.max_history_length:
            # Keep most recent conflicts and high-confidence ones
            recent_conflicts = self.conflict_history[-self.max_history_length//2:]
            high_confidence = [c for c in self.conflict_history[:-self.max_history_length//2]
                             if c.confidence_score > 0.8]
            self.conflict_history = high_confidence + recent_conflicts
            self.conflict_history = self.conflict_history[:self.max_history_length]

        # Manage pathway outputs memory
        for pathway_id in self.pathway_outputs:
            if len(self.pathway_outputs[pathway_id]) > self.max_history_length:
                self.pathway_outputs[pathway_id] = self.pathway_outputs[pathway_id][-self.max_history_length:]

        # Log conflicts if enabled
        if self.enable_detailed_logging and conflicts:
            print(f"Detected {len(conflicts)} conflicts: {[c.conflict_type.value for c in conflicts]}")

        return conflicts

    def _detect_pathway_deadlocks(
        self,
        pathway_outputs: Dict[str, Any],
        input_state: Dict[str, float],
        environmental_context: Optional[Dict[str, Any]]
    ) -> List[ConflictState]:
        """Detect when pathways produce contradictory outputs for same input."""
        conflicts = []

        # Compare all pathway pairs for contradictions
        pathway_items = list(pathway_outputs.items())
        for i, (pathway_a, output_a) in enumerate(pathway_items):
            for j, (pathway_b, output_b) in enumerate(pathway_items[i+1:], i+1):

                # Check if outputs are contradictory
                contradiction_score = self._calculate_contradiction_score(
                    output_a, output_b
                )

                if contradiction_score > self.threshold:
                    conflict = ConflictState(
                        conflict_type=ConflictType.PATHWAY_DEADLOCK,
                        pathway_ids=[pathway_a, pathway_b],
                        input_state=input_state,
                        conflicting_outputs={pathway_a: output_a, pathway_b: output_b},
                        confidence_score=contradiction_score,
                        environmental_context=environmental_context
                    )
                    conflicts.append(conflict)

        return conflicts

    def _detect_temporal_inconsistencies(
        self,
        pathway_outputs: Dict[str, Any],
        input_state: Dict[str, float]
    ) -> List[ConflictState]:
        """Detect when pathway outputs vary incorrectly across time."""
        conflicts = []

        for pathway_id, recent_outputs in self.pathway_outputs.items():
            if len(recent_outputs) >= self.temporal_window:
                # Check for inappropriate variation in outputs for similar inputs
                variance = np.var([self._output_to_float(out) for out in recent_outputs[-self.temporal_window:]])

                # Use configurable multiplier instead of hardcoded 2
                temporal_threshold = self.threshold * self.temporal_variance_multiplier

                if variance > temporal_threshold:
                    # Normalize variance to [0,1] range for confidence score
                    normalized_confidence = float(min(variance / (temporal_threshold * 2), 1.0))

                    conflict = ConflictState(
                        conflict_type=ConflictType.TEMPORAL_INCONSISTENCY,
                        pathway_ids=[pathway_id],
                        input_state=input_state,
                        conflicting_outputs={pathway_id: recent_outputs[-1]},
                        confidence_score=normalized_confidence,
                        environmental_context={
                            "temporal_variance": float(variance),
                            "variance_threshold": temporal_threshold,
                            "window_size": self.temporal_window
                        }
                    )
                    conflicts.append(conflict)

        return conflicts

    def _calculate_contradiction_score(self, output_a: Any, output_b: Any) -> float:
        """Calculate how contradictory two outputs are."""
        # Convert outputs to comparable format
        float_a = self._output_to_float(output_a)
        float_b = self._output_to_float(output_b)

        # For Putnam's baby example: high contradiction when both pathways are strongly activated
        # but have different directional preferences
        avg_activation = (float_a + float_b) / 2.0
        difference = abs(float_a - float_b)

        # High contradiction = both highly activated but different outputs
        if avg_activation > 0.6:  # Both pathways strongly activated
            return min(difference * 2.0, 1.0)  # Amplify differences when both are active
        else:
            # Standard contradiction measure for other cases
            return difference / max(avg_activation + 0.1, 1.0)

    def _output_to_float(self, output: Any) -> float:
        """Convert various output types to float for comparison."""
        if isinstance(output, (int, float)):
            return float(output)
        elif isinstance(output, torch.Tensor):
            return output.item() if output.numel() == 1 else float(torch.mean(output))
        elif isinstance(output, np.ndarray):
            return float(np.mean(output))
        else:
            # Hash-based conversion for other types
            return float(hash(str(output)) % 1000) / 1000.0

    def reset(self) -> None:
        """Reset conflict history and pathway outputs for fresh analysis."""
        self.conflict_history.clear()
        self.pathway_outputs.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about detected conflicts."""
        if not self.conflict_history:
            return {"total_conflicts": 0}

        conflict_types = [c.conflict_type for c in self.conflict_history]
        confidence_scores = [c.confidence_score for c in self.conflict_history]

        type_counts = {}
        for conflict_type in ConflictType:
            type_counts[conflict_type.value] = conflict_types.count(conflict_type)

        return {
            "total_conflicts": len(self.conflict_history),
            "average_confidence": float(np.mean(confidence_scores)),
            "max_confidence": float(np.max(confidence_scores)),
            "min_confidence": float(np.min(confidence_scores)),
            "conflict_type_distribution": type_counts,
            "pathways_tracked": len(self.pathway_outputs),
            "memory_usage": {
                "conflict_history_length": len(self.conflict_history),
                "max_allowed": self.max_history_length
            }
        }

    def adjust_threshold(self, new_threshold: float) -> None:
        """Dynamically adjust the conflict detection threshold."""
        self.threshold = max(0.01, min(1.0, new_threshold))

    def get_recent_conflicts(self, n: int = 10) -> List[ConflictState]:
        """Get the most recent n conflicts."""
        return self.conflict_history[-n:] if self.conflict_history else []
"""
Variable Space Explorer - Part of CDNE Core Engine

Implements Putnam's insight that contradictions are resolved by introducing
new variables from environmental or internal state space that can break
logical deadlocks between competing neural pathways.
"""


class VariableSpaceExplorer:
    """
    Searches for new variables that can resolve neural pathway contradictions.

    Based on Putnam's baby-turning-toward-warmth example: when conflicting rules
    "turn left" and "turn right" create deadlock, introducing warmth variable
    resolves contradiction into higher-order rule "turn toward warmer side".
    """

    def __init__(
        self,
        search_depth: int = 5,
        novelty_threshold: float = 0.3,
        max_variables_per_search: int = 10
    ):
        self.search_depth = search_depth
        self.novelty_threshold = novelty_threshold
        self.max_variables_per_search = max_variables_per_search
        self.discovered_variables: Dict[str, Any] = {}
        self.variable_effectiveness_history: Dict[str, List[float]] = {}

    def explore_resolution_space(
        self,
        conflict: ConflictState,
        current_variables: Dict[str, Any],
        environmental_sensors: Optional[Dict[str, Any]] = None
    ) -> List[ResolutionCandidate]:
        """
        Search for variables that could resolve the detected conflict.

        Args:
            conflict: The contradiction that needs resolution
            current_variables: Currently available variables in the system
            environmental_sensors: Available environmental information

        Returns:
            List of potential resolution strategies with new variables
        """
        resolution_candidates = []

        # Environmental variable discovery
        if environmental_sensors:
            env_candidates = self._explore_environmental_variables(
                conflict, environmental_sensors
            )
            resolution_candidates.extend(env_candidates)

        # Internal state variable discovery
        internal_candidates = self._explore_internal_variables(
            conflict, current_variables
        )
        resolution_candidates.extend(internal_candidates)

        # Cross-pathway variable synthesis
        synthesis_candidates = self._explore_variable_synthesis(
            conflict, current_variables
        )
        resolution_candidates.extend(synthesis_candidates)

        # Rank candidates by predicted effectiveness
        ranked_candidates = self._rank_resolution_candidates(
            resolution_candidates, conflict
        )

        return ranked_candidates[:self.max_variables_per_search]

    def _explore_environmental_variables(
        self,
        conflict: ConflictState,
        environmental_sensors: Dict[str, Any]
    ) -> List[ResolutionCandidate]:
        """Discover environmental variables that could resolve conflict."""
        candidates = []

        for sensor_name, sensor_data in environmental_sensors.items():
            # Check if this environmental variable correlates with conflict resolution
            correlation_strength = self._calculate_resolution_correlation(
                sensor_data, conflict
            )

            if correlation_strength > self.novelty_threshold:
                candidate = ResolutionCandidate(
                    strategy=ResolutionStrategy.VARIABLE_INTRODUCTION,
                    new_variables=[f"env_{sensor_name}"],
                    architectural_modifications={
                        "add_input_pathway": f"env_{sensor_name}",
                        "conflict_weighting": correlation_strength
                    },
                    predicted_effectiveness=correlation_strength,
                    computational_cost=0.1  # Environmental vars are cheap
                )
                candidates.append(candidate)

        return candidates
    def _explore_internal_variables(
        self,
        conflict: ConflictState,
        current_variables: Dict[str, Any]
    ) -> List[ResolutionCandidate]:
        """Discover internal state variables that could resolve conflict."""
        candidates = []

        # Analyze pathway activation patterns
        for pathway_id in conflict.pathway_ids:
            # Look for hidden internal states that differentiate successful vs failed activations
            internal_correlates = self._mine_internal_correlates(pathway_id, conflict)

            for correlate_name, correlate_strength in internal_correlates.items():
                if correlate_strength > self.novelty_threshold:
                    candidate = ResolutionCandidate(
                        strategy=ResolutionStrategy.VARIABLE_INTRODUCTION,
                        new_variables=[f"internal_{correlate_name}"],
                        architectural_modifications={
                            "add_internal_monitor": correlate_name,
                            "pathway_weighting": correlate_strength
                        },
                        predicted_effectiveness=correlate_strength,
                        computational_cost=0.3  # Internal monitoring has moderate cost
                    )
                    candidates.append(candidate)

        return candidates

    def _explore_variable_synthesis(
        self,
        conflict: ConflictState,
        current_variables: Dict[str, Any]
    ) -> List[ResolutionCandidate]:
        """Create new variables by combining existing ones."""
        candidates = []

        # Try combining pairs of existing variables
        variable_names = list(current_variables.keys())
        for i, var_a in enumerate(variable_names):
            for var_b in variable_names[i+1:]:
                # Create synthetic variable combining both
                synthesis_name = f"synth_{var_a}_{var_b}"
                effectiveness = self._predict_synthesis_effectiveness(
                    var_a, var_b, current_variables, conflict
                )

                if effectiveness > self.novelty_threshold:
                    candidate = ResolutionCandidate(
                        strategy=ResolutionStrategy.VARIABLE_INTRODUCTION,
                        new_variables=[synthesis_name],
                        architectural_modifications={
                            "synthesize_variables": [var_a, var_b],
                            "synthesis_function": "adaptive_combination"
                        },
                        predicted_effectiveness=effectiveness,
                        computational_cost=0.5  # Synthesis requires computation
                    )
                    candidates.append(candidate)

        return candidates
    def _calculate_resolution_correlation(
        self, sensor_data: Any, conflict: ConflictState
    ) -> float:
        """Calculate how well a sensor variable might resolve the conflict."""
        # Simplified correlation - in practice would be more sophisticated
        if isinstance(sensor_data, (int, float)):
            return min(abs(sensor_data) / 10.0, 1.0)
        else:
            return random.uniform(0.1, 0.8)  # Placeholder for complex analysis

    def _mine_internal_correlates(
        self, pathway_id: str, conflict: ConflictState
    ) -> Dict[str, float]:
        """Find internal state patterns that correlate with pathway success/failure."""
        # Placeholder - would analyze pathway activation history
        correlates = {
            f"{pathway_id}_activation_strength": random.uniform(0.2, 0.9),
            f"{pathway_id}_temporal_pattern": random.uniform(0.1, 0.7),
            f"{pathway_id}_resource_usage": random.uniform(0.3, 0.8)
        }
        return correlates

    def _predict_synthesis_effectiveness(
        self, var_a: str, var_b: str, variables: Dict[str, Any], conflict: ConflictState
    ) -> float:
        """Predict how effective combining two variables would be."""
        # Simplified prediction - would use learned models in practice
        return random.uniform(0.1, 0.9)

    def _rank_resolution_candidates(
        self, candidates: List[ResolutionCandidate], conflict: ConflictState
    ) -> List[ResolutionCandidate]:
        """Rank candidates by predicted effectiveness vs computational cost."""
        if not candidates:
            return []

        # Simple effectiveness/cost ratio ranking
        def ranking_score(candidate: ResolutionCandidate) -> float:
            return candidate.predicted_effectiveness / max(candidate.computational_cost, 0.01)

        return sorted(candidates, key=ranking_score, reverse=True)

    def update_effectiveness_history(self, variable_name: str, effectiveness: float) -> None:
        """Update the effectiveness history for a discovered variable."""
        if variable_name not in self.variable_effectiveness_history:
            self.variable_effectiveness_history[variable_name] = []
        self.variable_effectiveness_history[variable_name].append(effectiveness)

        # Limit history to prevent unbounded growth
        if len(self.variable_effectiveness_history[variable_name]) > 100:
            self.variable_effectiveness_history[variable_name] = self.variable_effectiveness_history[variable_name][-100:]

    def get_most_effective_variables(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get the most effective variables based on historical performance."""
        variable_scores = {}
        for var_name, history in self.variable_effectiveness_history.items():
            if history:
                variable_scores[var_name] = np.mean(history)

        return sorted(variable_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def clear_discovered_variables(self) -> None:
        """Clear the history of discovered variables."""
        self.discovered_variables.clear()
        self.variable_effectiveness_history.clear()
"""
Architectural Modifier - Final component of CDNE Core Engine

Implements the actual structural modifications to neural networks based on
Putnam's principle that intelligence emerges through architectural evolution
driven by contradiction resolution.
"""


class ArchitecturalModifier:
    """
    Modifies neural network architecture to resolve detected contradictions.

    Core implementation of Putnam's insight: when logical conflicts arise,
    the system must create new neural pathways that synthesize contradictions
    into higher-order rules rather than simply optimizing existing parameters.
    """

    def __init__(
        self,
        modification_rate: float = 0.1,
        structure_preservation_weight: float = 0.7,
        max_architectural_complexity: int = 1000
    ):
        self.modification_rate = modification_rate
        self.structure_preservation_weight = structure_preservation_weight
        self.max_architectural_complexity = max_architectural_complexity
        self.modification_history: List[Dict[str, Any]] = []

    def apply_resolution(
        self,
        network: nn.Module,
        conflict: ConflictState,
        resolution: ResolutionCandidate,
        genome: ArchitecturalGenome
    ) -> tuple[nn.Module, ArchitecturalGenome]:
        """
        Apply architectural modifications to resolve detected conflict.

        Args:
            network: Current neural network to modify
            conflict: The contradiction requiring resolution
            resolution: Chosen resolution strategy and modifications
            genome: Current architectural representation

        Returns:
            Modified network and updated genome
        """
        # For now, return the original network (no structural changes implemented yet)
        # and only update the genome to avoid deepcopy issues with PyTorch tensors
        modified_network = network

        # Create a shallow copy of the genome to avoid deepcopy issues
        updated_genome = genome.copy(deep=False)
        updated_genome.resolution_history = genome.resolution_history.copy()  # Shallow copy the list
        updated_genome.pathways = genome.pathways.copy()  # Shallow copy the dict
        updated_genome.connections = genome.connections.copy()  # Shallow copy the list
        updated_genome.variable_space = genome.variable_space.copy()  # Shallow copy the dict

        # Apply resolution based on strategy type
        if resolution.strategy == ResolutionStrategy.VARIABLE_INTRODUCTION:
            modified_network, updated_genome = self._introduce_variables(
                modified_network, resolution, updated_genome
            )

        elif resolution.strategy == ResolutionStrategy.PATHWAY_SYNTHESIS:
            modified_network, updated_genome = self._synthesize_pathways(
                modified_network, conflict, resolution, updated_genome
            )

        elif resolution.strategy == ResolutionStrategy.CONTEXT_PARTITIONING:
            modified_network, updated_genome = self._partition_contexts(
                modified_network, conflict, resolution, updated_genome
            )

        elif resolution.strategy == ResolutionStrategy.META_RULE_CREATION:
            modified_network, updated_genome = self._create_meta_rules(
                modified_network, conflict, resolution, updated_genome
            )

        # Update generation and history
        updated_genome.generation += 1
        updated_genome.resolution_history.append(resolution)

        # Log modification for analysis
        self.modification_history.append({
            "conflict_type": conflict.conflict_type,
            "resolution_strategy": resolution.strategy,
            "pathways_affected": conflict.pathway_ids,
            "architectural_changes": resolution.architectural_modifications,
            "generation": updated_genome.generation
        })

        return modified_network, updated_genome
    def _introduce_variables(
        self,
        network: nn.Module,
        resolution: ResolutionCandidate,
        genome: ArchitecturalGenome
    ) -> tuple[nn.Module, ArchitecturalGenome]:
        """Introduce new variables to break deadlock between pathways."""
        # Add new input pathways for the introduced variables
        for var_name in resolution.new_variables:
            # Create new pathway in genome
            new_pathway_id = f"pathway_{var_name}_{genome.generation}"
            genome.pathways[new_pathway_id] = {
                "variable_source": var_name,
                "pathway_type": "variable_input",
                "activation_function": "adaptive",
                "created_generation": genome.generation
            }

            # Add to variable space
            genome.variable_space[var_name] = {
                "introduced_for_conflict": True,
                "resolution_generation": genome.generation,
                "pathway_id": new_pathway_id
            }

        # Modify network architecture (simplified - would need more sophisticated implementation)
        # This is a conceptual placeholder for actual PyTorch network modification

        return network, genome

    def _synthesize_pathways(
        self,
        network: nn.Module,
        conflict: ConflictState,
        resolution: ResolutionCandidate,
        genome: ArchitecturalGenome
    ) -> tuple[nn.Module, ArchitecturalGenome]:
        """Combine conflicting pathways into higher-order synthesis rule."""
        # Create new synthesis pathway that combines the conflicting ones
        pathway_ids = conflict.pathway_ids
        synthesis_id = f"synthesis_{'+'.join(pathway_ids)}_{genome.generation}"

        genome.pathways[synthesis_id] = {
            "pathway_type": "synthesis",
            "source_pathways": pathway_ids,
            "synthesis_function": "contradiction_resolution",
            "created_generation": genome.generation,
            "resolution_variables": resolution.new_variables
        }

        # Create connections from original pathways to synthesis pathway
        for pathway_id in pathway_ids:
            connection = (pathway_id, synthesis_id, 0.8)  # High weight for synthesis
            genome.connections.append(connection)

        return network, genome
    def _partition_contexts(
        self,
        network: nn.Module,
        conflict: ConflictState,
        resolution: ResolutionCandidate,
        genome: ArchitecturalGenome
    ) -> tuple[nn.Module, ArchitecturalGenome]:
        """Create context-specific rules for different situations."""
        # Create context partitioning pathway
        partition_id = f"context_partition_{genome.generation}"

        genome.pathways[partition_id] = {
            "pathway_type": "context_partition",
            "conflicting_pathways": conflict.pathway_ids,
            "partition_variables": resolution.new_variables,
            "created_generation": genome.generation
        }

        # Create context-specific sub-pathways
        for i, pathway_id in enumerate(conflict.pathway_ids):
            context_specific_id = f"context_{i}_{pathway_id}_{genome.generation}"
            genome.pathways[context_specific_id] = {
                "pathway_type": "context_specific",
                "parent_pathway": pathway_id,
                "context_conditions": f"context_{i}_active",
                "created_generation": genome.generation
            }

        return network, genome

    def _create_meta_rules(
        self,
        network: nn.Module,
        conflict: ConflictState,
        resolution: ResolutionCandidate,
        genome: ArchitecturalGenome
    ) -> tuple[nn.Module, ArchitecturalGenome]:
        """Create rules that govern when other rules apply."""
        # Create meta-rule pathway
        meta_rule_id = f"meta_rule_{genome.generation}"

        genome.pathways[meta_rule_id] = {
            "pathway_type": "meta_rule",
            "governed_pathways": conflict.pathway_ids,
            "rule_conditions": resolution.new_variables,
            "decision_logic": "adaptive_selection",
            "created_generation": genome.generation
        }

        # Connect meta-rule to governed pathways with inhibitory connections
        for pathway_id in conflict.pathway_ids:
            # Meta-rule controls when each pathway is active
            control_connection = (meta_rule_id, pathway_id, -0.5)  # Inhibitory
            genome.connections.append(control_connection)

        return network, genome
