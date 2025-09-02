"""
Enhanced CDNE Network - Phase 1 Implementation

Integrates ConnectionMatrix and SynthesisEngine to enable true
architectural transcendence of contradictions through meaningful
connection formation and pathway synthesis.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from collections import defaultdict

from .enhanced_modifier import ConnectionMatrix, SynthesisEngine
from .core import ConflictDetectionEngine, VariableSpaceExplorer, ArchitecturalModifier
from .utils import (
    ConflictType, ResolutionStrategy, ConflictState, 
    ResolutionCandidate, ArchitecturalGenome
)


class EnhancedCDNENetwork(nn.Module):
    """
    Phase 1 Enhanced CDNE Network with true connection integration.
    
    Addresses architectural bottlenecks identified in analysis:
    - Meaningful connection formation between evolved pathways
    - True pathway synthesis creating higher-order rules
    - Connection-mediated conflict resolution
    """
    
    def __init__(
        self,
        input_size: int,
        initial_pathways: Dict[str, nn.Module],
        environmental_sensors: Optional[Dict[str, Any]] = None,
        evolution_rate: float = 0.1,
        connection_strength: float = 0.2
    ):
        super().__init__()
        
        # Enhanced core components
        self.conflict_detector = ConflictDetectionEngine(threshold=0.15)
        self.variable_explorer = VariableSpaceExplorer(search_depth=5)
        self.arch_modifier = ArchitecturalModifier(modification_rate=evolution_rate)
        
        # Phase 1 enhancements
        self.connection_matrix = ConnectionMatrix(
            list(initial_pathways.keys()), 
            connection_strength
        )
        self.synthesis_engine = SynthesisEngine(hidden_size=64)
        
        # Network components
        self.pathways = nn.ModuleDict(initial_pathways)
        self.environmental_sensors = environmental_sensors or {}
        self.current_variables = {}
        self.resolution_variables = nn.ModuleDict()  # For discovered variables
        
        # Enhanced genome tracking
        self.genome = ArchitecturalGenome(
            pathways={name: {"type": "initial", "generation": 0} 
                     for name in initial_pathways.keys()},
            connections=[(p1, p2, connection_strength) 
                        for p1 in initial_pathways.keys() 
                        for p2 in initial_pathways.keys() if p1 != p2],
            variable_space={},
            resolution_history=[],
            generation=0
        )
        
        # Phase 1 tracking
        self.synthesis_history = []
        self.connection_effectiveness = defaultdict(list)
        
    def forward(
        self, 
        x: torch.Tensor, 
        environmental_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Enhanced forward pass with connection-mediated conflict resolution.
        
        Phase 1 enhancements:
        - Pathway outputs mediated through connection matrix
        - Synthesis pathways activated for conflict resolution
        - Resolution variables integrated into computation
        """
        # Get raw outputs from all pathways
        raw_pathway_outputs = {}
        for name, pathway in self.pathways.items():
            try:
                raw_pathway_outputs[name] = pathway(x)
            except Exception:
                raw_pathway_outputs[name] = torch.zeros(x.size(0), 1)
                
        # Apply connection matrix to create connected outputs
        connected_outputs = self.connection_matrix(raw_pathway_outputs)
        
        # Integrate resolution variables if any exist
        enhanced_outputs = self._integrate_resolution_variables(
            connected_outputs, x, environmental_context
        )
        
        # Convert to input state for conflict detection
        input_state = {f"input_{i}": float(x[0, i]) for i in range(x.size(1))}
        
        # Detect conflicts in enhanced outputs
        conflicts = self.conflict_detector.detect_conflicts(
            enhanced_outputs, input_state, environmental_context
        )
        
        # Resolve conflicts through enhanced mechanisms
        if conflicts:
            synthesis_outputs = self._resolve_conflicts_with_synthesis(
                conflicts, enhanced_outputs, x, environmental_context
            )
            # Merge synthesis outputs with existing outputs
            enhanced_outputs.update(synthesis_outputs)
            
        # Final output integration
        final_output = self._integrate_final_output(enhanced_outputs)
        
        # Enhanced metadata
        metadata = {
            "conflicts_detected": len(conflicts),
            "pathways_active": len(self.pathways),
            "connections_active": len(self.genome.connections),
            "synthesis_pathways": len(self.synthesis_engine.synthesis_networks),
            "generation": self.genome.generation,
            "resolution_variables": len(self.resolution_variables)
        }
        
        return final_output, metadata
    def _integrate_resolution_variables(
        self,
        pathway_outputs: Dict[str, torch.Tensor],
        original_input: torch.Tensor,
        environmental_context: Optional[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """Integrate discovered resolution variables into computation."""
        enhanced_outputs = pathway_outputs.copy()
        
        # Apply resolution variables if any exist
        for var_name, var_module in self.resolution_variables.items():
            try:
                # Resolution variables take original input + environmental context
                var_input = original_input
                if environmental_context:
                    # Simple environmental integration - could be more sophisticated
                    env_tensor = torch.tensor([[float(v) for v in environmental_context.values()]], 
                                            dtype=torch.float32)
                    var_input = torch.cat([var_input, env_tensor], dim=1)
                    
                var_output = var_module(var_input)
                enhanced_outputs[var_name] = var_output
                
            except Exception as e:
                print(f"Warning: Resolution variable {var_name} failed: {e}")
                enhanced_outputs[var_name] = torch.zeros(1, 1)
                
        return enhanced_outputs
        
    def _resolve_conflicts_with_synthesis(
        self,
        conflicts: List[ConflictState],
        current_outputs: Dict[str, torch.Tensor],
        original_input: torch.Tensor,
        environmental_context: Optional[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """Resolve conflicts through pathway synthesis and variable introduction."""
        synthesis_outputs = {}
        
        for conflict in conflicts:
            # Explore resolution space
            resolution_candidates = self.variable_explorer.explore_resolution_space(
                conflict, self.current_variables, environmental_context
            )
            
            if resolution_candidates:
                best_resolution = resolution_candidates[0]
                
                if best_resolution.strategy == ResolutionStrategy.PATHWAY_SYNTHESIS:
                    synthesis_output = self._create_synthesis_pathway(
                        conflict, best_resolution, current_outputs, original_input
                    )
                    if synthesis_output is not None:
                        synthesis_outputs[f"synthesis_{self.genome.generation}"] = synthesis_output
                        
                elif best_resolution.strategy == ResolutionStrategy.VARIABLE_INTRODUCTION:
                    self._introduce_resolution_variable(best_resolution, original_input.size(1))
                    
                # Update genome
                self.genome.resolution_history.append(best_resolution)
                self.genome.generation += 1
                
        return synthesis_outputs
        
    def _create_synthesis_pathway(
        self,
        conflict: ConflictState,
        resolution: ResolutionCandidate,
        current_outputs: Dict[str, torch.Tensor],
        original_input: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Create new synthesis pathway for conflict resolution."""
        synthesis_id = f"synthesis_{conflict.pathway_ids[0]}_{conflict.pathway_ids[1]}_{self.genome.generation}"
        
        # Get conflicting outputs
        conflicting_outputs = {pid: current_outputs.get(pid, torch.zeros(1, 1)) 
                             for pid in conflict.pathway_ids}
        
        # Create synthesis pathway
        synthesis_network = self.synthesis_engine.create_synthesis_pathway(
            conflict.pathway_ids,
            synthesis_id,
            original_input.size(1),
            resolution.new_variables
        )
        
        # Add to pathways
        self.pathways[synthesis_id] = synthesis_network
        
        # Update connection matrix
        self.connection_matrix.add_pathway(synthesis_id)
        self.connection_matrix.create_synthesis_connection(
            conflict.pathway_ids, 
            synthesis_id,
            {"weight": 0.8}  # Strong synthesis connection
        )
        
        # Update genome
        self.genome.pathways[synthesis_id] = {
            "type": "synthesis",
            "source_pathways": conflict.pathway_ids,
            "generation": self.genome.generation,
            "resolution_strategy": resolution.strategy
        }
        
        # Generate synthesis output
        resolution_vars = {}  # Could be populated from discovered variables
        synthesis_output = self.synthesis_engine(
            synthesis_id, conflicting_outputs, resolution_vars, original_input
        )
        
        # Log synthesis event
        self.synthesis_history.append({
            "synthesis_id": synthesis_id,
            "conflict_type": conflict.conflict_type,
            "source_pathways": conflict.pathway_ids,
            "generation": self.genome.generation
        })
        
        return synthesis_output
        
    def _introduce_resolution_variable(self, resolution: ResolutionCandidate, input_size: int):
        """Create new resolution variable pathway."""
        for var_name in resolution.new_variables:
            # Create simple resolution variable network
            var_network = nn.Sequential(
                nn.Linear(input_size + 2, 16),  # Extra inputs for environmental context
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 1),
                nn.Tanh()  # Resolution variables can be positive or negative
            )
            
            self.resolution_variables[var_name] = var_network
            
            # Update genome variable space
            self.genome.variable_space[var_name] = {
                "type": "resolution_variable",
                "generation": self.genome.generation,
                "resolution_strategy": resolution.strategy
            }
            
    def _integrate_final_output(self, all_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Integrate all pathway outputs into final result."""
        if not all_outputs:
            return torch.zeros(1, 1)
            
        # Weighted integration based on pathway types and generations
        total_output = torch.zeros_like(list(all_outputs.values())[0])
        total_weight = 0.0
        
        for pathway_id, output in all_outputs.items():
            # Weight newer pathways more heavily (they resolved conflicts)
            pathway_info = self.genome.pathways.get(pathway_id, {"generation": 0})
            generation = pathway_info.get("generation", 0)
            
            # Synthesis pathways get higher weight
            if pathway_info.get("type") == "synthesis":
                weight = 1.0 + (generation * 0.2)
            else:
                weight = 0.8 + (generation * 0.1)
                
            total_output += output * weight
            total_weight += weight
            
        return total_output / max(total_weight, 1.0)
