"""
Enhanced Architectural Self-Modifier - Phase 1 Implementation

Addresses Ty's critique about connection integration and pathway synthesis.
Creates meaningful connections between evolved pathways and enables
true architectural transcendence of contradictions.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
class ConnectionMatrix(nn.Module):
    """
    Dynamic connection matrix that evolves with the architecture.

    Implements Putnam's insight that new neural pathways must form
    meaningful connections to resolve contradictions rather than
    simply adding isolated capabilities.
    """

    def __init__(self, initial_pathways: List[str], connection_strength: float = 0.1):
        super().__init__()
        self.pathway_names = initial_pathways.copy()
        self.connection_strength = connection_strength

        # Dynamic connection matrix - grows as pathways are added
        n_pathways = len(initial_pathways)
        self.connections = nn.Parameter(
            torch.zeros(n_pathways, n_pathways) + connection_strength
        )

        # Connection effectiveness tracking
        self.connection_history = {}
        self.pruning_threshold = 0.05

    def add_pathway(self, new_pathway_id: str) -> None:
        """Add new pathway to connection matrix."""
        if new_pathway_id in self.pathway_names:
            print(f"Warning: Pathway {new_pathway_id} already exists")
            return

        self.pathway_names.append(new_pathway_id)
        old_size = self.connections.size(0)
        new_size = old_size + 1

        # Expand connection matrix
        new_connections = torch.zeros(new_size, new_size)
        if old_size > 0:
            new_connections[:old_size, :old_size] = self.connections.data

        # Initialize new connections with small random weights
        if old_size > 0:
            new_connections[old_size, :old_size] = torch.randn(old_size) * 0.1
            new_connections[:old_size, old_size] = torch.randn(old_size) * 0.1

        self.connections = nn.Parameter(new_connections)

    def create_synthesis_connection(
        self,
        source_pathways: List[str],
        synthesis_pathway: str,
        synthesis_weights: Optional[Dict[str, float]] = None
    ) -> None:
        """Create weighted connections for pathway synthesis."""
        try:
            if synthesis_pathway not in self.pathway_names:
                print(f"Warning: Synthesis pathway {synthesis_pathway} not found")
                return

            synthesis_idx = self.pathway_names.index(synthesis_pathway)

            for source_pathway in source_pathways:
                if source_pathway in self.pathway_names:
                    source_idx = self.pathway_names.index(source_pathway)

                    # Weight based on synthesis requirements
                    weight = synthesis_weights.get(source_pathway, 0.5) if synthesis_weights else 0.5

                    # Bidirectional connection for synthesis
                    self.connections.data[source_idx, synthesis_idx] = weight
                    self.connections.data[synthesis_idx, source_idx] = weight * 0.8
                else:
                    print(f"Warning: Source pathway {source_pathway} not found")

        except Exception as e:
            print(f"Warning: Error creating synthesis connection: {e}")

    def forward(self, pathway_outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply connection matrix to pathway outputs."""
        if not pathway_outputs:
            return {}

        # Get the batch size from the first output
        first_output = next(iter(pathway_outputs.values()))
        batch_size = first_output.size(0)

        # Convert pathway outputs to tensor format
        output_tensor = torch.zeros(len(self.pathway_names), batch_size)

        for i, pathway_name in enumerate(self.pathway_names):
            if pathway_name in pathway_outputs:
                output = pathway_outputs[pathway_name]
                # Ensure output is the right shape
                if output.dim() > 1:
                    output = output.mean(dim=-1)  # Average across feature dimensions
                output_tensor[i] = output.flatten()[:batch_size]  # Take first batch_size elements

        # Apply connection matrix
        connected_outputs = torch.matmul(self.connections, output_tensor)

        # Convert back to dictionary format
        result = {}
        for i, pathway_name in enumerate(self.pathway_names):
            result[pathway_name] = connected_outputs[i].unsqueeze(0)

        return result


class SynthesisEngine(nn.Module):
    """
    Creates higher-order rules by combining conflicting pathways.

    Implements Putnam's "bat" principle: transcending "mammal" and "bird"
    contradictions by creating new category that preserves useful elements
    from both while resolving their conflict.
    """

    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.hidden_size = hidden_size
        self.synthesis_networks = nn.ModuleDict()

    def create_synthesis_pathway(
        self,
        pathway_ids: List[str],
        synthesis_id: str,
        input_size: int,
        resolution_variables: List[str]
    ) -> nn.Module:
        """Create new pathway that synthesizes conflicting ones."""

        # Multi-input synthesis network
        # Takes inputs from conflicting pathways plus resolution variables
        total_inputs = len(pathway_ids) + len(resolution_variables) + input_size

        synthesis_network = nn.Sequential(
            nn.Linear(total_inputs, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1),
            nn.Sigmoid()
        )

        # Initialize with bias toward synthesis rather than conflict
        with torch.no_grad():
            # Encourage combination rather than dominance of one pathway
            first_layer = synthesis_network[0]
            third_layer = synthesis_network[2]
            if isinstance(first_layer, nn.Linear) and first_layer.bias is not None:
                first_layer.bias.fill_(0.1)
            if isinstance(third_layer, nn.Linear) and third_layer.bias is not None:
                third_layer.bias.fill_(0.05)

        self.synthesis_networks[synthesis_id] = synthesis_network
        return synthesis_network

    def forward(
        self,
        synthesis_id: str,
        conflicting_outputs: Dict[str, torch.Tensor],
        resolution_variables: Dict[str, torch.Tensor],
        original_input: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through synthesis pathway."""
        if synthesis_id not in self.synthesis_networks:
            return torch.zeros(1, 1)

        try:
            # Combine all inputs for synthesis
            inputs = [original_input]

            # Add conflicting pathway outputs
            for output in conflicting_outputs.values():
                inputs.append(output)

            # Add resolution variables
            for var in resolution_variables.values():
                inputs.append(var)

            # Ensure all inputs have the same batch size
            batch_size = inputs[0].size(0)
            processed_inputs = []

            for inp in inputs:
                if inp.dim() == 0:  # Scalar tensor
                    processed_inputs.append(inp.unsqueeze(0).unsqueeze(0))
                elif inp.dim() == 1:  # 1D tensor
                    processed_inputs.append(inp.unsqueeze(0))
                else:  # Multi-dimensional tensor
                    processed_inputs.append(inp.view(batch_size, -1))

            # Concatenate along feature dimension
            combined_input = torch.cat(processed_inputs, dim=1)

            return self.synthesis_networks[synthesis_id](combined_input)

        except Exception as e:
            print(f"Warning: Error in synthesis forward pass: {e}")
            return torch.zeros(1, 1)
