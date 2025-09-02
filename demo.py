"""
CDNE Framework Demonstration

Practical example showing how Peter Putnam's theoretical insights about
brain function translate into working AI systems through contradiction-driven
neural evolution.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import logging
from putnamian_ai import CDNENetwork


# Configure logging for better debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PathwayConfig:
    """Configuration for pathway initialization."""
    hunger_weight: float = 5.0
    hunger_bias: float = -2.0
    input_size: int = 3
    hidden_size: int = 10


class PathwayFactory:
    """Factory for creating and configuring neural pathways."""

    @staticmethod
    def create_contradictory_pathway(config: PathwayConfig, direction: str = "left") -> nn.Sequential:
        """
        Create a pathway that activates on hunger but has directional preference.

        Args:
            config: Configuration parameters for the pathway
            direction: "left" or "right" to create opposing preferences

        Returns:
            Configured neural network pathway
        """
        pathway = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )

        # Initialize weights to create directional preference
        try:
            with torch.no_grad():
                # Get the first linear layer
                first_layer = pathway[0]
                if isinstance(first_layer, nn.Linear):
                    # Strong hunger activation
                    first_layer.weight.data[0, 0] = config.hunger_weight  # hunger input
                    first_layer.bias.data[0] = config.hunger_bias

                    # Create directional preference for warmth
                    if direction == "left":
                        # Prefers left warmth, dislikes right warmth
                        first_layer.weight.data[0, 1] = 2.0   # left warmth positive
                        first_layer.weight.data[0, 2] = -1.0  # right warmth negative
                    else:  # right
                        # Prefers right warmth, dislikes left warmth
                        first_layer.weight.data[0, 1] = -1.0  # left warmth negative
                        first_layer.weight.data[0, 2] = 2.0   # right warmth positive

                    # Add small noise for robustness
                    PathwayFactory._add_initialization_noise(pathway, config)

        except Exception as e:
            logger.error(f"Failed to initialize pathway: {e}")
            raise

        return pathway

    @staticmethod
    def _add_initialization_noise(pathway: nn.Sequential, config: PathwayConfig) -> None:
        """Add small random noise to prevent deterministic behavior."""
        noise_scale = config.hunger_weight * 0.01
        # Get the first linear layer
        first_layer = pathway[0]
        if isinstance(first_layer, nn.Linear) and first_layer.weight is not None:
            path_weights = first_layer.weight.data[0, 1:]  # Non-hunger weights
            noise = torch.randn_like(path_weights) * noise_scale
            first_layer.weight.data[0, 1:] += noise


@dataclass
class ScenarioConfig:
    """Configuration for environment scenarios."""
    hunger_level: float
    left_warmth: float
    right_warmth: float
    label: str


class ExperimentRunner:
    """Handles running experiments and analyzing results."""

    def __init__(self):
        self.results = []

    def run_scenario(self, brain: CDNENetwork, scenario: ScenarioConfig) -> Dict[str, Any]:
        """
        Run a single scenario and collect results.

        Args:
            brain: The neural network to test
            scenario: Configuration for this scenario

        Returns:
            Dictionary containing results and metadata
        """
        feedback_msg = f"Scenario: {scenario.label}"

        # Create input tensor
        input_tensor = torch.tensor([[
            scenario.hunger_level,
            scenario.left_warmth,
            scenario.right_warmth
        ]], dtype=torch.float32)

        # Environmental context
        env_context = {
            "left_warmth": scenario.left_warmth,
            "right_warmth": scenario.right_warmth
        }

        try:
            # Forward pass with error handling
            output, metadata = brain(input_tensor, env_context)

            result = {
                "label": scenario.label,
                "output": output.item(),
                "conflicts_detected": metadata['conflicts_detected'],
                "generation": metadata['generation'],
                "active_pathways": metadata['pathways_active'],
                "input_state": [
                    scenario.hunger_level,
                    scenario.left_warmth,
                    scenario.right_warmth
                ]
            }

            self.results.append(result)
            return result

        except Exception as e:
            logger.error(f"Failed to run {feedback_msg}: {e}")
            raise


def create_putnam_baby_scenario(
    pathway_config: PathwayConfig = PathwayConfig(),
    evolution_rate: float = 0.2
) -> CDNENetwork:
    """
    Recreation of Putnam's baby-turning-toward-warmth example.

    Two conflicting pathways: "turn_left" and "turn_right" create deadlock
    when baby is hungry. System must discover "warmth" variable to resolve
    contradiction into higher-order rule: "turn toward warmer side".

    Args:
        pathway_config: Configuration for pathway initialization
        evolution_rate: Rate at which the network evolves through conflict resolution

    Returns:
        Configured CDNENetwork instance
    """
    logger.info("Creating Putnam's baby scenario with contradictory pathways")

    # Create conflicting pathways using factory
    turn_left_pathway = PathwayFactory.create_contradictory_pathway(pathway_config, "left")
    turn_right_pathway = PathwayFactory.create_contradictory_pathway(pathway_config, "right")

    pathways = {
        "turn_left": turn_left_pathway,
        "turn_right": turn_right_pathway
    }

    # Environmental sensors (warmth detection) with validation
    environmental_sensors = {
        "left_warmth": lambda: np.clip(np.random.uniform(0.2, 0.8), 0.0, 1.0),
        "right_warmth": lambda: np.clip(np.random.uniform(0.2, 0.8), 0.0, 1.0)
    }

    # Validate pathways before creating network
    try:
        network = CDNENetwork(
            input_size=pathway_config.input_size,
            initial_pathways=pathways,
            environmental_sensors=environmental_sensors,
            evolution_rate=evolution_rate
        )
        logger.info(f"Successfully created CDNE network with {len(pathways)} pathways")
        return network

    except Exception as e:
        logger.error(f"Failed to create CDNENetwork: {e}")
        raise


def run_putnam_demonstration(
    pathway_config: PathwayConfig = PathwayConfig(),
    scenarios: Optional[List[ScenarioConfig]] = None
) -> Tuple[CDNENetwork, List[Dict[str, Any]]]:
    """
    Run the baby-warmth scenario demonstrating contradiction-driven evolution.

    Args:
        pathway_config: Configuration for pathway initialization
        scenarios: List of scenarios to run. If None, uses default scenarios.

    Returns:
        The evolved neural network
    """
    print("=== CDNE Framework: Putnam's Baby Scenario ===\n")

    # Use default scenarios if none provided
    if scenarios is None:
        scenarios = [
            ScenarioConfig(
                hunger_level=0.9,
                left_warmth=0.3,
                right_warmth=0.7,
                label="Right side warmer"
            ),
            ScenarioConfig(
                hunger_level=0.8,
                left_warmth=0.8,
                right_warmth=0.2,
                label="Left side warmer"
            ),
            ScenarioConfig(
                hunger_level=0.9,
                left_warmth=0.5,
                right_warmth=0.5,
                label="Equal warmth"
            ),
        ]

    # Create and configure the scenario
    baby_brain = create_putnam_baby_scenario(pathway_config)

    print("Initial State:")
    print(f"- Pathways: {list(baby_brain.pathways.keys())}")
    print(f"- Generation: {baby_brain.genome.generation}")
    print(f"- Conflicts resolved: {getattr(baby_brain, 'conflict_resolution_count', 0)}\n")

    # Initialize experiment runner
    runner = ExperimentRunner()

    print("Running scenarios to trigger contradictions...\n")

    for i, scenario in enumerate(scenarios):
        print(f"Scenario {i+1}: {scenario.label}")

        try:
            result = runner.run_scenario(baby_brain, scenario)

            print(f"  Output: {result['output']:.3f}")
            print(f"  Conflicts detected: {result['conflicts_detected']}")
            print(f"  Current generation: {result['generation']}")
            print(f"  Active pathways: {result['active_pathways']}")
            print()

        except Exception as e:
            print(f"  Error: {e}\n")
            continue

    # Show evolution summary
    try:
        summary = baby_brain.get_evolution_summary()
        print("=== Evolution Summary ===")
        for key, value in summary.items():
            print(f"{key}: {value}")

    except Exception as e:
        logger.error(f"Failed to get evolution summary: {e}")
        print("=== Evolution Summary ===")
        print("Evolution summary unavailable")

    return baby_brain, runner.results


if __name__ == "__main__":
    # Configure demonstration parameters
    config = PathwayConfig(
        hunger_weight=4.0,  # Reduced from 5.0 for more subtle contradictions
        hunger_bias=-1.5,   # Adjusted for better convergence
        hidden_size=16      # Increased for more complex patterns
    )

    evolved_brain, experiment_results = run_putnam_demonstration(
        pathway_config=config,
        scenarios=None  # Use defaults
    )

    # Optional: Additional analysis
    try:
        # Create architecture visualization
        arch_data = evolved_brain.visualize_architecture()
        print(f"\nFinal architecture: {len(arch_data['nodes'])} nodes, {len(arch_data['edges'])} connections")

        # Save experiment results for further analysis
        print(f"Results saved for {len(experiment_results)} scenarios")

    except Exception as e:
        logger.error(f"Failed to analyze final architecture: {e}")
