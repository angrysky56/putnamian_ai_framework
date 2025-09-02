#!/usr/bin/env python3
"""
Demonstration of the Modern Neuroevolution System for CDNE
Shows the hybrid ES + Custom Evolutionary approach in action
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from putnamian_ai.hcdci.modern_neuroevolution import (
    HybridNeuroEvolutionSystem,
    EvolutionConfig,
    CDNEVariableSpaceExplorer,
    CDNEArchitecturalModifier
)
import torch


def create_sample_network():
    """Create a sample neural network for evolution"""
    return {
        'architecture': {
            'input_size': 10,
            'hidden_sizes': [20, 15],
            'output_size': 5,
            'activation': 'relu',
            'synthesis_nodes': 0,
            'monitors': 0,
            'contradiction_detectors': 0,
            'memory_components': 0
        },
        'weights': {
            'layer_0': torch.randn(20, 10),
            'layer_1': torch.randn(15, 20),
            'layer_2': torch.randn(5, 15)
        }
    }


def sample_fitness_function(genome):
    """Sample fitness function that rewards architectural complexity"""
    arch = genome.get('architecture', {})
    weights = genome.get('weights', {})

    # Base fitness from network size
    fitness = sum(arch.get('hidden_sizes', [0])) / 100.0

    # Bonus for CDNE components
    fitness += arch.get('synthesis_nodes', 0) * 0.1
    fitness += arch.get('monitors', 0) * 0.05
    fitness += arch.get('contradiction_detectors', 0) * 0.15
    fitness += arch.get('memory_components', 0) * 0.08

    # Weight complexity bonus
    weight_complexity = sum(w.numel() for w in weights.values()) / 1000.0
    fitness += weight_complexity

    return min(1.0, fitness)  # Cap at 1.0


def demonstrate_hybrid_system():
    """Demonstrate the hybrid neuroevolution system"""
    print("🚀 Demonstrating Hybrid Neuroevolution System")
    print("=" * 50)

    # Create system
    es_config = EvolutionConfig(population_size=50, mutation_rate=0.1)
    custom_config = EvolutionConfig(population_size=30, mutation_rate=0.15)

    hybrid_system = HybridNeuroEvolutionSystem(es_config, custom_config)

    # Sample contradiction
    contradiction = {
        'type': 'logical_inconsistency',
        'description': 'Network cannot handle conflicting inputs',
        'severity': 0.8,
        'required_components': ['synthesis_nodes', 'contradiction_detectors']
    }

    # Initialize
    base_network = create_sample_network()
    hybrid_system.initialize_for_contradiction(base_network, contradiction)

    print("Initial Network Architecture:")
    print(f"   Hidden layers: {base_network['architecture']['hidden_sizes']}")
    print(f"   CDNE components: {base_network['architecture']['synthesis_nodes']} synthesis, "
          f"{base_network['architecture']['contradiction_detectors']} detectors")

    # Evolve
    print("\n🔄 Evolving for contradiction resolution...")
    result = hybrid_system.evolve_for_resolution(sample_fitness_function, max_generations=5)

    print("\nEvolution Complete!")
    print(f"   Generations run: {result['generations_run']}")
    print(f"   Final fitness: {result['final_fitness']:.3f}")
    print(f"   Contradiction resolved: {result['contradiction_resolved']}")

    if result['best_solution']:
        best_arch = result['best_solution'].genome['architecture']
        print("\nBest Solution Architecture:")
        print(f"   Hidden layers: {best_arch['hidden_sizes']}")
        print(f"   Synthesis nodes: {best_arch['synthesis_nodes']}")
        print(f"   Contradiction detectors: {best_arch['contradiction_detectors']}")
        print(f"   Monitors: {best_arch['monitors']}")
        print(f"   Memory components: {best_arch['memory_components']}")

    # Evolution summary
    summary = hybrid_system.get_evolution_summary()
    print("\nEvolution Summary:")
    print(f"   Total generations: {summary['total_generations']}")
    print(f"   Phases used: {summary['phases_used']}")
    print(f"   Best fitness achieved: {summary['best_fitness_achieved']:.3f}")
    print(f"   Evolution efficiency: {summary['evolution_efficiency']:.3f}")


def demonstrate_variable_explorer():
    """Demonstrate the CDNE Variable Space Explorer"""
    print("\n🔍 Demonstrating CDNE Variable Space Explorer")
    print("=" * 50)

    explorer = CDNEVariableSpaceExplorer()

    contradiction = {
        'type': 'architectural_limitation',
        'description': 'Network lacks capacity for complex pattern synthesis',
        'severity': 0.9,
        'focus_areas': ['synthesis_nodes', 'memory_components']
    }

    base_network = create_sample_network()

    print("🔬 Exploring variable space for contradiction resolution...")
    result = explorer.explore_contradiction_resolution(
        base_network, contradiction, sample_fitness_function
    )

    print("\nExploration Complete!")
    print(f"   Space coverage: {result['variable_space_covered']:.1%}")
    print(f"   Resolution quality: {result['resolution_quality']:.3f}")

    if result['best_solution']:
        best_arch = result['best_solution'].genome['architecture']
        print("\nBest Exploration Result:")
        print(f"   Synthesis nodes: {best_arch['synthesis_nodes']}")
        print(f"   Memory components: {best_arch['memory_components']}")
        print(f"   Fitness: {result['best_solution'].fitness:.3f}")


def demonstrate_architectural_modifier():
    """Demonstrate the CDNE Architectural Modifier"""
    print("\n🏗️  Demonstrating CDNE Architectural Modifier")
    print("=" * 50)

    modifier = CDNEArchitecturalModifier()

    contradiction = {
        'type': 'monitoring_deficit',
        'description': 'Network needs internal monitoring for stability',
        'severity': 0.7,
        'target_components': ['monitors', 'contradiction_detectors']
    }

    base_network = create_sample_network()

    print("🔧 Modifying architecture for contradiction resolution...")
    result = modifier.modify_for_contradiction(
        base_network, contradiction, sample_fitness_function
    )

    print("\nModification Complete!")
    print(f"   Improvement score: {result['improvement_score']:.3f}")

    changes = result['architectural_changes']
    print("\nArchitectural Changes:")
    for key, change in changes.items():
        if isinstance(change, dict):
            print(f"   {key}: {change['from']} -> {change['to']}")
        else:
            print(f"   {key}: {change}")

    # Show final architecture
    final_arch = result['modified_network']['architecture']
    print("\nFinal Architecture:")
    print(f"   Monitors: {final_arch['monitors']}")
    print(f"   Contradiction detectors: {final_arch['contradiction_detectors']}")
    print(f"   Synthesis nodes: {final_arch['synthesis_nodes']}")
    print(f"   Memory components: {final_arch['memory_components']}")


def main():
    """Main demonstration function"""
    print("🧠 Putnamian AI Framework - Modern Neuroevolution Demo")
    print("=" * 60)
    print("This demo showcases the hybrid neuroevolution system that combines:")
    print("• Evolution Strategies (ES) for rapid exploration")
    print("• Custom Evolutionary Toolkit for precise CDNE modifications")
    print("• Contradiction-Driven Neural Evolution (CDNE) capabilities")
    print()

    try:
        demonstrate_hybrid_system()
        demonstrate_variable_explorer()
        demonstrate_architectural_modifier()

        print("\n🎉 All demonstrations completed successfully!")
        print("The modern neuroevolution system is ready for CDNE applications.")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
