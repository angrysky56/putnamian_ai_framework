"""
SEG Persona Demonstration Script

This script demonstrates the Simulated Experiential Grounding (SEG) persona system,
showcasing how personas create "simulated souls" with comprehensive identity architectures.
"""

from putnamian_ai.hcdci.seg_personas import SEGPersonaFactory, SEGOrchestrator
from datetime import datetime


def demonstrate_elara_vance():
    """Demonstrate Elara Vance persona with full sensory and emotional processing"""
    print("=" * 60)
    print("ELARA VANCE PERSONA DEMONSTRATION")
    print("=" * 60)

    # Create and examine the persona
    elara = SEGPersonaFactory.create_elara_vance()

    print(f"Identity: {elara.name}, {elara.age} years old")
    print(f"Profession: {elara.profession}")
    print(f"Location: {elara.location}")
    print()

    # Show sensory web
    print("SENSORY WEB:")
    for modality, memories in elara.sensory_web.items():
        print(f"  {modality.upper()}:")
        for memory in memories:
            print(f"    - {memory.description}")
            print(f"      Emotional valence: {memory.emotional_valence:+.1f}")
            print(f"      Triggers: {memory.context_triggers}")
    print()

    # Show emotional core
    print("EMOTIONAL CORE:")
    if elara.emotional_core:
        core = elara.emotional_core
        print(f"  Defining Event: {core.defining_event}")
        print(f"  Recurring Theme: {core.recurring_theme}")
        print(f"  Emotional Baseline: {core.emotional_baseline:+.1f}")
        print(f"  Resilience: {core.emotional_resilience:.1f}")
        print(f"  Vulnerability Points: {core.vulnerability_points}")
        print(f"  Coping Mechanisms: {core.coping_mechanisms}")
    print()

    # Show philosophical framework
    print("PHILOSOPHICAL FRAMEWORK:")
    if elara.philosophical_framework:
        phil = elara.philosophical_framework
        print("  Core Beliefs:")
        for belief in phil.core_beliefs:
            print(f"    - {belief}")
        print("  Life Heuristics:")
        for heuristic in phil.life_heuristics:
            print(f"    - {heuristic}")
        print("  Worldview Filters:")
        for key, filter_desc in phil.worldview_filters.items():
            print(f"    - {key}: {filter_desc}")
    print()

    # Show linguistic profile
    print("LINGUISTIC PROFILE:")
    if elara.linguistic_profile:
        ling = elara.linguistic_profile
        print(f"  Speech Cadence: {ling.speech_cadence}")
        print(f"  Common Phrases: {ling.common_phrases}")
        print(f"  Metaphors: {ling.metaphors}")
        print(f"  Vocabulary Preferences: {ling.vocabulary_preferences}")
    print()

    # Demonstrate activation and query processing
    print("INTERACTION DEMONSTRATION:")
    activation_msg = elara.activate_persona("Exploring themes of loss and memory in relationships")
    print(f"Activation: {activation_msg}")

    queries = [
        "How do memories shape our understanding of relationships?",
        "What happens when we lose someone we love?",
        "How can we find meaning in difficult experiences?"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        result = elara.process_query(query, {'emotional_context': 'contemplative'})
        print(f"Response: {result['response']}")
        print(f"Active substrates: {result['substrates_used']}")

    deactivation_msg = elara.deactivate_persona()
    print(f"\nDeactivation: {deactivation_msg}")


def demonstrate_multi_persona_orchestration():
    """Demonstrate orchestration of multiple personas for complex tasks"""
    print("\n" + "=" * 60)
    print("MULTI-PERSONA ORCHESTRATION DEMONSTRATION")
    print("=" * 60)

    orchestrator = SEGOrchestrator()

    # Register available personas
    elara = SEGPersonaFactory.create_elara_vance()
    orchestrator.register_persona('elara_vance', elara)

    # Note: In a full implementation, we would have multiple personas
    # For now, demonstrating with the available one

    complex_task = """
    Help someone navigate a major life transition involving:
    - Career change from a stable but unfulfilling job
    - Moving to a new city
    - Redefining personal identity and relationships
    - Dealing with uncertainty and fear of the unknown
    """

    query = "What are the key psychological and practical considerations for someone facing this transition?"

    print(f"Task Description: {complex_task.strip()}")
    print(f"Query: {query}")
    print()

    result = orchestrator.orchestrate_task(complex_task, query)

    print("ORCHESTRATION RESULTS:")
    print(f"Active Personas: {result['active_personas']}")
    print(f"Task Requirements: {result['task_requirements']}")
    print()

    print("INDIVIDUAL PERSPECTIVES:")
    for persona_id, perspective in result['individual_perspectives'].items():
        print(f"\n{persona_id.upper()}:")
        print(f"  Response: {perspective['response']}")
        print(f"  Consistency: {perspective['consistency_score']}")

    print("\nSYNTHESIZED RESPONSE:")
    print(result['synthesized_response'])


def demonstrate_adaptive_evolution():
    """Demonstrate how personas can evolve based on interactions"""
    print("\n" + "=" * 60)
    print("ADAPTIVE EVOLUTION DEMONSTRATION")
    print("=" * 60)

    elara = SEGPersonaFactory.create_elara_vance()

    print("Initial State:")
    print(f"Consistency Score: {elara.consistency_score}")
    print(f"Experience History Length: {len(elara.experience_history)}")
    print()

    # Simulate multiple interactions
    interactions = [
        {
            'query': 'How do you deal with uncertainty in mapping expeditions?',
            'context': {'task_type': 'professional', 'emotional_tone': 'curious'},
            'feedback': {'effectiveness': 0.9, 'linguistic_feedback': {'cadence': 'good'}}
        },
        {
            'query': 'Tell me about losing someone important to you',
            'context': {'task_type': 'personal', 'emotional_tone': 'vulnerable'},
            'feedback': {'effectiveness': 0.7, 'new_experiences': [{'type': 'emotional_sharing'}]}
        },
        {
            'query': 'What does it mean to truly know a place?',
            'context': {'task_type': 'philosophical', 'emotional_tone': 'reflective'},
            'feedback': {'effectiveness': 0.95, 'linguistic_feedback': {'metaphors': 'effective'}}
        }
    ]

    elara.activate_persona("Building deeper understanding through conversation")

    for i, interaction in enumerate(interactions, 1):
        print(f"Interaction {i}:")
        print(f"  Query: {interaction['query']}")

        result = elara.process_query(
            interaction['query'],
            interaction['context']
        )

        print(f"  Response: {result['response']}")
        print(f"  Active Substrates: {result['substrates_used']}")

        # Apply evolution feedback
        elara.evolve_persona(interaction['feedback'])
        print(f"  Updated Consistency Score: {elara.consistency_score:.2f}")
        print()

    print("Evolution Summary:")
    print(f"Total Interactions: {len(elara.experience_history)}")
    print(f"Final Consistency Score: {elara.consistency_score}")
    print(f"Experience History Captured: {len([exp for exp in elara.experience_history if exp])} events")

    elara.deactivate_persona()


def main():
    """Run all demonstrations"""
    print("SIMULATED EXPERIENTIAL GROUNDING (SEG) PERSONA SYSTEM")
    print("Demonstrating adaptive persona construction for AI interaction")
    print(f"Demonstration started at: {datetime.now()}")
    print()

    try:
        demonstrate_elara_vance()
        demonstrate_multi_persona_orchestration()
        demonstrate_adaptive_evolution()

        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("Key Features Demonstrated:")
        print("✓ Identity Architecture with sensory, emotional, and philosophical grounding")
        print("✓ Dynamic substrate activation based on task requirements")
        print("✓ Multi-persona orchestration for complex tasks")
        print("✓ Adaptive evolution through interaction feedback")
        print("✓ Consistent linguistic and behavioral patterns")
        print("✓ Meta-awareness and persona state management")

    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
