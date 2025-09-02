"""
SLAP (Semantic Logic Auto-Progressor) Demonstration Script

This script demonstrates the capabilities of the Semantic Logic Auto-Progressor,
showcasing semantic parsing, logical inference, and automatic reasoning progression.
"""

from putnamian_ai.hcdci.semantic_logic_auto_progressor import (
    SemanticLogicAutoProgressor,
    SLAPIntegrationLayer,
    SemanticParser,
    LogicalInferenceEngine
)
from datetime import datetime


def demonstrate_semantic_parsing():
    """Demonstrate semantic parsing capabilities"""
    print("=" * 60)
    print("SEMANTIC PARSING DEMONSTRATION")
    print("=" * 60)

    parser = SemanticParser()

    # Test various types of input
    test_inputs = [
        "If it rains, then the ground gets wet",
        "Learning causes knowledge growth",
        "Wisdom is the ability to see patterns in complexity",
        "All humans are mortal and Socrates is a human",
        "The heart pumps blood and blood carries oxygen"
    ]

    for i, text in enumerate(test_inputs, 1):
        print(f"\nTest Input {i}: {text}")
        parsed = parser.parse_text(text)

        print(f"  Parsing Confidence: {parsed['parsing_confidence']:.2f}")
        print(f"  Concepts Found: {len(parsed['concepts'])}")
        print(f"  Propositions Found: {len(parsed['propositions'])}")
        print(f"  Relationships Found: {len(parsed['relationships'])}")

        if parsed['concepts']:
            print("  Concepts:")
            for concept in parsed['concepts']:
                print(f"    - {concept.name}: {concept.definition}")

        if parsed['propositions']:
            print("  Propositions:")
            for prop in parsed['propositions']:
                print(f"    - {prop.statement} (operators: {[op.value for op in prop.operators]})")

        if parsed['relationships']:
            print("  Relationships:")
            for subj, rel, obj in parsed['relationships']:
                print(f"    - {subj} {rel.value} {obj}")


def demonstrate_logical_inference():
    """Demonstrate logical inference capabilities"""
    print("\n" + "=" * 60)
    print("LOGICAL INFERENCE DEMONSTRATION")
    print("=" * 60)

    from putnamian_ai.hcdci.semantic_logic_auto_progressor import (
        LogicalProposition, LogicalOperator, SemanticConcept, SemanticRelation
    )

    inference_engine = LogicalInferenceEngine()

    # Create test propositions and concepts
    propositions = [
        LogicalProposition(
            statement="If it rains, then the ground gets wet",
            operators=[LogicalOperator.IMPLIES],
            confidence=0.9
        ),
        LogicalProposition(
            statement="It is raining",
            confidence=0.8
        ),
        LogicalProposition(
            statement="All humans are mortal",
            operators=[LogicalOperator.FOR_ALL],
            confidence=0.95
        ),
        LogicalProposition(
            statement="Socrates is human",
            confidence=0.9
        )
    ]

    concepts = [
        SemanticConcept(
            name="Learning",
            definition="The process of acquiring knowledge"
        ),
        SemanticConcept(
            name="Knowledge",
            definition="Information and understanding"
        )
    ]

    # Add relationship
    concepts[0].relationships[SemanticRelation.CAUSES].add("Knowledge")

    print("Input Propositions:")
    for prop in propositions:
        print(f"  - {prop.statement}")

    print("\nInput Concepts:")
    for concept in concepts:
        print(f"  - {concept.name}: {concept.definition}")
        for rel_type, related in concept.relationships.items():
            for rel in related:
                print(f"    {rel_type.value} {rel}")

    # Generate inferences
    inferences = inference_engine.generate_inferences(propositions, concepts)

    print(f"\nGenerated Inferences: {len(inferences)}")
    for i, inference in enumerate(inferences, 1):
        print(f"\nInference {i}:")
        print(f"  Rule Applied: {inference.rule_applied}")
        print(f"  Confidence: {inference.confidence:.2f}")
        print(f"  Conclusion: {inference.conclusion.statement}")
        print("  Reasoning Steps:")
        for step in inference.steps:
            print(f"    - {step}")


def demonstrate_auto_progression():
    """Demonstrate automatic reasoning progression"""
    print("\n" + "=" * 60)
    print("AUTO-PROGRESSION DEMONSTRATION")
    print("=" * 60)

    slap = SemanticLogicAutoProgressor()

    # Process a complex input that will trigger various progression strategies
    complex_input = """
    If someone studies diligently, then they gain knowledge.
    Knowledge leads to wisdom.
    All students should study diligently.
    John is a student.
    """

    print(f"Input Text: {complex_input.strip()}")

    result = slap.process_input(complex_input)

    print("\nProcessing Results:")
    print(f"  Original Input Length: {len(result['original_input'])} chars")
    print(f"  Concepts Integrated: {result['knowledge_integrated']}")
    print(f"  Inferences Generated: {len(result['generated_inferences'])}")
    print(f"  Progressions Made: {len(result['progressed_reasoning'])}")
    print(f"  Overall Confidence: {result['confidence_score']:.2f}")
    print(f"  Reasoning Chains: {result['reasoning_chains']}")

    print("\nGenerated Inferences:")
    for i, inference in enumerate(result['generated_inferences'], 1):
        print(f"  {i}. {inference.conclusion.statement} (confidence: {inference.confidence:.2f})")

    print("\nProgressed Reasoning:")
    for i, progression in enumerate(result['progressed_reasoning'], 1):
        if hasattr(progression, 'rule_applied') and progression.rule_applied != 'modus_ponens':  # Skip basic modus ponens for clarity
            print(f"  {i}. [{progression.rule_applied}] {progression.conclusion.statement}")

    # Show knowledge base growth
    summary = slap.get_reasoning_summary()
    print("\nKnowledge Base Summary:")
    print(f"  Concepts: {summary['knowledge_base_size']}")
    print(f"  Logical Propositions: {summary['logical_propositions']}")
    print(f"  Reasoning Chains: {summary['reasoning_chains']}")
    print(f"  Average Confidence: {summary['average_confidence']:.2f}")


def demonstrate_integration_layer():
    """Demonstrate integration with other HCDCI components"""
    print("\n" + "=" * 60)
    print("INTEGRATION LAYER DEMONSTRATION")
    print("=" * 60)

    integration_layer = SLAPIntegrationLayer()

    # Simulate persona context
    persona_context = {
        'name': 'Dr. Elena Vasquez',
        'expertise': 'cognitive science',
        'philosophical_framework': {
            'core_beliefs': ['Knowledge emerges from pattern recognition']
        },
        'linguistic_profile': {
            'vocabulary_preferences': ['cognitive', 'neural', 'pattern', 'emergent']
        }
    }

    input_text = "Learning involves changing neural connections in the brain"

    print(f"Input: {input_text}")
    print(f"Persona: {persona_context['name']} ({persona_context['expertise']})")

    # Process with persona context
    result = integration_layer.process_with_persona_context(input_text, persona_context)

    print("\nPersona-Enhanced Processing:")
    print(f"  Enhanced Input: {result['original_input'][:100]}...")
    print(f"  Inferences Generated: {len(result['generated_inferences'])}")
    print(f"  Persona Filtering Applied: {'Yes' if result.get('persona_enhanced') else 'No'}")

    # Show how persona context influenced processing
    if result.get('persona_enhanced'):
        enhanced = result['persona_enhanced']
        print(f"  Filtered Inferences: {len(enhanced.get('generated_inferences', []))}")

    # Demonstrate ethical filtering (placeholder)
    print("\nEthical Filtering (Simulated):")
    ethical_result = integration_layer.process_with_ethical_filter(input_text, None)
    print(f"  Ethical Check Passed: {len(ethical_result.get('ethically_filtered_inferences', []))} inferences approved")


def demonstrate_complex_reasoning_chain():
    """Demonstrate a complex multi-step reasoning chain"""
    print("\n" + "=" * 60)
    print("COMPLEX REASONING CHAIN DEMONSTRATION")
    print("=" * 60)

    slap = SemanticLogicAutoProgressor()

    # Build up a complex knowledge base step by step
    inputs = [
        "All mammals are animals",
        "Humans are mammals",
        "Animals need oxygen to survive",
        "If an organism needs oxygen, then it performs respiration",
        "Humans perform respiration",
        "Respiration involves gas exchange in lungs",
        "Lungs are organs specialized for breathing"
    ]

    print("Building Knowledge Base:")
    for i, input_text in enumerate(inputs, 1):
        print(f"  {i}. {input_text}")
        slap.process_input(input_text)

    # Now test a complex inference
    test_input = "Humans have lungs"
    print(f"\nTest Inference: {test_input}")

    result = slap.process_input(test_input)

    print("\nReasoning Results:")
    print(f"  Knowledge Base Size: {slap.get_reasoning_summary()['knowledge_base_size']}")
    print(f"  Logical Propositions: {slap.get_reasoning_summary()['logical_propositions']}")
    print(f"  Inferences Generated: {len(result['generated_inferences'])}")

    # Show the most confident inferences
    confident_inferences = [inf for inf in result['generated_inferences'] if inf.confidence > 0.7]
    print(f"  High-Confidence Inferences: {len(confident_inferences)}")

    for inference in confident_inferences[:3]:  # Show top 3
        print(f"    - {inference.conclusion.statement} (confidence: {inference.confidence:.2f})")


def main():
    """Run all demonstrations"""
    print("SEMANTIC LOGIC AUTO-PROGRESSOR (SLAP) SYSTEM")
    print("Demonstrating advanced semantic processing and logical reasoning")
    print(f"Demonstration started at: {datetime.now()}")
    print()

    try:
        demonstrate_semantic_parsing()
        demonstrate_logical_inference()
        demonstrate_auto_progression()
        demonstrate_integration_layer()
        demonstrate_complex_reasoning_chain()

        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("Key Capabilities Demonstrated:")
        print("✓ Semantic parsing of natural language")
        print("✓ Logical inference using formal rules (modus ponens, etc.)")
        print("✓ Automatic reasoning progression (analogy, generalization, etc.)")
        print("✓ Knowledge base integration and growth")
        print("✓ Integration with HCDCI components (ethical constraints, SEG personas)")
        print("✓ Complex multi-step reasoning chains")
        print("✓ Confidence scoring and uncertainty handling")

    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
