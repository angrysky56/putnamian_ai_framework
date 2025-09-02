#!/usr/bin/env python3
"""
Test script for the Adaptive Persona System.
Demonstrates how the enhanced SEG personas work with dynamic generation,
memory systems, and contextual adaptation.
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from putnamian_ai.hcdci.seg_personas import (
    AdaptivePersona,
    AdaptivePersonaFactory,
    AdaptivePersonaOrchestrator,
    PersonaLibrary,
    PERSONA_ARCHETYPES,
    AIService
)


class MockAIService(AIService):
    """Mock AI service for testing without external dependencies"""
    
    def __init__(self):
        super().__init__({"provider": "mock"})
        self.call_count = 0
    
    async def generate_response(self, messages, system_prompt=""):
        """Mock AI response generation"""
        self.call_count += 1
        user_message = messages[-1]["content"] if messages else ""
        
        # Simple keyword-based mock responses
        if "learning" in user_message.lower():
            return {
                "content": "Learning is a journey that requires both patience and curiosity. "
                          "Each step builds upon the last, creating a foundation for deeper understanding.",
                "error": None
            }
        elif "creative" in user_message.lower():
            return {
                "content": "Creativity flows from the intersection of knowledge and imagination. "
                          "Allow yourself to explore without judgment, letting ideas emerge naturally.",
                "error": None
            }
        elif "meaning" in user_message.lower():
            return {
                "content": "Meaning often reveals itself not in grand moments but in the quiet "
                          "spaces between thoughts, where understanding settles like morning dew.",
                "error": None
            }
        else:
            return {
                "content": "I find myself considering the layers beneath your question, "
                          "where deeper currents of understanding might flow.",
                "error": None
            }


async def test_basic_persona_functionality():
    """Test basic persona creation and interaction"""
    print("=== Test 1: Basic Persona Functionality ===")
    
    # Create factory without AI service (will use fallback generation)
    factory = AdaptivePersonaFactory()
    
    # Create a scholar persona
    scholar = await factory.create_scholar_persona("academic research context")
    print(f"Created persona: {scholar.name}, {scholar.age} years old, {scholar.profession}")
    print(f"Archetype: {scholar.archetype}")
    print(f"Core beliefs: {scholar.core_beliefs[0]}")
    print(f"Memory count: {len(scholar.memories)}")
    print()
    
    # Test activation and query processing
    scholar.activate_persona("Help with understanding complex topics")
    query = "How do you approach learning something new?"
    
    response = await scholar.process_query(query, {"domain": "education"})
    print(f"Query: {query}")
    print(f"Response: {response['response']}")
    print(f"Generation method: {response.get('generation_method', 'unknown')}")
    print(f"Used memories: {len(response.get('used_memory_ids', []))}")
    print()


async def test_memory_system():
    """Test the adaptive memory system"""
    print("=== Test 2: Memory System ===")
    
    factory = AdaptivePersonaFactory()
    mystic = await factory.create_mystic_persona("philosophical guidance")
    
    # Show initial memory stats
    initial_stats = mystic.get_memory_stats()
    print(f"Initial memories: {initial_stats['total_memories']}")
    print(f"Initial avg salience: {initial_stats['avg_salience']:.2f}")
    print(f"Subtlety ratio: {initial_stats['subtlety_ratio']:.2f}")
    print()
    
    # Activate and process multiple queries to build memory
    mystic.activate_persona("Philosophical discussion")
    
    queries = [
        "What is the meaning of existence?",
        "How do we find purpose in life?", 
        "What is the relationship between suffering and wisdom?"
    ]
    
    for i, query in enumerate(queries):
        response = await mystic.process_query(query, {"session": f"conversation_{i}"})
        print(f"Q{i+1}: {query}")
        print(f"A{i+1}: {response['response'][:100]}...")
        print()
    
    # Show memory evolution
    final_stats = mystic.get_memory_stats()
    print(f"Final memories: {final_stats['total_memories']}")
    print(f"Final avg salience: {final_stats['avg_salience']:.2f}")
    print(f"Conversation length: {final_stats['conversation_length']}")
    print()


async def test_orchestration():
    """Test multi-persona orchestration"""
    print("=== Test 3: Multi-Persona Orchestration ===")
    
    # Create orchestrator without AI service
    orchestrator = AdaptivePersonaOrchestrator()
    
    # Test complex task requiring multiple perspectives
    task = "Helping someone transition careers from engineering to creative work"
    query = "I've been an engineer for 10 years but want to become an artist. How should I approach this transition?"
    
    result = await orchestrator.orchestrate_task(task, query)
    
    print(f"Task: {task}")
    print(f"Query: {query}")
    print()
    print(f"Active personas: {result['active_personas']}")
    print(f"Task requirements: {result['task_requirements']}")
    print()
    print("=== Individual Perspectives ===")
    for persona_id, perspective in result['individual_perspectives'].items():
        if not perspective.get('error'):
            print(f"\n{persona_id.title()}: {perspective['response']}")
    
    print(f"\n=== Synthesized Response ===")
    print(result['synthesized_response'])
    print()


async def test_persona_library():
    """Test persona library functionality"""
    print("=== Test 4: Persona Library ===")
    
    # Create library and factory
    library = PersonaLibrary()
    factory = AdaptivePersonaFactory()
    
    # Create and save a persona
    artisan = await factory.create_artisan_persona("woodworking specialist")
    persona_id = library.save_persona(
        artisan, 
        "Master Woodworker", 
        "Expert craftsperson specializing in traditional woodworking",
        ["artisan", "woodworking", "craft"]
    )
    
    print(f"Saved persona with ID: {persona_id}")
    
    # List all personas
    all_personas = library.list_personas()
    print(f"Total personas in library: {len(all_personas)}")
    for persona in all_personas:
        print(f"  - {persona['title']} ({persona['archetype']}) - used {persona['use_count']} times")
    
    # Load and test the persona
    loaded_persona = library.load_persona(persona_id)
    if loaded_persona:
        loaded_persona.activate_persona("Woodworking advice")
        response = await loaded_persona.process_query(
            "What's the most important thing for a beginner to learn?",
            {"skill_level": "beginner"}
        )
        print(f"\nLoaded persona response: {response['response']}")
    
    print()


async def test_with_mock_ai():
    """Test with mock AI service"""
    print("=== Test 5: Mock AI Service Integration ===")
    
    # Create factory with mock AI service
    mock_ai = MockAIService()
    factory = AdaptivePersonaFactory(mock_ai)
    
    # Create persona (will still use fallback since mock AI returns None for generate_persona)
    wanderer = await factory.create_wanderer_persona("travel and exploration")
    
    # Activate and test with AI-powered responses
    wanderer.activate_persona("Travel advice and cultural insights")
    
    queries = [
        "How do you connect with locals when traveling?",
        "What's the most meaningful journey you can take?"
    ]
    
    for query in queries:
        response = await wanderer.process_query(query, {"context": "travel"}, mock_ai)
        print(f"Query: {query}")
        print(f"Response: {response['response']}")
        print(f"Generation method: {response.get('generation_method', 'unknown')}")
        print()
    
    print(f"AI service call count: {mock_ai.call_count}")
    print()





async def run_all_tests():
    """Run all test functions"""
    print("🚀 Starting Adaptive Persona System Tests\n")
    
    try:
        await test_basic_persona_functionality()
        await test_memory_system()
        await test_orchestration()
        await test_persona_library()
        await test_with_mock_ai()
        
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_all_tests())
