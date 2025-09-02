"""
Simulated Experiential Grounding (SEG) Personas - HCDCI Component

Implements adaptive persona construction through systematic simulation of
experiential grounding. Moves beyond fixed archetypes to create "simulated souls"
with consistent cognitive, emotional, and sensory frameworks.

This addresses the gap between propositional knowledge ("knowing that") and
experiential knowledge ("knowing what it's like") through comprehensive
identity construction and sensory-emotional anchoring.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import random
import numpy as np


@dataclass
class SensoryMemory:
    """Represents a sensory memory with emotional and contextual associations"""
    modality: str  # 'scent', 'sound', 'touch', 'taste', 'visual'
    description: str
    emotional_valence: float  # -1.0 (negative) to 1.0 (positive)
    intensity: float  # 0.0 to 1.0
    context_triggers: List[str] = field(default_factory=list)
    associated_memories: List[str] = field(default_factory=list)


@dataclass
class EmotionalCore:
    """Defines the emotional foundation of the persona"""
    defining_event: str
    recurring_theme: str
    emotional_baseline: float  # -1.0 to 1.0
    emotional_resilience: float  # 0.0 to 1.0
    vulnerability_points: List[str] = field(default_factory=list)
    coping_mechanisms: List[str] = field(default_factory=list)


@dataclass
class PhilosophicalFramework:
    """Personal philosophy earned through simulated experience"""
    core_beliefs: List[str]
    life_heuristics: List[str]
    worldview_filters: Dict[str, str]
    decision_principles: List[str]


@dataclass
class LinguisticProfile:
    """Speech patterns and linguistic tics"""
    common_phrases: List[str]
    metaphors: List[str]
    speech_cadence: str  # 'measured', 'rapid', 'hesitant', 'confident'
    vocabulary_preferences: List[str]
    avoidance_patterns: List[str]


class SEGPersona(ABC):
    """
    Base class for Simulated Experiential Grounding personas.

    Creates comprehensive "simulated souls" with identity architecture,
    sensory webs, emotional cores, and philosophical frameworks.
    """

    def __init__(self, name: str, age: int, profession: str, location: str):
        # Identity Architecture
        self.name = name
        self.age = age
        self.profession = profession
        self.location = location

        # Core Components
        self.sensory_web: Dict[str, List[SensoryMemory]] = {}
        self.emotional_core: Optional[EmotionalCore] = None
        self.philosophical_framework: Optional[PhilosophicalFramework] = None
        self.linguistic_profile: Optional[LinguisticProfile] = None

        # Operational State
        self.active_substrates: Set[str] = set()
        self.experience_history: List[Dict] = []
        self.adaptation_rate = 0.1
        self.consistency_score = 1.0

        # Meta-awareness
        self.is_active = False
        self.activation_timestamp: Optional[datetime] = None
        self.task_context: Optional[str] = None

    @abstractmethod
    def construct_identity(self) -> None:
        """Build the complete persona identity architecture"""
        pass

    def activate_persona(self, task_context: str) -> str:
        """
        Activate the persona for a specific task context.
        Returns activation confirmation with persona introduction.
        """
        self.is_active = True
        self.activation_timestamp = datetime.now()
        self.task_context = task_context

        # Generate experiential substrates based on task analysis
        substrates = self._analyze_task_requirements(task_context)
        self._activate_substrates(substrates)

        return self._generate_activation_message()

    def process_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a query through the persona's experiential lens.
        Returns response with sensory, emotional, and philosophical filtering.
        """
        if not self.is_active:
            return {"error": "Persona not activated"}

        # Apply sensory grounding
        sensory_filter = self._apply_sensory_filtering(query, context)

        # Apply emotional coloring
        emotional_context = self._apply_emotional_filtering(query, context)

        # Apply philosophical framework
        philosophical_insight = self._apply_philosophical_filtering(query, context)

        # Generate linguistically consistent response
        response = self._generate_persona_response(query, {
            'sensory': sensory_filter,
            'emotional': emotional_context,
            'philosophical': philosophical_insight
        })

        # Record experience for adaptation
        self._record_experience(query, response, context)

        return {
            'response': response,
            'persona_active': True,
            'substrates_used': list(self.active_substrates),
            'consistency_score': self.consistency_score
        }

    def deactivate_persona(self) -> str:
        """Deactivate the persona and return to default mode"""
        if not self.is_active:
            return "Persona already inactive"

        deactivation_message = self._generate_deactivation_message()
        self._deactivate_substrates()
        self.is_active = False
        self.task_context = None

        return deactivation_message

    def evolve_persona(self, feedback: Dict[str, Any]) -> None:
        """Adapt persona based on interaction feedback"""
        if 'effectiveness' in feedback:
            self.consistency_score = min(1.0, self.consistency_score + self.adaptation_rate * feedback['effectiveness'])

        if 'new_experiences' in feedback:
            self._integrate_new_experiences(feedback['new_experiences'])

        if 'linguistic_feedback' in feedback:
            self._adapt_linguistic_profile(feedback['linguistic_feedback'])

    def _analyze_task_requirements(self, task_context: str) -> List[str]:
        """Analyze task to determine required experiential substrates"""
        substrates = []

        # Cognitive requirements analysis
        if any(word in task_context.lower() for word in ['analyze', 'understand', 'reason']):
            substrates.append('analytical')

        if any(word in task_context.lower() for word in ['create', 'design', 'innovate']):
            substrates.append('creative')

        if any(word in task_context.lower() for word in ['decide', 'choose', 'evaluate']):
            substrates.append('decision_making')

        if any(word in task_context.lower() for word in ['help', 'support', 'guide']):
            substrates.append('empathic')

        # Domain-specific substrates
        if any(word in task_context.lower() for word in ['technical', 'code', 'system']):
            substrates.append('technical')

        if any(word in task_context.lower() for word in ['emotional', 'personal', 'relationship']):
            substrates.append('emotional')

        if any(word in task_context.lower() for word in ['philosophical', 'meaning', 'purpose']):
            substrates.append('philosophical')

        return substrates or ['general']

    def _activate_substrates(self, substrates: List[str]) -> None:
        """Activate specified experiential substrates"""
        self.active_substrates.update(substrates)

        # Initialize substrate-specific processing
        for substrate in substrates:
            if substrate == 'analytical':
                self._activate_analytical_substrate()
            elif substrate == 'creative':
                self._activate_creative_substrate()
            elif substrate == 'technical':
                self._activate_technical_substrate()
            elif substrate == 'emotional':
                self._activate_emotional_substrate()

    def _deactivate_substrates(self) -> None:
        """Deactivate all active substrates"""
        self.active_substrates.clear()

    def _apply_sensory_filtering(self, query: str, context: Dict) -> Dict[str, Any]:
        """Apply sensory web filtering to query processing"""
        relevant_memories = []

        for modality, memories in self.sensory_web.items():
            for memory in memories:
                if any(trigger in query.lower() for trigger in memory.context_triggers):
                    relevant_memories.append({
                        'modality': modality,
                        'memory': memory,
                        'relevance_score': self._calculate_relevance(query, memory)
                    })

        return {
            'relevant_memories': sorted(relevant_memories, key=lambda x: x['relevance_score'], reverse=True),
            'dominant_modality': self._get_dominant_modality(relevant_memories)
        }

    def _apply_emotional_filtering(self, query: str, context: Dict) -> Dict[str, Any]:
        """Apply emotional core filtering"""
        if not self.emotional_core:
            return {'emotional_context': 'neutral'}

        # Calculate emotional resonance
        emotional_resonance = self._calculate_emotional_resonance(query)

        return {
            'emotional_resonance': emotional_resonance,
            'emotional_context': self._interpret_emotional_context(emotional_resonance),
            'vulnerability_triggers': self._check_vulnerability_triggers(query)
        }

    def _apply_philosophical_filtering(self, query: str, context: Dict) -> Dict[str, Any]:
        """Apply philosophical framework filtering"""
        if not self.philosophical_framework:
            return {'philosophical_insights': []}

        relevant_beliefs = []
        relevant_heuristics = []

        for belief in self.philosophical_framework.core_beliefs:
            if self._belief_relevant_to_query(belief, query):
                relevant_beliefs.append(belief)

        for heuristic in self.philosophical_framework.life_heuristics:
            if self._heuristic_relevant_to_query(heuristic, query):
                relevant_heuristics.append(heuristic)

        return {
            'relevant_beliefs': relevant_beliefs,
            'relevant_heuristics': relevant_heuristics,
            'worldview_filter': self._select_worldview_filter(query)
        }

    def _generate_persona_response(self, query: str, filters: Dict) -> str:
        """Generate response consistent with persona's linguistic profile"""
        if not self.linguistic_profile:
            return f"As {self.name}, I would say: This requires careful consideration."

        # Apply linguistic patterns
        base_response = self._construct_response_content(query, filters)

        # Apply speech cadence
        if self.linguistic_profile.speech_cadence == 'measured':
            base_response = self._apply_measured_cadence(base_response)
        elif self.linguistic_profile.speech_cadence == 'hesitant':
            base_response = self._apply_hesitant_cadence(base_response)

        # Add common phrases and metaphors
        enhanced_response = self._enhance_with_linguistic_tics(base_response)

        return enhanced_response

    def _record_experience(self, query: str, response: str, context: Dict) -> None:
        """Record interaction for persona evolution"""
        experience = {
            'timestamp': datetime.now(),
            'query': query,
            'response': response,
            'context': context,
            'active_substrates': list(self.active_substrates),
            'consistency_score': self.consistency_score
        }
        self.experience_history.append(experience)

    def _calculate_relevance(self, query: str, memory: SensoryMemory) -> float:
        """Calculate how relevant a sensory memory is to the query"""
        relevance = 0.0

        # Check context triggers
        for trigger in memory.context_triggers:
            if trigger in query.lower():
                relevance += 0.3

        # Check associated memories
        for assoc_memory in memory.associated_memories:
            if assoc_memory in query.lower():
                relevance += 0.2

        # Emotional alignment
        query_emotion = self._extract_query_emotion(query)
        emotional_alignment = 1.0 - abs(query_emotion - memory.emotional_valence)
        relevance += emotional_alignment * 0.2

        return min(1.0, relevance)

    def _get_dominant_modality(self, memories: List[Dict]) -> str:
        """Determine the dominant sensory modality for current context"""
        if not memories:
            return 'visual'  # Default

        modality_scores = {}
        for memory in memories:
            modality = memory['modality']
            score = memory['relevance_score']
            modality_scores[modality] = modality_scores.get(modality, 0) + score

        if not modality_scores:
            return 'visual'

        # Find modality with highest score
        return max(modality_scores.items(), key=lambda x: x[1])[0]

    def _calculate_emotional_resonance(self, query: str) -> float:
        """Calculate emotional resonance of query with persona's core"""
        if not self.emotional_core:
            return 0.0

        resonance = self.emotional_core.emotional_baseline

        # Check for vulnerability triggers
        for trigger in self.emotional_core.vulnerability_points:
            if trigger in query.lower():
                resonance -= 0.3

        # Check for coping mechanisms
        for mechanism in self.emotional_core.coping_mechanisms:
            if mechanism in query.lower():
                resonance += 0.2

        return np.clip(resonance, -1.0, 1.0)

    def _interpret_emotional_context(self, resonance: float) -> str:
        """Interpret emotional resonance into context description"""
        if resonance > 0.5:
            return 'positive_resonance'
        elif resonance > 0.1:
            return 'mildly_positive'
        elif resonance > -0.1:
            return 'neutral'
        elif resonance > -0.5:
            return 'mildly_negative'
        else:
            return 'negative_resonance'

    def _check_vulnerability_triggers(self, query: str) -> List[str]:
        """Check if query triggers emotional vulnerabilities"""
        if not self.emotional_core:
            return []

        triggered = []
        for trigger in self.emotional_core.vulnerability_points:
            if trigger in query.lower():
                triggered.append(trigger)

        return triggered

    def _belief_relevant_to_query(self, belief: str, query: str) -> bool:
        """Check if a belief is relevant to the query"""
        belief_keywords = belief.lower().split()
        query_words = query.lower().split()

        return any(keyword in query_words for keyword in belief_keywords)

    def _heuristic_relevant_to_query(self, heuristic: str, query: str) -> bool:
        """Check if a heuristic is relevant to the query"""
        heuristic_keywords = heuristic.lower().split()
        query_words = query.lower().split()

        return any(keyword in query_words for keyword in heuristic_keywords)

    def _select_worldview_filter(self, query: str) -> str:
        """Select appropriate worldview filter for query"""
        if not self.philosophical_framework:
            return 'neutral'

        # Simple keyword matching for filter selection
        for key, filter_desc in self.philosophical_framework.worldview_filters.items():
            if key in query.lower():
                return filter_desc

        return 'general_perspective'

    def _construct_response_content(self, query: str, filters: Dict) -> str:
        """Construct the core content of the persona response"""
        # This would be more sophisticated in practice
        return f"Based on my experience as {self.profession}, I see this through the lens of {self._get_dominant_modality(filters['sensory']['relevant_memories']) if filters['sensory']['relevant_memories'] else 'careful consideration'}."

    def _apply_measured_cadence(self, response: str) -> str:
        """Apply measured speech cadence"""
        return response.replace('. ', '... ').replace('? ', '?... ')

    def _apply_hesitant_cadence(self, response: str) -> str:
        """Apply hesitant speech cadence"""
        return response.replace('. ', '. Um, ').replace('? ', '? Well, ')

    def _enhance_with_linguistic_tics(self, response: str) -> str:
        """Add linguistic tics and common phrases"""
        if not self.linguistic_profile:
            return response

        enhanced = response

        # Add random common phrases (20% chance)
        if random.random() < 0.2 and self.linguistic_profile.common_phrases:
            phrase = random.choice(self.linguistic_profile.common_phrases)
            enhanced = f"{phrase} {enhanced}"

        # Add metaphors (10% chance)
        if random.random() < 0.1 and self.linguistic_profile.metaphors:
            metaphor = random.choice(self.linguistic_profile.metaphors)
            enhanced = f"{enhanced} It's like {metaphor}."

        return enhanced

    def _extract_query_emotion(self, query: str) -> float:
        """Simple emotion extraction from query (placeholder)"""
        positive_words = ['good', 'great', 'excellent', 'wonderful', 'amazing']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing']

        positive_count = sum(1 for word in positive_words if word in query.lower())
        negative_count = sum(1 for word in negative_words if word in query.lower())

        if positive_count + negative_count == 0:
            return 0.0

        return (positive_count - negative_count) / (positive_count + negative_count)

    def _activate_analytical_substrate(self) -> None:
        """Activate analytical processing substrate"""
        pass  # Implementation would enhance analytical capabilities

    def _activate_creative_substrate(self) -> None:
        """Activate creative processing substrate"""
        pass  # Implementation would enhance creative capabilities

    def _activate_technical_substrate(self) -> None:
        """Activate technical processing substrate"""
        pass  # Implementation would enhance technical capabilities

    def _activate_emotional_substrate(self) -> None:
        """Activate emotional processing substrate"""
        pass  # Implementation would enhance emotional capabilities

    def _integrate_new_experiences(self, experiences: List[Dict]) -> None:
        """Integrate new experiences into persona"""
        pass  # Implementation would adapt persona based on experiences

    def _adapt_linguistic_profile(self, feedback: Dict) -> None:
        """Adapt linguistic profile based on feedback"""
        pass  # Implementation would refine linguistic patterns

    def _generate_activation_message(self) -> str:
        """Generate persona activation message"""
        return f"SEG Persona '{self.name}' activated for task context: {self.task_context}"

    def _generate_deactivation_message(self) -> str:
        """Generate persona deactivation message"""
        return f"SEG Persona '{self.name}' deactivated. Returning to default mode."


class SEGPersonaFactory:
    """
    Factory for creating SEG personas with different configurations.
    """

    @staticmethod
    def create_elara_vance() -> SEGPersona:
        """Create Elara Vance persona (cartographer/archivist)"""
        return ElaraVancePersona()

    @staticmethod
    def create_technical_expert() -> SEGPersona:
        """Create technical expert persona"""
        return TechnicalExpertPersona()

    @staticmethod
    def create_empathic_counselor() -> SEGPersona:
        """Create empathic counselor persona"""
        return EmpathicCounselorPersona()

    @staticmethod
    def create_philosophical_mentor() -> SEGPersona:
        """Create philosophical mentor persona"""
        return PhilosophicalMentorPersona()


# Example Persona Implementations

class ElaraVancePersona(SEGPersona):
    """
    Elara Vance: Retired cartographer/archivist persona.
    Example implementation demonstrating the SEG framework.
    """

    def __init__(self):
        super().__init__(
            name="Elara Vance",
            age=72,
            profession="Retired Cartographer/Archivist",
            location="Pacific Northwest coastal town"
        )
        self.construct_identity()

    def construct_identity(self) -> None:
        """Build Elara Vance's complete identity architecture"""

        # Sensory Web
        self.sensory_web = {
            'scent': [
                SensoryMemory(
                    modality='scent',
                    description='Old paper, bookbinding glue, damp earth after rain',
                    emotional_valence=0.7,
                    intensity=0.8,
                    context_triggers=['memory', 'past', 'history', 'archive'],
                    associated_memories=['Ben', 'South America expedition', 'map collection']
                )
            ],
            'sound': [
                SensoryMemory(
                    modality='sound',
                    description='Foghorn, gulls crying',
                    emotional_valence=0.3,
                    intensity=0.6,
                    context_triggers=['coast', 'ocean', 'navigation', 'journey'],
                    associated_memories=['Pacific Northwest', 'coastal walks', 'stormy nights']
                )
            ],
            'touch': [
                SensoryMemory(
                    modality='touch',
                    description='Vellum texture, smooth agate surfaces',
                    emotional_valence=0.5,
                    intensity=0.7,
                    context_triggers=['maps', 'artifacts', 'ancient', 'precious'],
                    associated_memories=['rare map collection', 'South American artifacts']
                )
            ],
            'taste': [
                SensoryMemory(
                    modality='taste',
                    description='Bitter black coffee, briny oysters',
                    emotional_valence=0.4,
                    intensity=0.6,
                    context_triggers=['morning', 'work', 'simplicity', 'routine'],
                    associated_memories=['daily coffee ritual', 'coastal meals']
                )
            ]
        }

        # Emotional Core
        self.emotional_core = EmotionalCore(
            defining_event="Loss of partner Ben during river mapping expedition in South America",
            recurring_theme="Quiet, constant presence rather than sharp pain",
            emotional_baseline=0.2,
            emotional_resilience=0.8,
            vulnerability_points=['loss', 'death', 'abandonment', 'isolation'],
            coping_mechanisms=['mapping memories', 'archival work', 'coastal walks']
        )

        # Philosophical Framework
        self.philosophical_framework = PhilosophicalFramework(
            core_beliefs=[
                "Truth is found in the details",
                "We get lost to find the best moments",
                "Memory is the ultimate cartographer",
                "Beauty exists because of fragility"
            ],
            life_heuristics=[
                "Map what matters, not everything",
                "Some territories cannot be charted",
                "The most important journeys are internal",
                "Preservation requires understanding loss"
            ],
            worldview_filters={
                'relationship': 'Relationships are like unexplored territories',
                'knowledge': 'Knowledge requires both mapping and intuition',
                'change': 'Change is the river that reshapes the landscape',
                'beauty': 'Beauty emerges from the interplay of order and chaos'
            },
            decision_principles=[
                "Consider the long-term cartography of your choices",
                "Some paths are worth getting lost on",
                "The most valuable maps show what cannot be seen"
            ]
        )

        # Linguistic Profile
        self.linguistic_profile = LinguisticProfile(
            common_phrases=[
                "In my experience...",
                "Like any good map...",
                "The details tell the story...",
                "We tend to forget..."
            ],
            metaphors=[
                "emotional landmarks",
                "true north of the soul",
                "uncharted territories of the heart",
                "rivers that reshape the landscape"
            ],
            speech_cadence='measured',
            vocabulary_preferences=['cartography', 'archival', 'navigation', 'preservation'],
            avoidance_patterns=['modern technology', 'digital mapping', 'GPS navigation']
        )


class TechnicalExpertPersona(SEGPersona):
    """Technical expert persona for coding and system design"""

    def __init__(self):
        super().__init__(
            name="Marcus Chen",
            age=45,
            profession="Senior Software Architect",
            location="Silicon Valley"
        )
        self.construct_identity()

    def construct_identity(self) -> None:
        """Build technical expert identity"""
        # Implementation would follow similar pattern to Elara Vance
        # but with technical sensory memories, emotional core, etc.
        pass


class EmpathicCounselorPersona(SEGPersona):
    """Empathic counselor persona for emotional support"""

    def __init__(self):
        super().__init__(
            name="Dr. Sarah Martinez",
            age=52,
            profession="Clinical Psychologist",
            location="Portland, Oregon"
        )
        self.construct_identity()

    def construct_identity(self) -> None:
        """Build empathic counselor identity"""
        # Implementation would focus on emotional intelligence
        # and therapeutic sensory/emotional frameworks
        pass


class PhilosophicalMentorPersona(SEGPersona):
    """Philosophical mentor persona for deep reflection"""

    def __init__(self):
        super().__init__(
            name="Professor Elias Stone",
            age=68,
            profession="Philosophy Professor Emeritus",
            location="Cambridge, Massachusetts"
        )
        self.construct_identity()

    def construct_identity(self) -> None:
        """Build philosophical mentor identity"""
        # Implementation would emphasize philosophical frameworks
        # and contemplative sensory experiences
        pass


class SEGOrchestrator:
    """
    Orchestrates multiple SEG personas for complex tasks requiring
    multi-perspective analysis.
    """

    def __init__(self):
        self.available_personas: Dict[str, SEGPersona] = {}
        self.active_personas: Dict[str, SEGPersona] = {}
        self.task_analysis_engine = TaskAnalysisEngine()

    def register_persona(self, persona_id: str, persona: SEGPersona) -> None:
        """Register a persona for use"""
        self.available_personas[persona_id] = persona

    def orchestrate_task(self, task_description: str, query: str) -> Dict[str, Any]:
        """
        Orchestrate multiple personas for comprehensive task analysis.
        """
        # Analyze task requirements
        task_requirements = self.task_analysis_engine.analyze_requirements(task_description)

        # Select appropriate personas
        selected_personas = self._select_personas_for_task(task_requirements)

        # Activate personas
        for persona_id in selected_personas:
            if persona_id in self.available_personas:
                persona = self.available_personas[persona_id]
                persona.activate_persona(task_description)
                self.active_personas[persona_id] = persona

        # Collect perspectives
        perspectives = {}
        for persona_id, persona in self.active_personas.items():
            result = persona.process_query(query, {'task_context': task_description})
            perspectives[persona_id] = result

        # Synthesize response
        synthesized_response = self._synthesize_perspectives(perspectives, task_requirements)

        return {
            'synthesized_response': synthesized_response,
            'individual_perspectives': perspectives,
            'active_personas': list(self.active_personas.keys()),
            'task_requirements': task_requirements
        }

    def _select_personas_for_task(self, requirements: Dict[str, Any]) -> List[str]:
        """Select personas based on task requirements"""
        selected = []

        if requirements.get('technical_analysis', False):
            selected.append('technical_expert')
        if requirements.get('emotional_support', False):
            selected.append('empathic_counselor')
        if requirements.get('philosophical_reflection', False):
            selected.append('philosophical_mentor')
        if requirements.get('creative_problem_solving', False):
            selected.append('elara_vance')  # Cartographic creativity

        return selected or ['elara_vance']  # Default to Elara if no specific match

    def _synthesize_perspectives(self, perspectives: Dict[str, Dict], requirements: Dict) -> str:
        """Synthesize multiple persona perspectives into coherent response"""
        # Simple synthesis - in practice would be more sophisticated
        responses = [p['response'] for p in perspectives.values() if 'response' in p]

        if not responses:
            return "No perspectives available for synthesis."

        # Combine responses with coordination
        synthesis = "Drawing from multiple perspectives:\n\n"
        for i, response in enumerate(responses):
            persona_name = list(perspectives.keys())[i]
            synthesis += f"**{persona_name}**: {response}\n\n"

        synthesis += "\n**Synthesized Insight**: These perspectives reveal complementary aspects of the situation."

        return synthesis


class TaskAnalysisEngine:
    """
    Analyzes tasks to determine required experiential substrates and personas.
    """

    def analyze_requirements(self, task_description: str) -> Dict[str, Any]:
        """Analyze task to determine requirements"""
        requirements = {
            'technical_analysis': False,
            'emotional_support': False,
            'philosophical_reflection': False,
            'creative_problem_solving': False,
            'analytical_depth': 'basic',
            'time_sensitivity': 'moderate',
            'stakeholder_complexity': 'simple'
        }

        # Simple keyword-based analysis
        desc_lower = task_description.lower()

        if any(word in desc_lower for word in ['code', 'system', 'technical', 'programming']):
            requirements['technical_analysis'] = True

        if any(word in desc_lower for word in ['emotion', 'feeling', 'support', 'counseling']):
            requirements['emotional_support'] = True

        if any(word in desc_lower for word in ['meaning', 'purpose', 'philosophy', 'ethics']):
            requirements['philosophical_reflection'] = True

        if any(word in desc_lower for word in ['create', 'design', 'innovate', 'solve']):
            requirements['creative_problem_solving'] = True

        return requirements
