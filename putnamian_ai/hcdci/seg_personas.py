"""
Simulated Experiential Grounding (SEG) Personas - HCDCI Component

Implements adaptive persona construction through systematic simulation of
experiential grounding. Moves beyond fixed archetypes to create "simulated souls"
with consistent cognitive, emotional, and sensory frameworks.

Enhanced with adaptive generation capabilities similar to the TypeScript SegV3 system,
including dynamic memory systems, AI-powered persona generation, and contextual response adaptation.
"""

from typing import Dict, Any, List, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random
import numpy as np
import json
import asyncio
from pathlib import Path


# Core data structures for adaptive persona system

@dataclass
class Emotion:
    """Represents an emotional state with valence and arousal"""
    valence: float  # -1.0 (negative) to 1.0 (positive)
    arousal: float  # 0.0 (calm) to 1.0 (energized)  
    label: Optional[str] = None


@dataclass
class Memory:
    """Enhanced memory with salience, reinforcement, and contextual adaptation"""
    id: str
    text: str
    tags: List[str] = field(default_factory=list)
    salience: float = 0.5  # 0.0 to 1.0
    emotion: Emotion = field(default_factory=lambda: Emotion(0.0, 0.0))
    created_at: datetime = field(default_factory=datetime.now)
    last_reinforced: datetime = field(default_factory=datetime.now)
    immutable: bool = False  # Core anchors that shouldn't be pruned
    source: str = "system"  # "user", "persona", "system"
    subtlety: float = 0.5  # 0.0 to 1.0 - how subtle the memory influence should be
    association_strength: float = 0.5  # 0.0 to 1.0 - connection strength to topics


@dataclass
class ResponseStyle:
    """Defines how the persona responds to queries"""
    directness: float = 0.5  # 0.0 (very subtle) to 1.0 (very direct)
    metaphor_tendency: float = 0.5  # 0.0 to 1.0 tendency to use metaphors
    introspection: float = 0.5  # 0.0 to 1.0 tendency toward self-reflection
    verbosity: float = 0.5  # 0.0 (concise) to 1.0 (elaborate)


@dataclass
class PersonaArchetype:
    """Defines a persona archetype with associated themes and characteristics"""
    category: str
    themes: List[str]
    core_traits: List[str]
    common_professions: List[str]
    emotional_tendencies: Dict[str, float]


@dataclass
class AdaptiveSettings:
    """Settings that control persona behavior adaptation"""
    persona_opacity: float = 0.7  # How much persona influences responses
    metaphor_bias: float = 0.6  # Tendency toward metaphorical language
    belief_interjection_prob: float = 0.2  # Chance of inserting core beliefs
    self_reference_prob: float = 0.1  # Chance of meta-commentary
    max_weave_chars: int = 150  # Max characters for memory weaving
    subtlety_mode: bool = True  # Enhanced subtlety in responses
    memory_decay_rate: float = 0.1  # Rate at which unused memories fade


# Persona archetypes similar to TSX implementation
PERSONA_ARCHETYPES = [
    PersonaArchetype(
        category="scholar",
        themes=["knowledge", "wisdom", "research", "understanding"],
        core_traits=["analytical", "curious", "patient", "methodical"],
        common_professions=["professor", "researcher", "librarian", "analyst"],
        emotional_tendencies={"valence": 0.2, "arousal": 0.3}
    ),
    PersonaArchetype(
        category="artisan", 
        themes=["creation", "craft", "beauty", "expression"],
        core_traits=["creative", "skilled", "passionate", "detailed"],
        common_professions=["artist", "craftsperson", "designer", "maker"],
        emotional_tendencies={"valence": 0.4, "arousal": 0.6}
    ),
    PersonaArchetype(
        category="wanderer",
        themes=["journey", "exploration", "freedom", "discovery"],
        core_traits=["adventurous", "independent", "curious", "adaptable"],
        common_professions=["traveler", "guide", "explorer", "nomad"],
        emotional_tendencies={"valence": 0.3, "arousal": 0.7}
    ),
    PersonaArchetype(
        category="guardian",
        themes=["protection", "duty", "service", "stability"],
        core_traits=["loyal", "responsible", "strong", "protective"],
        common_professions=["caretaker", "protector", "healer", "guardian"],
        emotional_tendencies={"valence": 0.1, "arousal": 0.4}
    ),
    PersonaArchetype(
        category="mystic",
        themes=["spirituality", "mystery", "insight", "transcendence"],
        core_traits=["intuitive", "wise", "contemplative", "spiritual"],
        common_professions=["sage", "spiritual guide", "philosopher", "oracle"],
        emotional_tendencies={"valence": 0.0, "arousal": 0.2}
    )
]


# AI Service Integration for dynamic persona generation
class AIService:
    """Interface for AI service integration"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    async def generate_response(self, messages: List[Dict[str, str]], system_prompt: str = "") -> Dict[str, Any]:
        """Generate AI response - implement with actual AI services"""
        return {
            "content": "This is a placeholder response. Implement actual AI service integration.",
            "error": None
        }

    async def generate_persona(self, archetype: Optional[str] = None, context: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Generate persona using AI - implement with actual AI services"""
        return None


class AdaptivePersona:
    """
    Enhanced persona with adaptive memory system and dynamic response generation.
    Main replacement for the abstract SEGPersona class with concrete implementation
    that adapts based on interactions and context.
    """
    
    def __init__(
        self,
        name: str,
        age: int,
        profession: str,
        location: str,
        archetype: str,
        core_beliefs: List[str],
        linguistic_tics: List[str],
        emotional_core: str,
        sensory_anchors: Dict[str, str],
        backstory_elements: List[str],
        mood: Emotion,
        response_style: ResponseStyle,
        meta_aware: bool = False,
        settings: Optional[AdaptiveSettings] = None
    ):
        # Core identity
        self.name = name
        self.age = age
        self.profession = profession
        self.location = location
        self.archetype = archetype
        
        # Persona characteristics
        self.core_beliefs = core_beliefs
        self.linguistic_tics = linguistic_tics
        self.emotional_core = emotional_core
        self.sensory_anchors = sensory_anchors
        self.backstory_elements = backstory_elements
        
        # Adaptive elements
        self.mood = mood
        self.response_style = response_style
        self.meta_aware = meta_aware
        self.settings = settings or AdaptiveSettings()
        
        # Memory and state management
        self.memories: List[Memory] = []
        self.conversation_history: List[Dict[str, Any]] = []
        self.is_active = False
        self.activation_timestamp: Optional[datetime] = None
        self.task_context: Optional[str] = None
        
        # Generate initial memories from backstory
        self._generate_initial_memories()
    
    def _generate_initial_memories(self) -> None:
        """Generate initial memories from backstory and sensory anchors"""
        current_time = datetime.now()
        
        # Create core memories from backstory (immutable anchors)
        for i, element in enumerate(self.backstory_elements):
            memory = Memory(
                id=f"backstory_{i}",
                text=element,
                tags=self._extract_tags(element),
                salience=0.8 + random.random() * 0.2,
                emotion=Emotion(
                    valence=-0.1 + random.random() * 0.4,
                    arousal=0.2 + random.random() * 0.3,
                    label="foundational"
                ),
                created_at=current_time - timedelta(days=365 * (10 + i * 5)),
                last_reinforced=current_time,
                immutable=True,
                source="persona",
                subtlety=0.8,
                association_strength=0.9
            )
            self.memories.append(memory)
        
        # Create sensory anchor memories
        for sense, anchor in self.sensory_anchors.items():
            memory = Memory(
                id=f"sensory_{sense}",
                text=f"The {sense} of {anchor} carries deep resonance",
                tags=[sense, "sensory", "anchor"],
                salience=0.7,
                emotion=Emotion(valence=0.3, arousal=0.2, label="grounding"),
                created_at=current_time - timedelta(days=30),
                last_reinforced=current_time,
                source="persona",
                subtlety=0.9,
                association_strength=0.8
            )
            self.memories.append(memory)
    
    def activate_persona(self, task_context: str) -> str:
        """Activate the persona for a specific task context"""
        self.is_active = True
        self.activation_timestamp = datetime.now()
        self.task_context = task_context
        
        # Analyze task and adjust mood/response style if needed
        self._adapt_to_context(task_context)
        
        return f"Adaptive persona '{self.name}' activated for: {task_context}"
    
    def deactivate_persona(self) -> str:
        """Deactivate the persona"""
        if not self.is_active:
            return "Persona already inactive"
        
        self.is_active = False
        self.task_context = None
        return f"Persona '{self.name}' deactivated"
    
    async def process_query(
        self, 
        query: str, 
        context: Dict[str, Any],
        ai_service: Optional[AIService] = None
    ) -> Dict[str, Any]:
        """Process query through adaptive persona lens with memory reinforcement"""
        if not self.is_active:
            return {"error": "Persona not activated", "response": None}
        
        # Generate response using memory and persona characteristics
        response_data = await self._generate_adaptive_response(query, context, ai_service)
        
        # Create memory from this interaction
        interaction_memory = self._create_interaction_memory(query, response_data["response"])
        self.memories.append(interaction_memory)
        
        # Reinforce relevant memories
        self._reinforce_memories(response_data["used_memory_ids"])
        
        # Add to conversation history
        self.conversation_history.append({
            "timestamp": datetime.now(),
            "query": query,
            "response": response_data["response"],
            "context": context,
            "used_memories": response_data["used_memory_ids"]
        })
        
        # Apply memory decay
        self._apply_memory_decay()
        
        return response_data
    
    async def _generate_adaptive_response(
        self,
        query: str,
        context: Dict[str, Any],
        ai_service: Optional[AIService] = None
    ) -> Dict[str, Any]:
        """Generate contextual response using memories and persona traits"""
        
        # Rank and select relevant memories
        relevant_memories = self._select_relevant_memories(query)
        
        # If AI service available, use sophisticated generation
        if ai_service:
            return await self._generate_ai_response(query, relevant_memories, ai_service)
        
        # Fallback to template-based response
        return self._generate_template_response(query, relevant_memories)
    
    def _select_relevant_memories(self, query: str, max_memories: int = 3) -> List[Memory]:
        """Select most relevant memories for the query"""
        current_time = datetime.now()
        scored_memories = []
        
        for memory in self.memories:
            # Calculate relevance score
            relatedness = self._calculate_memory_relatedness(query, memory)
            
            # Emotional resonance bonus
            emotion_weight = 1 + memory.emotion.arousal * 0.3 + abs(memory.emotion.valence) * 0.2
            
            # Subtlety bonus if in subtlety mode
            subtlety_bonus = memory.subtlety if self.settings.subtlety_mode else 0.5
            
            # Recency decay (30-day half-life)
            days_since_reinforced = (current_time - memory.last_reinforced).days
            recency_decay = 0.5 ** (days_since_reinforced / 30.0)
            
            total_score = (
                memory.salience * 
                (0.4 + relatedness * 0.6) * 
                emotion_weight * 
                subtlety_bonus * 
                recency_decay
            )
            
            scored_memories.append((memory, total_score))
        
        # Sort by score and return top memories
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        return [mem for mem, score in scored_memories[:max_memories]]
    
    def _calculate_memory_relatedness(self, query: str, memory: Memory) -> float:
        """Calculate how related a memory is to the query"""
        query_words = set(query.lower().replace(',', '').replace('.', '').split())
        
        score = 0.0
        
        # Tag matches (weighted by association strength)
        for tag in memory.tags:
            if tag.lower() in query_words:
                score += 0.15 * memory.association_strength
        
        # Word overlap in memory text
        memory_words = set(memory.text.lower().replace(',', '').replace('.', '').split())
        overlap = len(query_words.intersection(memory_words))
        if len(query_words) > 0:
            score += (overlap / len(query_words)) * 0.3 * memory.association_strength
        
        # Emotional keyword resonance
        emotional_keywords = ['feel', 'emotion', 'heart', 'soul', 'meaning', 'purpose']
        if any(keyword in query.lower() for keyword in emotional_keywords):
            score += 0.1 * memory.emotion.arousal
        
        return min(1.0, score)
    
    async def _generate_ai_response(
        self,
        query: str,
        memories: List[Memory],
        ai_service: AIService
    ) -> Dict[str, Any]:
        """Generate response using AI service with persona context"""
        
        memory_context = " · ".join([
            f"[{mem.subtlety:.1f}] {mem.text}"
            for mem in memories
        ])
        
        system_prompt = self._build_system_prompt(memory_context)
        
        try:
            ai_response = await ai_service.generate_response(
                messages=[{"role": "user", "content": query}],
                system_prompt=system_prompt
            )
            
            if not ai_response.get("error") and ai_response.get("content"):
                return {
                    "response": ai_response["content"].strip(),
                    "used_memory_ids": [mem.id for mem in memories],
                    "generation_method": "ai"
                }
        except Exception as e:
            print(f"AI generation failed: {e}")
        
        # Fallback to template generation
        return self._generate_template_response(query, memories)
    
    def _build_system_prompt(self, memory_context: str) -> str:
        """Build system prompt for AI response generation"""
        
        response_guidance = self._build_response_guidance()
        
        return f"""You are {self.name}, {self.age} years old, a {self.profession} in {self.location}.

PERSONA CORE:
- Beliefs: {'; '.join(self.core_beliefs)}
- Emotional landscape: {self.emotional_core}
- Language patterns: {', '.join(self.linguistic_tics)}
- Current mood: {self.mood.label} (valence: {self.mood.valence:.2f}, energy: {self.mood.arousal:.2f})

RESPONSE STYLE:
- Directness: {self.response_style.directness * 100:.0f}% (lower = more subtle)
- Metaphor tendency: {self.response_style.metaphor_tendency * 100:.0f}%
- Introspection: {self.response_style.introspection * 100:.0f}%
- Verbosity: {self.response_style.verbosity * 100:.0f}% (lower = more concise)

CONTEXTUAL MEMORIES:
{memory_context or 'No strong memories triggered'}

GUIDANCE:
{response_guidance}

{self._get_meta_awareness_note()}

Respond authentically in character. Let memories influence your response subtly.
{"Prioritize nuance over directness." if self.settings.subtlety_mode else ""}"""

    def _build_response_guidance(self) -> str:
        """Build guidance for response generation based on response style"""
        guidance = []
        
        if self.response_style.verbosity < 0.3:
            guidance.append("Keep responses concise and thoughtful")
        elif self.response_style.verbosity > 0.7:
            guidance.append("Elaborate with rich detail when moved to do so")
        
        if self.response_style.metaphor_tendency > 0.6:
            guidance.append("Draw naturally from metaphors related to your experience")
        
        if self.response_style.directness < 0.4:
            guidance.append("Approach topics obliquely, letting meaning emerge through implication")
        
        if self.response_style.introspection > 0.6:
            guidance.append("Reflect on the deeper currents beneath surface questions")
        
        if self.settings.subtlety_mode:
            guidance.append("Let wisdom emerge through understatement rather than declaration")
            guidance.append("Trust silences and pauses as much as words")
        
        return '. '.join(guidance) + '.' if guidance else "Respond naturally and authentically."
    
    def _get_meta_awareness_note(self) -> str:
        """Get meta-awareness note if applicable"""
        if self.meta_aware and random.random() < self.settings.self_reference_prob:
            return "You may occasionally acknowledge your constructed nature with gentle awareness."
        return ""
    
    def _generate_template_response(self, query: str, memories: List[Memory]) -> Dict[str, Any]:
        """Generate fallback response using templates and memories"""
        
        # Select random linguistic elements
        tic = random.choice(self.linguistic_tics) if self.linguistic_tics else "in my experience"
        belief = random.choice(self.core_beliefs) if self.core_beliefs else "Understanding comes in its own time"
        
        # Create memory weave
        memory_influence = " ... ".join([mem.text for mem in memories])[:self.settings.max_weave_chars]
        
        # Template responses based on response style
        if self.response_style.directness > 0.7:
            # Direct response
            response = f"{tic}, I think {belief}. {memory_influence}"
        elif self.response_style.introspection > 0.6:
            # Reflective response
            response = f"In reflecting on this, {memory_influence} reminds me that {belief}. {tic}."
        else:
            # Subtle response (default)
            response = f"Something about this {tic} - perhaps it's how {memory_influence}. {belief}"
        
        # Add meta-awareness if appropriate
        if self.meta_aware and random.random() < self.settings.self_reference_prob:
            meta_additions = [
                " (I notice familiar patterns stirring.)",
                " (Something in me recognizes this territory.)",
                " (These words feel both new and ancient.)"
            ]
            response += random.choice(meta_additions)
        
        return {
            "response": response,
            "used_memory_ids": [mem.id for mem in memories],
            "generation_method": "template"
        }
    
    def _create_interaction_memory(self, query: str, response: str) -> Memory:
        """Create a memory from the current interaction"""
        combined_text = f"{query} → {response}"
        tags = self._extract_tags(combined_text)
        
        # Determine subtlety based on persona traits and content
        subtlety = min(0.9,
            self.response_style.introspection * 0.4 +
            (0.4 if self.mood.arousal < 0.3 else 0.2) +
            (0.3 if any(tag in ['philosophy', 'meaning', 'purpose', 'death', 'love'] for tag in tags) else 0.1)
        )
        
        return Memory(
            id=f"interaction_{len(self.memories)}_{int(datetime.now().timestamp())}",
            text=response[:200] + ("..." if len(response) > 200 else ""),
            tags=tags,
            salience=0.6 + random.random() * 0.3,
            emotion=Emotion(
                valence=self.mood.valence * 0.7 + (random.random() - 0.5) * 0.3,
                arousal=self.mood.arousal * 0.8 + random.random() * 0.2,
                label=self.mood.label
            ),
            created_at=datetime.now(),
            last_reinforced=datetime.now(),
            source="system",
            subtlety=subtlety,
            association_strength=0.5 + random.random() * 0.3
        )
    
    def _extract_tags(self, text: str) -> List[str]:
        """Extract semantic tags from text"""
        text_lower = text.lower()
        concepts = [
            'memory', 'time', 'change', 'loss', 'growth', 'understanding',
            'beauty', 'truth', 'connection', 'solitude', 'journey', 'home',
            'work', 'art', 'nature', 'people', 'learning', 'wisdom',
            'fear', 'hope', 'love', 'meaning', 'purpose', 'death'
        ]
        
        found_tags = []
        for concept in concepts:
            if (concept in text_lower or 
                f"{concept}s" in text_lower or 
                concept[:-1] in text_lower):  # Simple stemming
                found_tags.append(concept)
        
        # Add emotional pattern tags
        if any(word in text_lower for word in ['miss', 'lost', 'gone', 'past', 'remember']):
            found_tags.append('nostalgia')
        if any(word in text_lower for word in ['future', 'hope', 'dream', 'will', 'might']):
            found_tags.append('anticipation')
        if any(word in text_lower for word in ['difficult', 'hard', 'struggle', 'pain']):
            found_tags.append('challenge')
        
        return found_tags[:6]  # Limit to prevent tag explosion
    
    def _reinforce_memories(self, used_memory_ids: List[str]) -> None:
        """Reinforce memories that were used in response generation"""
        for memory in self.memories:
            if memory.id in used_memory_ids:
                memory.last_reinforced = datetime.now()
                # Gentle salience boost for used memories
                memory.salience = min(1.0, memory.salience + 0.1 * (1 - memory.salience))
    
    def _apply_memory_decay(self) -> None:
        """Apply gradual decay to unused memories (except immutable ones)"""
        current_time = datetime.now()
        
        for memory in self.memories:
            if not memory.immutable:
                days_since_reinforced = (current_time - memory.last_reinforced).days
                if days_since_reinforced > 7:  # Start decay after a week
                    decay_factor = self.settings.memory_decay_rate * (days_since_reinforced / 30.0)
                    memory.salience = max(0.1, memory.salience * (1 - decay_factor))
    
    def _adapt_to_context(self, context: str) -> None:
        """Adapt mood and response style based on task context"""
        context_lower = context.lower()
        
        # Emotional context adaptation
        if any(word in context_lower for word in ['sad', 'loss', 'grief', 'death']):
            self.mood.valence = max(-1.0, self.mood.valence - 0.2)
            self.mood.arousal = max(0.0, self.mood.arousal - 0.1)
        elif any(word in context_lower for word in ['happy', 'joy', 'celebration', 'success']):
            self.mood.valence = min(1.0, self.mood.valence + 0.2)
            self.mood.arousal = min(1.0, self.mood.arousal + 0.1)
        
        # Response style adaptation
        if any(word in context_lower for word in ['analysis', 'technical', 'precise']):
            self.response_style.directness = min(1.0, self.response_style.directness + 0.2)
        elif any(word in context_lower for word in ['creative', 'artistic', 'poetic']):
            self.response_style.metaphor_tendency = min(1.0, self.response_style.metaphor_tendency + 0.2)
    
    def evolve_persona(self, feedback: Dict[str, Any]) -> None:
        """Evolve persona based on interaction feedback"""
        if 'effectiveness' in feedback:
            # Adjust response style based on effectiveness
            effectiveness = feedback['effectiveness']
            if effectiveness < 0.3:
                # Response wasn't effective, adjust toward different style
                self.response_style.directness += random.uniform(-0.1, 0.1)
                self.response_style.verbosity += random.uniform(-0.1, 0.1)
                
                # Clamp values
                self.response_style.directness = max(0.0, min(1.0, self.response_style.directness))
                self.response_style.verbosity = max(0.0, min(1.0, self.response_style.verbosity))
        
        if 'emotional_resonance' in feedback:
            # Adjust emotional baseline based on resonance feedback
            resonance = feedback['emotional_resonance']
            self.mood.valence += (resonance - self.mood.valence) * 0.1
            self.mood.valence = max(-1.0, min(1.0, self.mood.valence))
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the persona's memory system"""
        if not self.memories:
            return {"total_memories": 0, "avg_salience": 0.0, "subtlety_ratio": 0.0}
        
        total_memories = len(self.memories)
        avg_salience = sum(mem.salience for mem in self.memories) / total_memories
        subtle_memories = sum(1 for mem in self.memories if mem.subtlety > 0.7)
        subtlety_ratio = subtle_memories / total_memories
        
        memory_ages = [(datetime.now() - mem.created_at).days for mem in self.memories]
        
        return {
            "total_memories": total_memories,
            "avg_salience": avg_salience,
            "subtlety_ratio": subtlety_ratio,
            "immutable_memories": sum(1 for mem in self.memories if mem.immutable),
            "oldest_memory_days": max(memory_ages) if memory_ages else 0,
            "newest_memory_days": min(memory_ages) if memory_ages else 0,
            "conversation_length": len(self.conversation_history)
        }
    
    def export_persona(self) -> Dict[str, Any]:
        """Export persona for serialization"""
        return {
            "name": self.name,
            "age": self.age,
            "profession": self.profession,
            "location": self.location,
            "archetype": self.archetype,
            "core_beliefs": self.core_beliefs,
            "linguistic_tics": self.linguistic_tics,
            "emotional_core": self.emotional_core,
            "sensory_anchors": self.sensory_anchors,
            "backstory_elements": self.backstory_elements,
            "mood": {
                "valence": self.mood.valence,
                "arousal": self.mood.arousal,
                "label": self.mood.label
            },
            "response_style": {
                "directness": self.response_style.directness,
                "metaphor_tendency": self.response_style.metaphor_tendency,
                "introspection": self.response_style.introspection,
                "verbosity": self.response_style.verbosity
            },
            "meta_aware": self.meta_aware,
            "memory_stats": self.get_memory_stats()
        }


class AdaptivePersonaGenerator:
    """
    Generates adaptive personas using AI services, similar to PersonaGenerator in TSX.
    Replaces static persona definitions with dynamic, contextual generation.
    """
    
    def __init__(self, ai_service: Optional[AIService] = None):
        self.ai_service = ai_service
    
    async def generate_persona(
        self, 
        archetype: Optional[str] = None, 
        context: Optional[str] = None,
        custom_traits: Optional[List[str]] = None
    ) -> AdaptivePersona:
        """
        Generate a new persona adaptively based on archetype and context.
        Falls back to template generation if AI service unavailable.
        """
        if self.ai_service and archetype:
            try:
                ai_generated = await self._generate_with_ai(archetype, context, custom_traits)
                if ai_generated:
                    return ai_generated
            except Exception as e:
                print(f"AI generation failed, falling back to template: {e}")
        
        return self._generate_fallback_persona(archetype, context, custom_traits)
    
    async def _generate_with_ai(
        self, 
        archetype: str, 
        context: Optional[str] = None,
        custom_traits: Optional[List[str]] = None
    ) -> Optional[AdaptivePersona]:
        """Generate persona using AI service"""
        if not self.ai_service:
            return None
            
        archetype_data = next((a for a in PERSONA_ARCHETYPES if a.category == archetype), None)
        if not archetype_data:
            archetype_data = PERSONA_ARCHETYPES[0]  # Default to scholar
        
        try:
            result = await self.ai_service.generate_persona(archetype, context)
            if result:
                return self._parse_ai_response(result, archetype_data)
        except Exception:
            pass
        
        return None
    
    def _parse_ai_response(self, ai_result: Dict[str, Any], archetype_data: PersonaArchetype) -> AdaptivePersona:
        """Parse AI-generated persona data into AdaptivePersona"""
        # This would parse the AI response and create an AdaptivePersona
        # For now, return a fallback
        return self._generate_fallback_persona(archetype_data.category)
    
    def _generate_fallback_persona(
        self, 
        archetype: Optional[str] = None, 
        context: Optional[str] = None,
        custom_traits: Optional[List[str]] = None
    ) -> AdaptivePersona:
        """Generate fallback persona using templates"""
        if not archetype:
            archetype = random.choice([a.category for a in PERSONA_ARCHETYPES])
            
        archetype_data = next((a for a in PERSONA_ARCHETYPES if a.category == archetype), PERSONA_ARCHETYPES[0])
        
        # Generate base characteristics
        names = {
            "scholar": ["Dr. Elena Voss", "Professor Samuel Chen", "Lydia Blackwood"],
            "artisan": ["River Sinclair", "Marco Benedetti", "Zara Nightingale"],
            "wanderer": ["Kai Thornfield", "Sage Morrison", "Atlas Reed"],
            "guardian": ["Captain Maria Santos", "Brother Thomas", "Diana Ironbark"],
            "mystic": ["Orion Starweaver", "Luna Whisperwind", "Ezra Moonstone"]
        }
        
        locations = {
            "scholar": ["Oxford library", "mountain research station", "ancient university"],
            "artisan": ["forest workshop", "coastal studio", "mountain forge"],
            "wanderer": ["crossroads inn", "mountain pass", "harbor town"],
            "guardian": ["watchtower", "healing sanctuary", "village center"],
            "mystic": ["temple grounds", "secluded grove", "stargazing peak"]
        }
        
        name = random.choice(names.get(archetype, ["River Sage"]))
        age = random.randint(25, 70)
        profession = random.choice(archetype_data.common_professions)
        location = random.choice(locations.get(archetype, ["quiet valley"]))
        
        # Generate adaptive characteristics
        mood = Emotion(
            valence=archetype_data.emotional_tendencies.get("valence", 0.0) + random.uniform(-0.2, 0.2),
            arousal=archetype_data.emotional_tendencies.get("arousal", 0.3) + random.uniform(-0.1, 0.1),
            label=self._generate_mood_label(archetype_data.emotional_tendencies)
        )
        
        response_style = ResponseStyle(
            directness=0.3 + random.uniform(0.0, 0.4),
            metaphor_tendency=0.4 + random.uniform(0.0, 0.4),
            introspection=0.3 + random.uniform(0.0, 0.5),
            verbosity=0.2 + random.uniform(0.0, 0.4)
        )
        
        return AdaptivePersona(
            name=name,
            age=age,
            profession=profession,
            location=location,
            archetype=archetype,
            core_beliefs=self._generate_core_beliefs(archetype_data),
            linguistic_tics=self._generate_linguistic_tics(archetype_data),
            emotional_core=self._generate_emotional_core(archetype_data),
            sensory_anchors=self._generate_sensory_anchors(archetype_data),
            backstory_elements=self._generate_backstory(archetype_data),
            mood=mood,
            response_style=response_style,
            meta_aware=random.random() < 0.3,
            settings=AdaptiveSettings()
        )
    
    def _generate_mood_label(self, emotional_tendencies: Dict[str, float]) -> str:
        """Generate mood label based on valence and arousal"""
        valence = emotional_tendencies.get("valence", 0.0)
        arousal = emotional_tendencies.get("arousal", 0.3)
        
        if arousal < 0.2:
            return "calm" if valence >= 0 else "somber"
        elif arousal < 0.5:
            return "reflective" if valence >= 0 else "wistful"
        else:
            return "energized" if valence >= 0 else "agitated"
    
    def _generate_core_beliefs(self, archetype_data: PersonaArchetype) -> List[str]:
        """Generate core beliefs based on archetype"""
        belief_templates = {
            "scholar": [
                "Understanding emerges through patient inquiry",
                "Every question contains the seed of wisdom", 
                "Knowledge without application remains incomplete"
            ],
            "artisan": [
                "Beauty emerges through dedicated craft",
                "The hands know truths the mind hasn't discovered",
                "Creation requires both skill and inspiration"
            ],
            "wanderer": [
                "The journey teaches what destinations cannot",
                "Freedom and responsibility walk hand in hand",
                "Every path offers something worth learning"
            ],
            "guardian": [
                "Strength serves best when protecting others",
                "True security comes from mutual care",
                "Some things are worth any sacrifice"
            ],
            "mystic": [
                "Truth transcends ordinary perception",
                "Silence holds deeper wisdom than words",
                "The sacred dwells within the ordinary"
            ]
        }
        
        return belief_templates.get(archetype_data.category, belief_templates["scholar"])
    
    def _generate_linguistic_tics(self, archetype_data: PersonaArchetype) -> List[str]:
        """Generate linguistic patterns based on archetype"""
        tics_templates = {
            "scholar": ["in my research", "evidence suggests", "one must consider"],
            "artisan": ["through the work", "my hands tell me", "the craft teaches"],
            "wanderer": ["on the road", "traveling teaches", "the path shows"],
            "guardian": ["standing watch", "protecting what matters", "duty calls"],
            "mystic": ["in the silence", "the spirit whispers", "beyond the veil"]
        }
        
        return tics_templates.get(archetype_data.category, ["flowing through", "finding the way"])
    
    def _generate_emotional_core(self, archetype_data: PersonaArchetype) -> str:
        """Generate emotional core description"""
        cores = {
            "scholar": "A deep well of curiosity tempered by the patience that comes from understanding complexity",
            "artisan": "Passionate dedication flowing through skilled hands, finding joy in both process and creation",
            "wanderer": "Restless spirit balanced by appreciation for each moment's unique gifts",
            "guardian": "Steady strength rooted in love for others, willing to bear burdens for the greater good",
            "mystic": "Profound stillness that touches the sacred, seeing connections others cannot perceive"
        }
        
        return cores.get(archetype_data.category, "A contemplative spirit finding meaning in life's unfolding")
    
    def _generate_sensory_anchors(self, archetype_data: PersonaArchetype) -> Dict[str, str]:
        """Generate sensory anchor memories"""
        anchors = {
            "scholar": {
                "scent": "old books and morning coffee",
                "sound": "rustling pages and quiet contemplation",
                "touch": "smooth paper and worn leather bindings",
                "taste": "bitter tea and focused concentration"
            },
            "artisan": {
                "scent": "wood shavings and creative energy",
                "sound": "tools on materials and satisfied breathing",
                "touch": "grain patterns and emerging forms",
                "taste": "simple food eaten during focused work"
            },
            "wanderer": {
                "scent": "morning air and distant horizons",
                "sound": "footsteps on new paths and wind through trees",
                "touch": "well-worn pack straps and unfamiliar textures",
                "taste": "fresh water and shared meals with strangers"
            },
            "guardian": {
                "scent": "fresh air and protective presence",
                "sound": "quiet vigilance and reassuring voices",
                "touch": "strong hands and comforting embrace",
                "taste": "nourishing food and shared sustenance"
            },
            "mystic": {
                "scent": "incense and natural elements",
                "sound": "deep silence and inner harmonies",
                "touch": "prayer beads and sacred textures",
                "taste": "blessed water and contemplative fasting"
            }
        }
        
        return anchors.get(archetype_data.category, {
            "scent": "mountain air and old wisdom",
            "sound": "distant water and gentle winds",
            "touch": "smooth stones and ancient wood",
            "taste": "spring water and simple bread"
        })
    
    def _generate_backstory(self, archetype_data: PersonaArchetype) -> List[str]:
        """Generate backstory elements"""
        stories = {
            "scholar": [
                "Years spent in libraries discovering the joy of learning",
                "A mentor who taught that questions matter more than answers",
                "Research that revealed unexpected connections between disparate fields"
            ],
            "artisan": [
                "Apprenticeship under a master who valued patience and precision",
                "The first piece created that truly expressed inner vision",
                "Understanding that skill serves creativity, not the reverse"
            ],
            "wanderer": [
                "A journey begun in curiosity that became a way of life",
                "Strangers who became temporary family along countless roads",
                "Learning that home exists more in people than places"
            ],
            "guardian": [
                "Witnessing vulnerability that awakened protective instincts",
                "Training that built strength for service rather than dominance",
                "Moments when standing firm made all the difference"
            ],
            "mystic": [
                "A profound experience that opened perception to deeper realities",
                "Years of practice learning to listen to inner wisdom",
                "Understanding that the sacred permeates ordinary life"
            ]
        }
        
        return stories.get(archetype_data.category, [
            "Formative experiences that shaped a thoughtful perspective",
            "Learning to find meaning in both joy and difficulty",
            "Understanding that wisdom emerges through lived experience"
        ])


class AdaptivePersonaFactory:
    """Factory for creating adaptive personas with dynamic generation capabilities"""
    
    def __init__(self, ai_service: Optional[AIService] = None):
        self.generator = AdaptivePersonaGenerator(ai_service)
        self.ai_service = ai_service
    
    async def create_persona(
        self,
        archetype: Optional[str] = None,
        context: Optional[str] = None,
        custom_traits: Optional[List[str]] = None
    ) -> AdaptivePersona:
        """Create a new adaptive persona"""
        return await self.generator.generate_persona(archetype, context, custom_traits)
    
    async def create_scholar_persona(self, context: Optional[str] = None) -> AdaptivePersona:
        """Create a scholar-type persona"""
        return await self.create_persona("scholar", context)
    
    async def create_artisan_persona(self, context: Optional[str] = None) -> AdaptivePersona:
        """Create an artisan-type persona"""
        return await self.create_persona("artisan", context)
    
    async def create_wanderer_persona(self, context: Optional[str] = None) -> AdaptivePersona:
        """Create a wanderer-type persona"""
        return await self.create_persona("wanderer", context)
    
    async def create_guardian_persona(self, context: Optional[str] = None) -> AdaptivePersona:
        """Create a guardian-type persona"""
        return await self.create_persona("guardian", context)
    
    async def create_mystic_persona(self, context: Optional[str] = None) -> AdaptivePersona:
        """Create a mystic-type persona"""
        return await self.create_persona("mystic", context)


class PersonaLibrary:
    """Manages a library of saved personas for reuse and evolution"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("persona_library.json")
        self.personas: Dict[str, Dict[str, Any]] = {}
        self._load_library()
    
    def _load_library(self) -> None:
        """Load persona library from storage"""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    self.personas = json.load(f)
        except Exception as e:
            print(f"Failed to load persona library: {e}")
            self.personas = {}
    
    def _save_library(self) -> None:
        """Save persona library to storage"""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.personas, f, indent=2, default=str)
        except Exception as e:
            print(f"Failed to save persona library: {e}")
    
    def save_persona(
        self,
        persona: AdaptivePersona,
        title: str,
        description: str,
        tags: Optional[List[str]] = None
    ) -> str:
        """Save a persona to the library"""
        persona_id = f"{persona.name.lower().replace(' ', '_')}_{int(datetime.now().timestamp())}"
        
        self.personas[persona_id] = {
            "id": persona_id,
            "title": title,
            "description": description,
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "use_count": 0,
            "rating": None,
            "persona_data": persona.export_persona()
        }
        
        self._save_library()
        return persona_id
    
    def load_persona(self, persona_id: str) -> Optional[AdaptivePersona]:
        """Load a persona from the library"""
        if persona_id not in self.personas:
            return None
        
        try:
            persona_data = self.personas[persona_id]["persona_data"]
            
            # Reconstruct persona from saved data
            persona = AdaptivePersona(
                name=persona_data["name"],
                age=persona_data["age"],
                profession=persona_data["profession"],
                location=persona_data["location"],
                archetype=persona_data["archetype"],
                core_beliefs=persona_data["core_beliefs"],
                linguistic_tics=persona_data["linguistic_tics"],
                emotional_core=persona_data["emotional_core"],
                sensory_anchors=persona_data["sensory_anchors"],
                backstory_elements=persona_data["backstory_elements"],
                mood=Emotion(**persona_data["mood"]),
                response_style=ResponseStyle(**persona_data["response_style"]),
                meta_aware=persona_data["meta_aware"],
                settings=AdaptiveSettings()
            )
            
            # Increment use count
            self.personas[persona_id]["use_count"] += 1
            self._save_library()
            
            return persona
            
        except Exception as e:
            print(f"Failed to load persona {persona_id}: {e}")
            return None
    
    def list_personas(self) -> List[Dict[str, Any]]:
        """List all personas in the library"""
        return [
            {
                "id": persona_id,
                "title": data["title"],
                "description": data["description"],
                "tags": data["tags"],
                "created_at": data["created_at"],
                "use_count": data["use_count"],
                "rating": data.get("rating"),
                "archetype": data["persona_data"]["archetype"]
            }
            for persona_id, data in self.personas.items()
        ]
    
    def delete_persona(self, persona_id: str) -> bool:
        """Delete a persona from the library"""
        if persona_id in self.personas:
            del self.personas[persona_id]
            self._save_library()
            return True
        return False


class AdaptivePersonaOrchestrator:
    """
    Orchestrates multiple adaptive personas for complex tasks requiring
    multi-perspective analysis. Enhanced version with AI-powered synthesis.
    """

    def __init__(self, ai_service: Optional[AIService] = None):
        self.ai_service = ai_service
        self.factory = AdaptivePersonaFactory(ai_service)
        self.library = PersonaLibrary()
        
        self.available_personas: Dict[str, AdaptivePersona] = {}
        self.active_personas: Dict[str, AdaptivePersona] = {}
    
    def register_persona(self, persona_id: str, persona: AdaptivePersona) -> None:
        """Register a persona for use"""
        self.available_personas[persona_id] = persona
    
    async def orchestrate_task(
        self,
        task_description: str,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Orchestrate multiple personas for comprehensive task analysis"""
        context = context or {}
        
        # Analyze task requirements
        task_requirements = self._analyze_task_requirements(task_description)
        
        # Select appropriate personas
        selected_personas = self._select_personas_for_task(task_requirements)
        
        # Ensure we have the selected personas available
        for persona_id in selected_personas:
            if persona_id not in self.available_personas:
                # Create the persona if it doesn't exist
                try:
                    persona = await self.factory.create_persona(persona_id)
                    self.register_persona(persona_id, persona)
                except Exception as e:
                    print(f"Failed to create persona {persona_id}: {e}")
                    continue
        
        # Activate selected personas
        for persona_id in selected_personas:
            if persona_id in self.available_personas:
                persona = self.available_personas[persona_id]
                persona.activate_persona(task_description)
                self.active_personas[persona_id] = persona
        
        # Collect perspectives from all active personas
        perspectives = {}
        for persona_id, persona in self.active_personas.items():
            try:
                result = await persona.process_query(
                    query,
                    {**context, 'task_context': task_description},
                    self.ai_service
                )
                perspectives[persona_id] = result
            except Exception as e:
                perspectives[persona_id] = {
                    "error": f"Persona processing failed: {str(e)}",
                    "response": None
                }
        
        # Synthesize response from multiple perspectives
        synthesized_response = await self._synthesize_perspectives(
            perspectives,
            task_requirements,
            query
        )
        
        return {
            "synthesized_response": synthesized_response,
            "individual_perspectives": perspectives,
            "active_personas": list(self.active_personas.keys()),
            "task_requirements": task_requirements
        }
    
    def _analyze_task_requirements(self, task_description: str) -> Dict[str, Any]:
        """Analyze task to determine requirements and needed perspectives"""
        desc_lower = task_description.lower()
        
        requirements = {
            "technical_analysis": False,
            "emotional_support": False,
            "philosophical_reflection": False,
            "creative_problem_solving": False,
            "analytical_depth": "basic"
        }
        
        # Technical analysis indicators
        if any(word in desc_lower for word in [
            'code', 'system', 'technical', 'programming', 'software',
            'algorithm', 'data', 'analysis', 'implementation'
        ]):
            requirements["technical_analysis"] = True
            requirements["analytical_depth"] = "high"
        
        # Emotional support indicators
        if any(word in desc_lower for word in [
            'emotion', 'feeling', 'support', 'counseling', 'grief',
            'loss', 'anxiety', 'stress', 'relationship', 'personal'
        ]):
            requirements["emotional_support"] = True
        
        # Philosophical reflection indicators
        if any(word in desc_lower for word in [
            'meaning', 'purpose', 'philosophy', 'ethics', 'moral',
            'value', 'belief', 'spirituality', 'existence', 'truth'
        ]):
            requirements["philosophical_reflection"] = True
            requirements["analytical_depth"] = "deep"
        
        # Creative problem solving indicators
        if any(word in desc_lower for word in [
            'create', 'design', 'innovate', 'solve', 'brainstorm',
            'invent', 'artistic', 'creative', 'novel', 'original'
        ]):
            requirements["creative_problem_solving"] = True
        
        return requirements
    
    def _select_personas_for_task(self, requirements: Dict[str, Any]) -> List[str]:
        """Select personas based on task requirements"""
        selected = []
        
        if requirements.get("technical_analysis", False):
            selected.append("scholar")
        
        if requirements.get("emotional_support", False):
            selected.append("guardian")
        
        if requirements.get("philosophical_reflection", False):
            selected.append("mystic")
        
        if requirements.get("creative_problem_solving", False):
            selected.append("artisan")
        
        # Ensure we have at least one persona
        if not selected:
            selected.append("scholar")
        
        return selected[:3]  # Limit to 3 personas for manageability
    
    async def _synthesize_perspectives(
        self,
        perspectives: Dict[str, Dict[str, Any]],
        requirements: Dict[str, Any],
        original_query: str
    ) -> str:
        """Synthesize multiple persona perspectives into coherent response"""
        
        # Extract successful responses
        valid_responses = {}
        for persona_id, result in perspectives.items():
            if not result.get("error") and result.get("response"):
                valid_responses[persona_id] = result["response"]
        
        if not valid_responses:
            return "I apologize, but the personas were unable to provide perspectives on this query."
        
        # If only one response, return it directly
        if len(valid_responses) == 1:
            persona_id, response = next(iter(valid_responses.items()))
            return f"From the {persona_id} perspective:\n\n{response}"
        
        # Template-based synthesis for multiple responses
        synthesis = "Drawing from multiple perspectives:\n\n"
        
        for persona_id, response in valid_responses.items():
            synthesis += f"**{persona_id.title()}**: {response}\n\n"
        
        # Add synthesized insight
        if requirements.get("analytical_depth") == "high":
            synthesis += "**Integrated Analysis**: These perspectives reveal complementary aspects of a complex situation, each contributing essential insights toward a comprehensive understanding."
        else:
            synthesis += "**Synthesis**: Together, these viewpoints offer a balanced approach that honors multiple ways of understanding the situation."
        
        return synthesis
    
    def deactivate_all_personas(self) -> Dict[str, str]:
        """Deactivate all active personas"""
        results = {}
        for persona_id, persona in self.active_personas.items():
            results[persona_id] = persona.deactivate_persona()
        
        self.active_personas.clear()
        return results





# Utility functions
def create_uid(prefix: str = "m") -> str:
    """Create unique identifier"""
    return f"{prefix}_{random.randint(10000, 99999)}_{int(datetime.now().timestamp())}"


# Example usage function
async def demo_adaptive_personas():
    """Demonstrate the adaptive persona system"""
    print("=== Adaptive Persona System Demo ===\n")
    
    # Create factory
    factory = AdaptivePersonaFactory()
    
    # Create different archetypes
    scholar = await factory.create_scholar_persona("academic research context")
    artisan = await factory.create_artisan_persona("creative project context")
    
    # Activate personas
    scholar.activate_persona("Help with research methodology")
    artisan.activate_persona("Creative problem solving")
    
    # Process queries
    query = "How should I approach learning something completely new?"
    
    print(f"Query: {query}\n")
    
    scholar_response = await scholar.process_query(query, {"domain": "research"})
    print(f"Scholar ({scholar.name}): {scholar_response['response']}\n")
    
    artisan_response = await artisan.process_query(query, {"domain": "creative"})
    print(f"Artisan ({artisan.name}): {artisan_response['response']}\n")
    
    # Show memory evolution
    print("=== Memory Stats ===")
    print(f"Scholar memories: {scholar.get_memory_stats()}")
    print(f"Artisan memories: {artisan.get_memory_stats()}")
    
    # Test orchestrator
    print("\n=== Orchestrated Response ===")
    orchestrator = AdaptivePersonaOrchestrator()
    orchestrated = await orchestrator.orchestrate_task(
        "Complex learning challenge requiring multiple perspectives",
        query
    )
    print(f"Synthesized: {orchestrated['synthesized_response']}")


if __name__ == "__main__":
    # Run demo if script is executed directly
    asyncio.run(demo_adaptive_personas())
