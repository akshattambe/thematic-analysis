from dataclasses import dataclass, field
from typing import Dict, List, Optional
import uuid

from src.models import Segment


@dataclass
class FirstOrderConcept:
    """First-order concept: stays close to informant's own language."""
    segment: Segment
    concept: str       # label using informant's vocabulary
    verbatim: str      # exact direct quote from the text
    memo: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class SecondOrderTheme:
    """Second-order theme: researcher/theoretical language."""
    id: int
    name: str
    description: str
    first_order_concepts: List[FirstOrderConcept] = field(default_factory=list)

    @property
    def concept_count(self) -> int:
        return len(self.first_order_concepts)


@dataclass
class AggregateDimension:
    """Aggregate dimension: highest level of abstraction."""
    id: int
    name: str
    description: str
    second_order_themes: List[SecondOrderTheme] = field(default_factory=list)

    @property
    def theme_count(self) -> int:
        return len(self.second_order_themes)

    @property
    def concept_count(self) -> int:
        return sum(t.concept_count for t in self.second_order_themes)


@dataclass
class GioiaResult:
    timestamp: str
    source_files: List[str]
    total_segments: int
    first_order_concepts: List[FirstOrderConcept]
    second_order_themes: List[SecondOrderTheme]
    aggregate_dimensions: List[AggregateDimension]
    nlp_data: Dict = field(default_factory=dict)
