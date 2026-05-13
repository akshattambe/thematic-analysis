from dataclasses import dataclass, field
from typing import Optional, List, Dict
import uuid


@dataclass
class Segment:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_file: str = ""
    participant: Optional[str] = None
    text: str = ""
    position: int = 0
    word_count: int = 0


@dataclass
class CodedSegment:
    segment: Segment
    codes: List[str] = field(default_factory=list)
    memo: str = ""


@dataclass
class Quote:
    text: str
    source_file: str
    participant: Optional[str] = None


@dataclass
class Theme:
    id: int
    name: str
    description: str
    codes: List[str]
    representative_quotes: List[Quote]
    coded_segments: List[CodedSegment]

    @property
    def segment_count(self) -> int:
        return len(self.coded_segments)


@dataclass
class AnalysisResult:
    timestamp: str
    source_files: List[str]
    total_segments: int
    coded_segments: List[CodedSegment]
    themes: List[Theme]
    nlp_data: Dict
    metadata: Dict = field(default_factory=dict)
