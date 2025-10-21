from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class SpeakerMode(str, Enum):
    NONE = "none"
    FORMAT = "format"
    INTELLIGENT = "intelligent"


class LabelFormat(str, Enum):
    SPEAKER = "speaker"
    BRACKET = "bracket"


class NarratorAttribution(str, Enum):
    REMOVE = "remove"
    VERBATIM = "verbatim"
    CONTEXTUAL = "contextual"


class ModelSource(str, Enum):
    API = "api"
    OLLAMA = "ollama"


@dataclass
class CleaningOptions:
    replace_smart_quotes: bool = True
    fix_ocr_errors: bool = True
    correct_spelling: bool = False
    remove_urls: bool = True
    remove_footnotes: bool = True
    add_punctuation: bool = True
    fix_hyphenation: bool = False


@dataclass
class CharacterMapping:
    name: str
    speaker_number: int


@dataclass
class SpeakerConfig:
    mode: SpeakerMode = SpeakerMode.NONE
    speaker_count: int = 2
    label_format: LabelFormat = LabelFormat.SPEAKER
    speaker_mapping: Dict[str, str] = field(default_factory=dict)
    extract_characters: bool = False
    sample_size: int = 50
    include_narrator: bool = False
    narrator_attribution: NarratorAttribution = NarratorAttribution.REMOVE
    character_mapping: List[CharacterMapping] = field(default_factory=list)
    narrator_character_name: Optional[str] = None

    def is_enabled(self) -> bool:
        return self.mode != SpeakerMode.NONE


@dataclass
class ProcessingConfig:
    batch_size: int = 10
    cleaning_options: CleaningOptions = field(default_factory=CleaningOptions)
    speaker_config: Optional[SpeakerConfig] = None
    model_source: ModelSource = ModelSource.API
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    ollama_model_name: Optional[str] = None
    temperature: float = 0.3
    llm_cleaning_disabled: bool = False
    custom_instructions: Optional[str] = None
    single_pass: bool = False
    extended_examples: bool = False


@dataclass
class ProcessChunkResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    input_cost: float = 0.0
    output_cost: float = 0.0
    applied_steps: List[str] = field(default_factory=list)


@dataclass
class ProcessingProgress:
    chunk_index: int
    processed_text: str
    status: str
    retry_count: int
    last_chunk_ms: Optional[int] = None
    avg_chunk_ms: Optional[float] = None
    eta_ms: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    input_cost: Optional[float] = None
    output_cost: Optional[float] = None
    total_input_tokens: Optional[int] = None
    total_output_tokens: Optional[int] = None
    total_cost: Optional[float] = None


@dataclass
class ProcessingSummary:
    text: str
    total_chunks: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float
    applied_cleaning_steps: List[str]
    logs: List[str]
