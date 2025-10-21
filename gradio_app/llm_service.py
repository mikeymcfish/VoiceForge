from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import requests

from huggingface_hub import InferenceClient

from .models import (
    CleaningOptions,
    LabelFormat,
    ModelSource,
    NarratorAttribution,
    ProcessChunkResult,
    SpeakerConfig,
    SpeakerMode,
)
from .text_cleaner import apply_deterministic_cleaning

LOGGER = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Simple heuristic: 1 token ~= 4 characters
    return max(1, len(text.strip()) // 4)


@dataclass
class ProcessOptions:
    text: str
    cleaning_options: CleaningOptions
    speaker_config: Optional[SpeakerConfig] = None
    model_source: ModelSource = ModelSource.API
    model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    ollama_model_name: Optional[str] = None
    temperature: float = 0.3
    custom_instructions: Optional[str] = None
    single_pass: bool = False
    llm_cleaning_disabled: bool = False
    extended_examples: bool = False


class LLMService:
    """Minimal Python port of the TypeScript LLM orchestration layer."""

    def __init__(self) -> None:
        self._client: Optional[InferenceClient] = None
        self._token: Optional[str] = None
        self._update_client_from_env()

    # ------------------------------------------------------------------
    # Token/Client helpers
    # ------------------------------------------------------------------
    def _update_client_from_env(self) -> None:
        token = (
            os.getenv("HUGGINGFACE_API_TOKEN")
            or os.getenv("HF_TOKEN")
            or None
        )
        if token != self._token:
            self._token = token
            self._client = InferenceClient(token=token) if token else None
            LOGGER.info("Configured HuggingFace InferenceClient: %s", "yes" if token else "no token")

    def set_api_token(self, token: Optional[str]) -> None:
        if token:
            os.environ["HUGGINGFACE_API_TOKEN"] = token
        elif "HUGGINGFACE_API_TOKEN" in os.environ:
            del os.environ["HUGGINGFACE_API_TOKEN"]
        self._update_client_from_env()

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------
    def _build_cleaning_prompt(
        self,
        text: str,
        options: CleaningOptions,
        custom_instructions: Optional[str],
    ) -> str:
        tasks = []
        if options.replace_smart_quotes:
            tasks.append("* Replace smart quotes with standard ASCII quotes.")
        if options.fix_ocr_errors:
            tasks.append("* Fix OCR errors such as merged words or missing spaces.")
        if options.fix_hyphenation:
            tasks.append("* Repair hyphenation splits introduced by line breaks.")
        if options.correct_spelling:
            tasks.append("* Correct obvious spelling mistakes and typos.")
        if options.remove_urls:
            tasks.append("* Remove URLs, web links, and email addresses.")
        if options.remove_footnotes:
            tasks.append("* Remove footnote markers, metadata, or bracketed references.")
        if options.add_punctuation:
            tasks.append("* Ensure stray headings or numbers end with appropriate punctuation.")

        prompt = (
            "You are a TTS preprocessing assistant. Clean and repair the text using ONLY the listed transformations.\n\n"
            "Preprocessing Steps:\n"
            + "\n".join(tasks)
            + "\n\nRules:\n"
            "- Preserve the original meaning and paragraph structure.\n"
            "- Only fix errors; do not rewrite or summarize.\n"
            "- Return ONLY the cleaned text with no commentary.\n"
        )

        if custom_instructions:
            prompt += f"\nAdditional custom instructions:\n{custom_instructions.strip()}\n"

        prompt += f"\nText:\n{text}\n\nCleaned:"
        return prompt

    def _build_speaker_prompt(
        self,
        text: str,
        config: SpeakerConfig,
        custom_instructions: Optional[str],
        extended_examples: bool,
    ) -> str:
        label_example = "[1]:" if config.label_format == LabelFormat.BRACKET else "Speaker 1:"
        if config.label_format == LabelFormat.BRACKET:
            instructions = (
                "Identify unique speaking characters. Label them dynamically using bracket format:"
                " the first character is [1]:, the next is [2]:, etc."
            )
        else:
            instructions = (
                "Identify unique speaking characters. Label them dynamically as Speaker 1:, Speaker 2:, etc."
            )

        if config.include_narrator:
            narrator_rule = (
                "All non-quoted narration must use the Narrator: tag; never attribute narration to speakers."
            )
        else:
            narrator_rule = "Omit narration; output only spoken dialogue with speaker labels."

        narrator_identity = ""
        if config.narrator_character_name:
            narrator_identity = (
                "Narrator Identity: The narrator is "
                f"\"{config.narrator_character_name}\"."
            )

        if config.mode == SpeakerMode.FORMAT:
            attribution_rule = (
                "Do not transform attribution tags. Preserve text and punctuation; only add speaker labels."
            )
        else:
            if config.narrator_attribution == NarratorAttribution.REMOVE:
                attribution_rule = (
                    "Remove redundant attribution tags (e.g., he said) because the speaker label replaces them."
                )
            elif config.narrator_attribution == NarratorAttribution.VERBATIM:
                attribution_rule = (
                    "Move attribution tags into a Narrator: line immediately after the spoken line, preserving punctuation."
                )
            else:
                attribution_rule = (
                    "Transform attribution or action tags into concise Narrator: lines, omitting redundant verbs."
                )

        examples = ""
        if extended_examples and config.mode == SpeakerMode.INTELLIGENT:
            names = [mapping.name for mapping in config.character_mapping] or ["Alice", "Bob", "Charlie"]
            speaker_two = "[2]:" if config.label_format == LabelFormat.BRACKET else "Speaker 2:"
            examples = (
                "Examples:\n"
                f"Input: \"Are you coming to the party?\" {names[0]} asked.\n"
                f"Output: {label_example} Are you coming to the party?\n"
                f"Input: \"It's a beautiful day,\" {names[1] if len(names) > 1 else 'Bob'} said, looking up.\n"
                f"Output: {speaker_two} It's a beautiful day.\n"
            )
            if config.include_narrator:
                examples += "Narrator: They looked up at the sky.\n"

        prompt_parts = [
            "You are a dialogue structuring assistant for multi-speaker TTS.",
            instructions,
            narrator_rule,
            "Remove quotation marks from dialogue.",
            attribution_rule,
        ]

        if narrator_identity:
            prompt_parts.append(narrator_identity)
        if custom_instructions:
            prompt_parts.append(f"Additional instructions: {custom_instructions.strip()}")
        if examples:
            prompt_parts.append(examples.strip())

        prompt_parts.append(f"\nText:\n{text}\n\nFormatted:")
        return "\n".join(prompt_parts)

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------
    def process_chunk(self, options: ProcessOptions) -> ProcessChunkResult:
        self._update_client_from_env()
        if options.model_source == ModelSource.API and not self._client:
            raise RuntimeError(
                "HuggingFace API token not configured. Set HUGGINGFACE_API_TOKEN or provide a token in the UI."
            )

        text = options.text
        applied_steps = []
        if not options.llm_cleaning_disabled:
            clean_result = apply_deterministic_cleaning(text, options.cleaning_options, phase="pre")
            text = clean_result.text
            applied_steps.extend(clean_result.applied)

        if options.speaker_config and options.speaker_config.is_enabled() and options.single_pass:
            prompt = self._build_speaker_prompt(
                text,
                options.speaker_config,
                options.custom_instructions,
                options.extended_examples,
            )
        else:
            prompt = self._build_cleaning_prompt(text, options.cleaning_options, options.custom_instructions)

        generated, usage = self._generate_text(prompt, options)

        if options.speaker_config and options.speaker_config.is_enabled():
            if options.single_pass:
                applied_steps.append("llmSpeakerSinglePass")
            else:
                applied_steps.append("llmCleaning")
        else:
            applied_steps.append("llmCleaning")

        if options.speaker_config and options.speaker_config.is_enabled():
            if not options.single_pass:
                speaker_prompt = self._build_speaker_prompt(
                    generated,
                    options.speaker_config,
                    options.custom_instructions,
                    options.extended_examples,
                )
                generated, usage_stage2 = self._generate_text(
                    speaker_prompt, options
                )
                usage = {
                    "input_tokens": usage["input_tokens"] + usage_stage2["input_tokens"],
                    "output_tokens": usage["output_tokens"] + usage_stage2["output_tokens"],
                    "input_cost": usage.get("input_cost", 0.0) + usage_stage2.get("input_cost", 0.0),
                    "output_cost": usage.get("output_cost", 0.0) + usage_stage2.get("output_cost", 0.0),
                }
                applied_steps.append("llmSpeakerFormatting")

        clean_post = apply_deterministic_cleaning(generated, options.cleaning_options, phase="post")
        text = clean_post.text
        applied_steps.extend(clean_post.applied)

        return ProcessChunkResult(
            text=text.strip(),
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            input_cost=usage.get("input_cost", 0.0),
            output_cost=usage.get("output_cost", 0.0),
            applied_steps=applied_steps,
        )

    # ------------------------------------------------------------------
    def _generate_text(self, prompt: str, options: ProcessOptions) -> Tuple[str, Dict[str, float]]:
        if options.model_source == ModelSource.OLLAMA:
            return self._generate_with_ollama(prompt, options)
        return self._generate_with_hf(prompt, options.model_name, options.temperature)

    def _generate_with_hf(self, prompt: str, model_name: str, temperature: float) -> Tuple[str, Dict[str, float]]:
        if not self._client:
            raise RuntimeError("HuggingFace client not configured")
        LOGGER.debug("Generating with HuggingFace model=%s temperature=%s", model_name, temperature)
        start = time.perf_counter()
        response = self._client.text_generation(
            model=model_name,
            prompt=prompt,
            max_new_tokens=512,
            temperature=max(0.0, min(temperature, 1.5)),
            top_p=0.95,
            repetition_penalty=1.05,
            return_full_text=False,
        )
        duration = time.perf_counter() - start
        LOGGER.debug("HuggingFace generation completed in %.2fs", duration)
        text = response.strip() if isinstance(response, str) else str(response)
        usage = {
            "input_tokens": estimate_tokens(prompt),
            "output_tokens": estimate_tokens(text),
            "input_cost": 0.0,
            "output_cost": 0.0,
        }
        return text, usage

    def _generate_with_ollama(self, prompt: str, options: ProcessOptions) -> Tuple[str, Dict[str, float]]:
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        model = (
            options.ollama_model_name
            or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
            or options.model_name
        )
        temperature = max(0.0, min(options.temperature, 2.0)) if options.temperature is not None else 0.3
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "15m",
            "options": {
                "temperature": temperature,
                "num_predict": 2000,
                "num_ctx": 8192,
            },
        }
        url = f"{base_url}/api/generate"
        LOGGER.debug("Generating with Ollama model=%s at %s", model, url)
        try:
            response = requests.post(url, json=payload, timeout=600)
        except requests.RequestException as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to reach Ollama at {url}: {exc}") from exc
        if not response.ok:
            raise RuntimeError(f"Ollama request failed ({response.status_code}): {response.text}")
        text = ""
        try:
            data = response.json()
        except ValueError as exc:  # noqa: BLE001
            raise RuntimeError("Ollama response was not valid JSON") from exc
        text = (
            data.get("response")
            or data.get("final_response")
            or data.get("message", {}).get("content")
            or ""
        )
        if not text.strip():
            chat_payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "keep_alive": "15m",
                "options": payload["options"],
            }
            chat_url = f"{base_url}/api/chat"
            try:
                chat_response = requests.post(chat_url, json=chat_payload, timeout=600)
            except requests.RequestException as exc:  # noqa: BLE001
                raise RuntimeError(f"Failed to reach Ollama chat endpoint at {chat_url}: {exc}") from exc
            if not chat_response.ok:
                raise RuntimeError(
                    f"Ollama chat request failed ({chat_response.status_code}): {chat_response.text}"
                )
            try:
                chat_data = chat_response.json()
            except ValueError as exc:  # noqa: BLE001
                raise RuntimeError("Ollama chat response was not valid JSON") from exc
            text = chat_data.get("message", {}).get("content") or chat_data.get("response", "")
        usage = {
            "input_tokens": estimate_tokens(prompt),
            "output_tokens": estimate_tokens(text),
            "input_cost": 0.0,
            "output_cost": 0.0,
        }
        return text, usage

    # ------------------------------------------------------------------
    def validate_output(self, original: str, generated: str) -> bool:
        if not generated.strip():
            return False
        if generated.strip() == original.strip():
            return True
        return True


llm_service = LLMService()
