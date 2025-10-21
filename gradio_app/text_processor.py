from __future__ import annotations

import re
import time
from typing import Callable, List, Optional

from .llm_service import LLMService, ProcessOptions, llm_service
from .models import (
    CleaningOptions,
    ProcessingConfig,
    ProcessingProgress,
    ProcessingSummary,
)
from .text_cleaner import apply_deterministic_cleaning


class TextProcessor:
    def __init__(self, service: LLMService | None = None) -> None:
        self.service = service or llm_service

    def split_into_chunks(self, text: str, batch_size: int) -> List[str]:
        sentences = re.findall(r"[^.!?]+(?:[.!?]+|$)", text) or [text]
        sentences = [s.strip() for s in sentences if s.strip()]
        chunks: List[str] = []
        current: List[str] = []
        for sentence in sentences:
            current.append(sentence)
            if len(current) >= batch_size:
                chunks.append(" ".join(current))
                current = []
        if current:
            chunks.append(" ".join(current))
        return [chunk for chunk in chunks if chunk]

    def process_text(
        self,
        text: str,
        config: ProcessingConfig,
        on_progress: Optional[Callable[[ProcessingProgress], None]] = None,
    ) -> ProcessingSummary:
        chunks = self.split_into_chunks(text, config.batch_size)
        total_chunks = len(chunks)
        processed: List[str] = []
        applied_steps: List[str] = []
        logs: List[str] = []

        total_input_tokens = 0
        total_output_tokens = 0
        total_cost = 0.0

        durations_total = 0.0

        for idx, chunk in enumerate(chunks):
            start_time = time.perf_counter()
            retry_count = 0
            success = False
            processed_chunk = chunk

            while retry_count < 2 and not success:
                try:
                    options = ProcessOptions(
                        text=chunk,
                        cleaning_options=config.cleaning_options,
                        speaker_config=config.speaker_config,
                        model_source=config.model_source,
                        model_name=config.model_name,
                        ollama_model_name=config.ollama_model_name,
                        temperature=config.temperature,
                        custom_instructions=config.custom_instructions,
                        single_pass=config.single_pass,
                        llm_cleaning_disabled=config.llm_cleaning_disabled,
                        extended_examples=config.extended_examples,
                    )
                    result = self.service.process_chunk(options)
                    processed_chunk = result.text
                    total_input_tokens += result.input_tokens
                    total_output_tokens += result.output_tokens
                    total_cost += result.input_cost + result.output_cost
                    applied_steps.extend(result.applied_steps)
                    success = self.service.validate_output(chunk, processed_chunk)
                except Exception as exc:  # noqa: BLE001 - surface any errors
                    retry_count += 1
                    logs.append(f"Chunk {idx + 1}: {type(exc).__name__}: {exc}")
                    if retry_count >= 2:
                        processed_chunk = chunk
                        logs.append(f"Chunk {idx + 1}: falling back to original text after errors.")
                        break
                    continue

                if not success:
                    retry_count += 1
                    time.sleep(1)

            processed.append(processed_chunk)
            elapsed = (time.perf_counter() - start_time) * 1000
            durations_total += elapsed
            avg_chunk_ms = durations_total / (idx + 1)
            remaining = total_chunks - (idx + 1)
            eta_ms = max(0, int(avg_chunk_ms * remaining))

            progress = ProcessingProgress(
                chunk_index=idx,
                processed_text=processed_chunk,
                status="success" if success else "failed",
                retry_count=retry_count,
                last_chunk_ms=int(elapsed),
                avg_chunk_ms=avg_chunk_ms,
                eta_ms=eta_ms,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                total_input_tokens=total_input_tokens,
                total_output_tokens=total_output_tokens,
                total_cost=total_cost,
            )
            if on_progress:
                on_progress(progress)

        final_text = "\n\n".join(processed)
        return ProcessingSummary(
            text=final_text,
            total_chunks=total_chunks,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_cost=total_cost,
            applied_cleaning_steps=applied_steps,
            logs=logs,
        )

    def deterministic_clean(self, text: str, options: CleaningOptions) -> ProcessingSummary:
        result = apply_deterministic_cleaning(text, options)
        return ProcessingSummary(
            text=result.text,
            total_chunks=1,
            total_input_tokens=0,
            total_output_tokens=0,
            total_cost=0.0,
            applied_cleaning_steps=result.applied,
            logs=[],
        )
