from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr

from .llm_service import llm_service
from .models import (
    CharacterMapping,
    CleaningOptions,
    LabelFormat,
    ModelSource,
    NarratorAttribution,
    ProcessingConfig,
    SpeakerConfig,
    SpeakerMode,
)
from .text_processor import TextProcessor

try:
    from ebooklib import epub  # type: ignore
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # noqa: BLE001 - import error will be surfaced during use
    epub = None  # type: ignore
    BeautifulSoup = None  # type: ignore


processor = TextProcessor()


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_epub(path: Path) -> str:
    if epub is None or BeautifulSoup is None:
        raise RuntimeError("ebooklib and beautifulsoup4 are required for EPUB processing")
    book = epub.read_epub(str(path))
    texts: List[str] = []
    for item in book.get_items():
        if item.get_type() == epub.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_body_content(), "html.parser")
            texts.append(soup.get_text(" "))
    return "\n".join(texts)


def _load_file(file: Optional[gr.File]) -> Tuple[str, str]:
    if file is None:
        raise gr.Error("No file uploaded")
    path = Path(file.name)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        text = _read_txt(path)
    elif suffix == ".epub":
        text = _read_epub(path)
    else:
        raise gr.Error("Unsupported file type. Please upload a .txt or .epub file.")
    word_count = len(text.split())
    char_count = len(text)
    return text, f"Words: {word_count:,} | Characters: {char_count:,}"


def _build_cleaning_options(
    replace_smart_quotes: bool,
    fix_ocr_errors: bool,
    correct_spelling: bool,
    remove_urls: bool,
    remove_footnotes: bool,
    add_punctuation: bool,
    fix_hyphenation: bool,
) -> CleaningOptions:
    return CleaningOptions(
        replace_smart_quotes=replace_smart_quotes,
        fix_ocr_errors=fix_ocr_errors,
        correct_spelling=correct_spelling,
        remove_urls=remove_urls,
        remove_footnotes=remove_footnotes,
        add_punctuation=add_punctuation,
        fix_hyphenation=fix_hyphenation,
    )


def _parse_character_mapping(raw: str) -> List[CharacterMapping]:
    mappings: List[CharacterMapping] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        if "=" not in line:
            raise gr.Error("Invalid character mapping format. Use `Name = SpeakerNumber` per line.")
        name, number = line.split("=", 1)
        mappings.append(CharacterMapping(name=name.strip(), speaker_number=int(number.strip())))
    return mappings


def _build_speaker_config(
    mode: str,
    speaker_count: int,
    label_format: str,
    include_narrator: bool,
    narrator_attribution: str,
    sample_size: int,
    narrator_name: str,
    mapping_text: str,
) -> Optional[SpeakerConfig]:
    selected_mode = SpeakerMode(mode)
    if selected_mode == SpeakerMode.NONE:
        return None
    config = SpeakerConfig(
        mode=selected_mode,
        speaker_count=speaker_count,
        label_format=LabelFormat(label_format),
        include_narrator=include_narrator,
        narrator_attribution=NarratorAttribution(narrator_attribution),
        sample_size=sample_size,
        narrator_character_name=narrator_name or None,
    )
    if mapping_text.strip():
        config.character_mapping = _parse_character_mapping(mapping_text)
    return config


def _build_processing_config(
    text: str,
    options: CleaningOptions,
    batch_size: int,
    model_source: str,
    model_name: str,
    temperature: float,
    llm_cleaning_disabled: bool,
    custom_instructions: str,
    single_pass: bool,
    extended_examples: bool,
    speaker_config: Optional[SpeakerConfig],
) -> ProcessingConfig:
    if not text.strip():
        raise gr.Error("Please provide text to process")
    return ProcessingConfig(
        batch_size=batch_size,
        cleaning_options=options,
        speaker_config=speaker_config,
        model_source=ModelSource(model_source),
        model_name=model_name,
        temperature=temperature,
        llm_cleaning_disabled=llm_cleaning_disabled,
        custom_instructions=custom_instructions or None,
        single_pass=single_pass,
        extended_examples=extended_examples,
    )


def _format_summary(summary) -> str:
    lines = [
        f"Total chunks: {summary.total_chunks}",
        f"Input tokens: {summary.total_input_tokens:,}",
        f"Output tokens: {summary.total_output_tokens:,}",
        f"Estimated cost: ${summary.total_cost:,.4f}",
    ]
    if summary.applied_cleaning_steps:
        lines.append("Applied steps: " + ", ".join(summary.applied_cleaning_steps))
    return "\n".join(lines)


def run_deterministic(
    text: str,
    replace_smart_quotes: bool,
    fix_ocr_errors: bool,
    correct_spelling: bool,
    remove_urls: bool,
    remove_footnotes: bool,
    add_punctuation: bool,
    fix_hyphenation: bool,
) -> Tuple[str, str, str]:
    if not text.strip():
        raise gr.Error("Please provide text to clean")
    options = _build_cleaning_options(
        replace_smart_quotes,
        fix_ocr_errors,
        correct_spelling,
        remove_urls,
        remove_footnotes,
        add_punctuation,
        fix_hyphenation,
    )
    summary = processor.deterministic_clean(text, options)
    return summary.text, _format_summary(summary), ""


def run_processing(
    text: str,
    replace_smart_quotes: bool,
    fix_ocr_errors: bool,
    correct_spelling: bool,
    remove_urls: bool,
    remove_footnotes: bool,
    add_punctuation: bool,
    fix_hyphenation: bool,
    batch_size: int,
    model_source: str,
    model_name: str,
    temperature: float,
    llm_cleaning_disabled: bool,
    custom_instructions: str,
    single_pass: bool,
    extended_examples: bool,
    speaker_mode: str,
    speaker_count: int,
    label_format: str,
    include_narrator: bool,
    narrator_attribution: str,
    sample_size: int,
    narrator_name: str,
    mapping_text: str,
) -> Tuple[str, str, str]:
    options = _build_cleaning_options(
        replace_smart_quotes,
        fix_ocr_errors,
        correct_spelling,
        remove_urls,
        remove_footnotes,
        add_punctuation,
        fix_hyphenation,
    )
    speaker_config = _build_speaker_config(
        speaker_mode,
        speaker_count,
        label_format,
        include_narrator,
        narrator_attribution,
        sample_size,
        narrator_name,
        mapping_text,
    )
    config = _build_processing_config(
        text,
        options,
        batch_size,
        model_source,
        model_name,
        temperature,
        llm_cleaning_disabled,
        custom_instructions,
        single_pass,
        extended_examples,
        speaker_config,
    )
    summary = processor.process_text(text, config)
    logs = "\n".join(summary.logs)
    return summary.text, _format_summary(summary), logs


def configure_token(token: str) -> str:
    llm_service.set_api_token(token.strip() or None)
    if token.strip():
        return "HuggingFace token configured"
    return "Token cleared"


def create_app() -> gr.Blocks:
    with gr.Blocks(title="VoiceForge Gradio") as demo:
        gr.Markdown(
            """
            # VoiceForge – Python & Gradio Edition

            Upload text or EPUB files, clean them with deterministic rules, and optionally call HuggingFace models for dialogue
            formatting and multi-speaker labelling. Configure cleaning options, batching, speaker behaviour, and API credentials
            directly from this interface.
            """
        )

        with gr.Accordion("HuggingFace API", open=False):
            token_box = gr.Textbox(
                label="HuggingFace API Token",
                value=os.getenv("HUGGINGFACE_API_TOKEN", ""),
                type="password",
            )
            token_status = gr.Markdown("Token status: configured" if os.getenv("HUGGINGFACE_API_TOKEN") else "Token status: missing")
            token_button = gr.Button("Apply Token")
            token_button.click(configure_token, inputs=[token_box], outputs=[token_status])

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(label="Upload .txt or .epub")
                load_button = gr.Button("Load File")
                file_stats = gr.Markdown()
            with gr.Column(scale=2):
                text_input = gr.Textbox(label="Input Text", lines=18)

        load_button.click(_load_file, inputs=[file_input], outputs=[text_input, file_stats])

        gr.Markdown("## Cleaning Options")
        with gr.Row():
            replace_smart_quotes = gr.Checkbox(True, label="Replace smart quotes")
            fix_ocr_errors = gr.Checkbox(True, label="Fix OCR errors")
            correct_spelling = gr.Checkbox(False, label="Correct spelling")
            remove_urls = gr.Checkbox(True, label="Remove URLs")
            remove_footnotes = gr.Checkbox(True, label="Remove footnotes")
            add_punctuation = gr.Checkbox(True, label="Add punctuation")
            fix_hyphenation = gr.Checkbox(False, label="Fix hyphenation")

        gr.Markdown("## Speaker Configuration")
        with gr.Row():
            speaker_mode = gr.Dropdown(
                choices=[m.value for m in SpeakerMode],
                value=SpeakerMode.NONE.value,
                label="Mode",
            )
            label_format = gr.Dropdown(
                choices=[f.value for f in LabelFormat],
                value=LabelFormat.SPEAKER.value,
                label="Label Format",
            )
            speaker_count = gr.Slider(1, 10, value=2, step=1, label="Speaker Count")
            include_narrator = gr.Checkbox(False, label="Include Narrator")
            narrator_attribution = gr.Dropdown(
                choices=[n.value for n in NarratorAttribution],
                value=NarratorAttribution.REMOVE.value,
                label="Narrator Attribution",
            )
            sample_size = gr.Slider(5, 160, value=50, step=5, label="Sample Size")
        narrator_name = gr.Textbox(label="Narrator Character Name", placeholder="Optional")
        mapping_text = gr.Textbox(
            label="Character Mapping",
            placeholder="One mapping per line, e.g. Alice = 1",
            lines=3,
        )

        gr.Markdown("## Processing Configuration")
        with gr.Row():
            batch_size = gr.Slider(1, 20, value=10, step=1, label="Batch size (sentences per chunk)")
            model_source = gr.Dropdown(
                choices=[ModelSource.API.value],
                value=ModelSource.API.value,
                label="Model Source",
            )
            model_name = gr.Textbox(
                value="meta-llama/Meta-Llama-3.1-8B-Instruct",
                label="Model Name",
            )
            temperature = gr.Slider(0.0, 1.5, value=0.3, step=0.05, label="Temperature")
            llm_cleaning_disabled = gr.Checkbox(False, label="Disable LLM cleaning")
        custom_instructions = gr.Textbox(label="Custom Instructions", lines=3)
        with gr.Row():
            single_pass = gr.Checkbox(False, label="Single-pass speaker formatting")
            extended_examples = gr.Checkbox(False, label="Extended examples")

        gr.Markdown("## Run")
        with gr.Row():
            deterministic_button = gr.Button("Deterministic Clean Only", variant="secondary")
            process_button = gr.Button("Process with LLM", variant="primary")

        output_text = gr.Textbox(label="Processed Text", lines=18)
        summary_box = gr.Textbox(label="Summary", lines=4)
        log_box = gr.Textbox(label="Logs", lines=6)

        deterministic_button.click(
            run_deterministic,
            inputs=[
                text_input,
                replace_smart_quotes,
                fix_ocr_errors,
                correct_spelling,
                remove_urls,
                remove_footnotes,
                add_punctuation,
                fix_hyphenation,
            ],
            outputs=[output_text, summary_box, log_box],
        )

        process_button.click(
            run_processing,
            inputs=[
                text_input,
                replace_smart_quotes,
                fix_ocr_errors,
                correct_spelling,
                remove_urls,
                remove_footnotes,
                add_punctuation,
                fix_hyphenation,
                batch_size,
                model_source,
                model_name,
                temperature,
                llm_cleaning_disabled,
                custom_instructions,
                single_pass,
                extended_examples,
                speaker_mode,
                speaker_count,
                label_format,
                include_narrator,
                narrator_attribution,
                sample_size,
                narrator_name,
                mapping_text,
            ],
            outputs=[output_text, summary_box, log_box],
        )

    return demo
