"""
Standalone Prompt + Output Tester (Python)

What it does
- Generates a 20‑sentence sample with 3 characters (Alice, Bob, Evelyn) including:
  narration, attribution‑only, attribution+action, merge cases, and first‑person narrator.
- Renders Stage 1 (cleaning) and Stage 2 (speaker formatting) prompts.
- Optionally runs inference using:
  - Ollama (set OLLAMA_BASE_URL, default http://localhost:11434, model env LLM_TEST_OLLAMA=qwen3:8b)
  - HuggingFace Inference (set HUGGINGFACE_API_TOKEN, model env LLM_TEST_MODEL)
- Analyzes outputs and writes a concise report to logs/llm-py-report.txt

Usage examples
- Prompts only:
    python scripts/prompt_test.py
- Run tests with Ollama:
    set OLLAMA_BASE_URL=http://localhost:11434
    set RUN_TESTS=1
    python scripts/prompt_test.py
- Run tests with HF Inference:
    set HUGGINGFACE_API_TOKEN=hf_...
    set LLM_TEST_MODEL=meta-llama/Llama-3.1-8B-Instruct
    set RUN_TESTS=1
    python scripts/prompt_test.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib import request


# -------- Sample + Options --------


def generate_sample_dialogue() -> str:
    n = lambda s: s
    parts = [
        n("It was nearly dusk when I reached the old station."),
        '"We are late," Alice said.',
        'Bob asked, "Did you bring the map?"',
        '"I did," I replied.',  # first‑person narrator exception
        n("The wind rattled the signs along the platform."),
        '"It’s fine," Alice said, looking up at the schedule.',  # attribution + action
        '"We still have time," she added.',
        '"Time for what?" Bob asked.',
        n("I hesitated, remembering the warning."),
        '"Just go, then," I said.',  # first‑person narrator exception inline
        '"Go where?" he said. "Home?"',  # merge case
        '"Home, then work," Alice said. "No more delays."',  # merge two quotes
        n("A light flickered above the exit."),
        '"Did you hear that?" Bob whispered, glancing behind us.',  # attribution + action
        n("For a moment, nobody moved."),
        '"All right," I said, "let’s move."',  # narrator speaking
        '"Fine," Alice replied.',
        '"Fine," Bob replied.',
        n("We hurried down the steps into the street."),
        n("Somewhere, a siren wailed."),
    ]
    return " ".join(parts)


@dataclass
class CleaningOptions:
    replaceSmartQuotes: bool = True
    fixOcrErrors: bool = True
    correctSpelling: bool = False
    removeUrls: bool = True
    removeFootnotes: bool = True
    addPunctuation: bool = True
    fixHyphenation: bool = False


@dataclass
class Character:
    name: str
    speakerNumber: int


@dataclass
class SpeakerConfig:
    mode: str = "intelligent"  # "intelligent" | "format"
    speakerCount: int = 3
    labelFormat: str = "speaker"  # "speaker" | "bracket"
    includeNarrator: bool = True
    narratorAttribution: str = "contextual"  # "remove" | "verbatim" | "contextual"
    characterMapping: List[Character] = None  # list of Characters
    narratorCharacterName: Optional[str] = None


def build_cleaning_prompt(text: str, opt: CleaningOptions) -> str:
    tasks: List[str] = []
    if opt.replaceSmartQuotes:
        tasks.append('* Replace all smart quotes (“ ” ‘ ’) with standard ASCII quotes (" and \").')
    if opt.fixOcrErrors:
        tasks.append("* Fix common OCR errors, such as spacing issues and merged words (e.g., 'thebook' -> 'the book').")
    if opt.fixHyphenation:
        tasks.append("* Fix hyphenation splits (merge words split by line breaks/hyphens).")
    if opt.correctSpelling:
        tasks.append("* Correct common spelling mistakes and typos.")
    if opt.removeUrls:
        tasks.append("* Remove all URLs, web links, and email addresses.")
    if opt.removeFootnotes:
        tasks.append("* Remove footnote markers (e.g., numbers, asterisks) and any other extraneous metadata.")
    if opt.addPunctuation:
        tasks.append("* Ensure appropriate punctuation (like a period) follows any headers or loose numbers for better TTS prosody.")

    prompt = (
        "You are a TTS preprocessing assistant. Clean and repair the text using ONLY the listed transformations.\n\n"
        "Preprocessing Steps:\n" + "\n".join(tasks) + "\n\n"
        "Rules:\n"
        "- Preserve the original meaning and content.\n"
        "- Only fix errors; do not rewrite or rephrase.\n"
        "- Maintain paragraph structure.\n"
        "- Return ONLY the cleaned text, no explanations.\n\n"
        f"Text:\n{text}\n\nCleaned:"
    )
    return prompt


def read_template() -> Optional[str]:
    p = Path.cwd() / "sample_prompt.md"
    return p.read_text(encoding="utf-8") if p.exists() else None


def render_template(tpl: str, vars: Dict[str, str]) -> str:
    out = tpl
    for k, v in vars.items():
        out = out.replace("{{" + k + "}}", v)
    return out


def build_stage2_prompt(text: str, sc: SpeakerConfig, include_examples: bool = True) -> str:
    tpl = read_template()
    mapping = (
        "; ".join(f"{c.name} = Speaker {c.speakerNumber}" for c in (sc.characterMapping or []))
        if sc.characterMapping else "none"
    )
    label_format = sc.labelFormat or "speaker"
    speaker_label_example = "[1]:" if label_format == "bracket" else "Speaker 1:"
    speaker_label_instructions = (
        "Identify all unique speaking characters. Assign them labels dynamically using bracket format: the first character to speak becomes [1]:, the second becomes [2]:, and so on."
        if label_format == "bracket"
        else "Identify all unique speaking characters. Assign them labels dynamically: the first character to speak becomes Speaker 1:, the second becomes Speaker 2:, and so on."
    )
    narrator_rule = (
        "All non‑quoted narrative, descriptive, or action text MUST be labeled using the Narrator: tag — never as any Speaker X:. Do not attribute narration content to speakers."
        if sc.includeNarrator
        else "Omit non‑quoted narrative, descriptive, or action text. Output only spoken dialogue labeled with the chosen speaker format."
    )
    narrator_identity = (
        f"Narrator Identity: The narrator is \"{sc.narratorCharacterName}\". For narration, write from first‑person (\"I\") from that character’s perspective. Do not include the narrator’s name inside narration."
        if sc.narratorCharacterName else ""
    )

    if sc.mode == "format":
        attribution_rule = (
            "Do not transform attribution tags. Preserve the original text and punctuation. Only add speaker labels and remove surrounding quotation marks for dialogue."
        )
    else:
        attr = sc.narratorAttribution or "contextual"
        attribution_rule = {
            "remove": "Remove simple attribution tags (e.g., he said, Alice asked) entirely. The Speaker label makes them redundant.",
            "verbatim": "Move attribution tags (e.g., he said, Alice asked) into a separate Narrator: line immediately after the spoken line (preserve punctuation).",
            "contextual": "Transform attribution and action: if attribution is paired with action, convert it into a concise Narrator: line (omit the attribution verb); if attribution is simple and provides no new action, omit it. Preserve the first‑person narrator exception.",
        }.get(attr, "")

    examples = ""
    if include_examples and sc.mode == "intelligent":
        names = [c.name for c in (sc.characterMapping or []) if c.name]
        n1 = names[0] if len(names) > 0 else "Alice"
        n2 = names[1] if len(names) > 1 else "Bob"
        ex_s1 = speaker_label_example
        ex_s2 = "[2]:" if label_format == "bracket" else "Speaker 2:"
        ex = [
            "Examples:",
            f'Input: "Are you coming to the party?" {n1} asked.',
            f"Output: {ex_s1} Are you coming to the party?",
            f'Input: "It\'s a beautiful day," {n2} said, looking up at the sky.',
            f"Output: {ex_s2} It's a beautiful day.",
            f"Narrator: {n2} looked up at the sky.",
        ]
        examples = "\n".join(ex)

    if tpl:
        return render_template(
            tpl,
            {
                "mapping": mapping,
                "preprocessing": "",  # ensure Stage 2 has no cleaning section
                "text": text,
                "speaker_label_example": speaker_label_example,
                "speaker_label_instructions": speaker_label_instructions,
                "narrator_rule": narrator_rule,
                "narrator_identity": narrator_identity,
                "attribution_rule": attribution_rule,
                "examples": examples,
            },
        )

    # Fallback minimal Stage 2 prompt
    out = [
        "You are a dialogue structuring assistant for multi-speaker TTS.",
        speaker_label_instructions,
        narrator_rule,
        "Remove quotation marks from dialogue.",
        "Merge dialogue from same speaker when separated only by attribution.",
        attribution_rule,
        narrator_identity,
        "\nText:",
        text,
        "\nFormatted:",
    ]
    return "\n".join([s for s in out if s])


# -------- Inference Clients (optional) --------


def http_post_json(url: str, payload: Dict, headers: Optional[Dict[str, str]] = None, timeout: int = 120) -> Tuple[int, Dict, str]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})})
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="ignore")
        try:
            return resp.status, json.loads(body), body
        except Exception:
            return resp.status, {}, body


def http_get_json(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 120) -> Tuple[int, Dict, str]:
    req = request.Request(url, headers=headers or {})
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="ignore")
        try:
            return resp.status, json.loads(body), body
        except Exception:
            return resp.status, {}, body


def run_ollama(prompt: str, model: str = "qwen3:8b") -> str:
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    status, js, raw = http_post_json(
        f"{base}/api/generate",
        {"model": model, "prompt": prompt, "stream": False},
    )
    if status != 200:
        raise RuntimeError(f"ollama error {status}: {raw[:200]}")
    return js.get("response") or js.get("final_response") or js.get("message", {}).get("content", "")


def run_hf(prompt: str, model: str, token: str) -> str:
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": prompt, "parameters": {"return_full_text": False, "max_new_tokens": 800, "temperature": 0.3}}
    status, js, raw = http_post_json(url, payload, headers=headers)
    if status != 200:
        raise RuntimeError(f"hf error {status}: {raw[:200]}")
    # output can be a list of dicts with generated_text
    if isinstance(js, list) and js and isinstance(js[0], dict):
        return js[0].get("generated_text", "")
    if isinstance(js, dict):
        return js.get("generated_text", "")
    return ""


# -------- Analysis & Report --------


def analyze_output(text: str, label_format: str, include_narrator: bool, narrator_mode: str) -> List[str]:
    lines = [l.strip() for l in re.split(r"\r?\n", text) if l.strip()]
    issues: List[str] = []
    label_ok = re.compile(r"^(Speaker\s+\d+:|\[\d+\]:|Narrator:)\s").match
    bad = [l for l in lines if not label_ok(l)]
    if bad:
        issues.append(f"Unlabeled lines: {bad[:3]}")
    narrator_lines = [l for l in lines if l.lower().startswith("narrator:")]
    if include_narrator and not narrator_lines:
        issues.append("Expected Narrator lines, found none.")
    if not include_narrator and narrator_lines:
        issues.append("Narrator lines present though includeNarrator=false.")
    if any('"' in l for l in lines):
        issues.append("Quotes still present in output.")
    if narrator_mode == "remove":
        said = [l for l in lines if not l.lower().startswith("narrator:") and re.search(r"\b(said|asked|replied|whispered)\b", l, re.I)]
        if said:
            issues.append("Attribution verbs in speaker lines under remove mode.")
    if narrator_mode != "remove" and not narrator_lines:
        issues.append("No Narrator lines under non-remove mode.")
    return issues


def ensure_logs_dir() -> Path:
    p = Path.cwd() / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def main() -> None:
    # Load .env / env.txt into process env so OLLAMA_BASE_URL and tokens are seen
    def load_env_file(p: Path) -> None:
        if not p.exists():
            return
        try:
            for raw in p.read_text(encoding="utf-8").splitlines():
                line = raw.strip().rstrip("\r")
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export ") :]
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                if k and (k not in os.environ or not os.environ[k]):
                    os.environ[k] = v
        except Exception:
            pass

    cwd = Path.cwd()
    load_env_file(cwd / ".env")
    load_env_file(cwd / "env.txt")

    sample = generate_sample_dialogue()

    # Build prompts
    clean_prompt = build_cleaning_prompt(sample, CleaningOptions())
    sc = SpeakerConfig(
        mode="intelligent",
        speakerCount=3,
        labelFormat="speaker",
        includeNarrator=True,
        narratorAttribution="contextual",
        characterMapping=[Character("Alice", 1), Character("Bob", 2), Character("Evelyn", 3)],
        narratorCharacterName=None,
    )
    stage2_prompt = build_stage2_prompt(sample, sc, include_examples=True)

    logs = ensure_logs_dir()
    # Always write prompts to logs for inspection
    try:
        (logs / "py-stage1-prompt.txt").write_text(clean_prompt, encoding="utf-8")
        (logs / "py-stage2-prompt.txt").write_text(stage2_prompt, encoding="utf-8")
    except Exception:
        pass

    print("=== Stage 1 (Cleaning) Prompt ===\n")
    print(clean_prompt)
    print("\n=== Stage 2 (Speaker Formatting) Prompt ===\n")
    print(stage2_prompt)
    if "preprocessing" in stage2_prompt.lower():
        print("\n[WARN] Stage 2 prompt still contains preprocessing info.")
    else:
        print("\n[OK] Stage 2 prompt contains no preprocessing section.")

    if os.environ.get("RUN_TESTS", "0").strip() != "1":
        return

    # Optional inference + report
    model_source = "ollama" if os.environ.get("OLLAMA_BASE_URL") else "api"
    model = os.environ.get("LLM_TEST_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    ollama_model = os.environ.get("LLM_TEST_OLLAMA", "qwen3:8b")
    token = os.environ.get("HUGGINGFACE_API_TOKEN") or os.environ.get("HF_TOKEN")

    # Quick Ollama availability check with GET /api/tags
    if model_source == "ollama":
        try:
            base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            status, js, raw = http_get_json(base + "/api/tags", timeout=5)
            if status != 200:
                print(f"[WARN] Ollama at {base} responded with status {status}; falling back to API if available.")
                if token:
                    model_source = "api"
            else:
                # Write available models once for debugging
                try:
                    models = js.get("models") if isinstance(js, dict) else None
                    if models:
                        ensure_logs_dir().joinpath("py-ollama-tags.json").write_text(json.dumps(js, indent=2), encoding="utf-8")
                except Exception:
                    pass
        except Exception as e:
            print(f"[WARN] Could not reach Ollama at {os.environ.get('OLLAMA_BASE_URL')}: {e}")
            if token:
                model_source = "api"

    tests = [
        ("speaker+contextual+single", dict(labelFormat="speaker", includeNarrator=True, narratorAttribution="contextual", singlePass=True)),
        ("speaker+remove+two", dict(labelFormat="speaker", includeNarrator=True, narratorAttribution="remove", singlePass=False)),
        ("speaker+verbatim+two", dict(labelFormat="speaker", includeNarrator=True, narratorAttribution="verbatim", singlePass=False)),
        ("bracket+contextual+two", dict(labelFormat="bracket", includeNarrator=True, narratorAttribution="contextual", singlePass=False)),
        ("speaker+noNarr+two", dict(labelFormat="speaker", includeNarrator=False, narratorAttribution="remove", singlePass=False)),
    ]

    logs = ensure_logs_dir()
    report_lines: List[str] = []

    for name, cfg in tests:
        cfg_sc = SpeakerConfig(
            mode="intelligent",
            speakerCount=3,
            labelFormat=cfg["labelFormat"],
            includeNarrator=cfg["includeNarrator"],
            narratorAttribution=cfg["narratorAttribution"],
            characterMapping=[Character("Alice", 1), Character("Bob", 2), Character("Evelyn", 3)],
        )
        prompt = build_stage2_prompt(sample, cfg_sc, include_examples=True)
        started = time.time()
        try:
            if model_source == "ollama":
                out = run_ollama(prompt, ollama_model)
            else:
                if not token:
                    raise RuntimeError("HUGGINGFACE_API_TOKEN not set")
                    
                out = run_hf(prompt, model, token)
        except Exception as e:
            out = f"[error] {e}"
        elapsed = int((time.time() - started) * 1000)

        if not out.startswith("[error]"):
            issues = analyze_output(out, cfg_sc.labelFormat, cfg_sc.includeNarrator, cfg_sc.narratorAttribution)
        else:
            issues = [out]

        report_lines.append(
            f"\n=== {name} ({cfg_sc.labelFormat}, narr={cfg_sc.includeNarrator}, attr={cfg_sc.narratorAttribution}) in {elapsed}ms ==="
        )
        report_lines.append(f"Issues: {' | '.join(issues) if issues else 'None'}")
        sample_out = "\n".join(out.splitlines()[:12])
        report_lines.append("Output (first lines):")
        report_lines.append(sample_out)

        try:
            (logs / f"py-llm-output-{name}.txt").write_text(out, encoding="utf-8")
        except Exception:
            pass

    (logs / "llm-py-report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nReport written to {logs / 'llm-py-report.txt'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
