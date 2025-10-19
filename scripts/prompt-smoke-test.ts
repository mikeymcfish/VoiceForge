/*
 Comprehensive prompt and inference test using Ollama (qwen3:8b) by default.
 Usage examples:
   # Just show prompts for a generated 20-sentence sample
   npx tsx scripts/prompt-smoke-test.ts

   # Run tests with Ollama qwen3:8b (requires Ollama running locally)
   RUN_TESTS=1 OLLAMA_BASE_URL=http://localhost:11434 npx tsx scripts/prompt-smoke-test.ts

   # Force API model instead (requires HF token)
   RUN_TESTS=1 LLM_TEST_MODEL=meta-llama/Llama-3.1-8B-Instruct npx tsx scripts/prompt-smoke-test.ts
*/
import fs from 'fs';
import path from 'path';
import { llmService } from '../server/llm-service';
import type { CleaningOptions, SpeakerConfig } from '../shared/schema';

type TestConfig = {
  name: string;
  singlePass: boolean;
  includeNarrator: boolean;
  labelFormat: 'speaker' | 'bracket';
  narratorAttribution: 'remove' | 'verbatim' | 'contextual';
};

function ensureLogsDir(): string {
  const dir = path.resolve(process.cwd(), 'logs');
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  return dir;
}

function generateSampleDialogue(): string {
  const n = (s: string) => s; // helper
  const parts: string[] = [];
  // 20 sentences spread across narration and dialogues with various patterns
  parts.push(n('It was nearly dusk when I reached the old station.'));
  parts.push('"We are late," Alice said.');
  parts.push('Bob asked, "Did you bring the map?"');
  parts.push('"I did," I replied.'); // first-person narrator exception
  parts.push(n('The wind rattled the signs along the platform.'));
  parts.push('"It’s fine," Alice said, looking up at the schedule.'); // attribution + action
  parts.push('"We still have time," she added.');
  parts.push('"Time for what?" Bob asked.');
  parts.push(n('I hesitated, remembering the warning.'));
  parts.push('"Just go, then," I said.'); // first-person narrator exception inline
  parts.push('"Go where?" he said. "Home?"'); // merge case
  parts.push('"Home, then work," Alice said. "No more delays."'); // merge two quotes
  parts.push(n('A light flickered above the exit.'));
  parts.push('"Did you hear that?" Bob whispered, glancing behind us.'); // attribution + action
  parts.push(n('For a moment, nobody moved.'));
  parts.push('"All right," I said, "let’s move."'); // narrator speaking
  parts.push('"Fine," Alice replied.');
  parts.push('"Fine," Bob replied.');
  parts.push(n('We hurried down the steps into the street.'));
  parts.push(n('Somewhere, a siren wailed.'));
  return parts.join(' ');
}

function buildCleaning(): CleaningOptions {
  return {
    replaceSmartQuotes: true,
    fixOcrErrors: true,
    correctSpelling: false,
    removeUrls: true,
    removeFootnotes: true,
    addPunctuation: true,
    fixHyphenation: false,
  } as any;
}

function buildSpeakerConfig(base: Partial<SpeakerConfig>): SpeakerConfig {
  return {
    mode: 'intelligent',
    speakerCount: 3,
    labelFormat: 'speaker',
    extractCharacters: false,
    sampleSize: 50,
    includeNarrator: true,
    narratorAttribution: 'contextual',
    characterMapping: [
      { name: 'Alice', speakerNumber: 1 },
      { name: 'Bob', speakerNumber: 2 },
      { name: 'Evelyn', speakerNumber: 3 },
    ],
    ...(base as any),
  } as any;
}

function analyzeOutput(text: string, cfg: TestConfig) {
  const lines = text.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
  const issues: string[] = [];

  const narratorLines = lines.filter(l => /^Narrator:/i.test(l));
  const speakerLines = cfg.labelFormat === 'bracket'
    ? lines.filter(l => /^\[\d+\]:/.test(l))
    : lines.filter(l => /^Speaker\s+\d+:/i.test(l));

  // Label format compliance
  const badLabels = lines.filter(l => /^(Speaker\s+\d+:|\[\d+\]:|Narrator:)/i.test(l) === false);
  if (badLabels.length > 0) issues.push(`Unlabeled lines: ${badLabels.slice(0, 3).join(' | ')}`);

  // Narrator presence
  if (cfg.includeNarrator && narratorLines.length === 0) issues.push('Expected Narrator lines, found none.');
  if (!cfg.includeNarrator && narratorLines.length > 0) issues.push('Narrator lines present but includeNarrator=false.');

  // Quotes should be removed
  if (lines.some(l => /"/.test(l))) issues.push('Quotes still present in output.');

  // Attribution handling heuristics
  if (cfg.narratorAttribution === 'remove') {
    // "said", "asked" etc should not appear except potentially as part of dialogue content, which is rare; we warn if found outside Narrator lines
    const saidOutsideNarrator = lines.filter(l => !/^Narrator:/i.test(l) && /\b(said|asked|replied|whispered)\b/i.test(l));
    if (saidOutsideNarrator.length > 0) issues.push('Attribution verbs found in speaker lines under remove mode.');
  }
  if (cfg.narratorAttribution !== 'remove') {
    // There should be some narration for action conversions
    if (narratorLines.length === 0) issues.push('No Narrator lines despite non-remove attribution mode.');
  }

  // Merge heuristic: look for lines with multiple sentences of same speaker
  const multiSentenceSpeakers = speakerLines.filter(l => /\.[^\w]*\w/.test(l));
  if (multiSentenceSpeakers.length === 0) issues.push('No merged speaker lines detected (merge heuristic failed).');

  return { lines, narratorLines, speakerLines, issues };
}

async function runTests(sampleText: string) {
  const modelSource = (process.env.OLLAMA_BASE_URL ? 'ollama' : 'api') as any;
  const modelName = process.env.LLM_TEST_MODEL || 'meta-llama/Llama-3.1-8B-Instruct';
  const ollamaModelName = process.env.LLM_TEST_OLLAMA || 'qwen3:8b';
  const cleaning = buildCleaning();

  const matrix: TestConfig[] = [
    { name: 'speaker+contextual+single', singlePass: true,  includeNarrator: true,  labelFormat: 'speaker', narratorAttribution: 'contextual' },
    { name: 'speaker+remove+two',      singlePass: false, includeNarrator: true,  labelFormat: 'speaker', narratorAttribution: 'remove' },
    { name: 'speaker+verbatim+two',    singlePass: false, includeNarrator: true,  labelFormat: 'speaker', narratorAttribution: 'verbatim' },
    { name: 'bracket+contextual+two',  singlePass: false, includeNarrator: true,  labelFormat: 'bracket', narratorAttribution: 'contextual' },
    { name: 'speaker+noNarr+two',      singlePass: false, includeNarrator: false, labelFormat: 'speaker', narratorAttribution: 'remove' },
  ];

  const logsDir = ensureLogsDir();
  const report: string[] = [];

  for (const cfg of matrix) {
    const sc = buildSpeakerConfig({
      labelFormat: cfg.labelFormat,
      includeNarrator: cfg.includeNarrator,
      narratorAttribution: cfg.narratorAttribution,
    });

    const opts = {
      text: sampleText,
      cleaningOptions: cleaning,
      speakerConfig: sc,
      modelSource,
      modelName,
      ollamaModelName,
      temperature: 0.3,
      llmCleaningDisabled: false,
      singlePass: cfg.singlePass,
      extendedExamples: true,
    } as const;

    let out = '';
    let error: string | undefined;
    const started = Date.now();
    try {
      const res = await llmService.processChunk(opts as any);
      out = res.text;
    } catch (e: any) {
      error = e?.message || String(e);
    }
    const elapsed = Date.now() - started;

    const { issues } = out ? analyzeOutput(out, cfg) : { issues: [`No output. Error: ${error || 'unknown'}`] } as any;

    const header = `\n=== ${cfg.name} (${cfg.labelFormat}, narr=${cfg.includeNarrator}, attr=${cfg.narratorAttribution}, singlePass=${cfg.singlePass}) in ${elapsed}ms ===`;
    report.push(header);
    if (error) report.push(`Error: ${error}`);
    report.push(`Issues: ${issues.length > 0 ? issues.join(' | ') : 'None'}`);
    const sampleOut = out.split(/\r?\n/).slice(0, 12).join('\n');
    report.push('Output (first lines):');
    report.push(sampleOut || '[no output]');

    const outfile = path.join(logsDir, `llm-test-output-${cfg.name}.txt`);
    try { fs.writeFileSync(outfile, out || `[error] ${error}`); } catch {}
  }

  const reportPath = path.join(logsDir, `llm-test-report.txt`);
  fs.writeFileSync(reportPath, report.join('\n'));
  console.log(`\nReport written to ${reportPath}`);
}

async function main() {
  const sampleText = generateSampleDialogue();

  // Show prompts for the default config
  const previews = llmService.getPromptPreviews(
    sampleText,
    buildCleaning(),
    buildSpeakerConfig({}),
    undefined,
    false, // singlePass
    true,  // extendedExamples
    false  // llmCleaningDisabled
  );

  console.log('=== Stage 1 (Cleaning) Prompt ===\n');
  console.log(previews.stage1);
  console.log('\n=== Stage 2 (Speaker Formatting) Prompt ===\n');
  console.log(previews.stage2 || '[no stage 2]');

  if ((previews.stage2 || '').toLowerCase().includes('preprocessing')) {
    console.warn('\n[WARN] Stage 2 prompt still contains preprocessing information.');
  } else {
    console.log('\n[OK] Stage 2 prompt contains no preprocessing section.');
  }

  if (process.env.RUN_TESTS === '1') {
    await runTests(sampleText);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
