import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import fsPromises from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import {
  CONSERVATIVE_CLEANUP_FILTER,
  buildAudioSrArgs,
  buildCleanupFfmpegArgs,
  buildFfmetadata,
  buildFfmetadataDocument,
  buildMp3FfmpegArgs,
  enhanceReferenceAudio,
  escapeFfmetadataValue,
  finalizeSpeechAudio,
  getAudioProcessingCapabilities,
  parseWorkerChapterManifest,
  resolveAudioSrBinary,
  resolveFfmpegBinary,
  validateWorkerChapterManifest,
  type WorkerChapterManifest,
} from "../server/audio-postprocess-service";

const manifest: WorkerChapterManifest = {
  version: 1,
  sample_rate: 48_000,
  total_samples: 96_000,
  chapters: [
    {
      title: "Prologue; #1=\\",
      start_sample: 0,
      start_ms: 0,
      start_seconds: 0,
    },
    {
      title: null,
      start_sample: 24_001,
      start_ms: (24_001 * 1_000) / 48_000,
      start_seconds: 24_001 / 48_000,
    },
  ],
};

async function runBinary(executable: string, args: string[]): Promise<Buffer> {
  return new Promise<Buffer>((resolve, reject) => {
    const child = spawn(executable, args, {
      windowsHide: true,
      shell: false,
      stdio: ["ignore", "pipe", "pipe"],
    });
    const stdout: Buffer[] = [];
    const stderr: Buffer[] = [];
    child.stdout.on("data", (chunk: Buffer) => stdout.push(Buffer.from(chunk)));
    child.stderr.on("data", (chunk: Buffer) => stderr.push(Buffer.from(chunk)));
    child.once("error", reject);
    child.once("close", (code, signal) => {
      if (code === 0) {
        resolve(Buffer.concat(stdout));
        return;
      }
      const detail = Buffer.concat(stderr).toString("utf8").trim();
      reject(
        new Error(
          `${path.basename(executable)} exited with ${
            code === null ? `signal ${signal || "unknown"}` : `code ${code}`
          }.${detail ? `\n${detail}` : ""}`
        )
      );
    });
  });
}

async function adjacentFfprobe(ffmpeg: string): Promise<string | undefined> {
  const candidate = path.join(
    path.dirname(ffmpeg),
    process.platform === "win32" ? "ffprobe.exe" : "ffprobe"
  );
  const stats = await fsPromises.stat(candidate).catch(() => undefined);
  return stats?.isFile() ? candidate : undefined;
}

function testManifestValidation(): void {
  assert.deepEqual(
    validateWorkerChapterManifest(JSON.parse(JSON.stringify(manifest))),
    manifest
  );
  assert.deepEqual(parseWorkerChapterManifest(JSON.stringify(manifest)), manifest);

  assert.throws(
    () => validateWorkerChapterManifest({ ...manifest, extra: true }),
    /unknown fields/i
  );
  assert.throws(
    () => validateWorkerChapterManifest({ ...manifest, version: 2 }),
    /version must be exactly 1/i
  );
  assert.throws(
    () =>
      validateWorkerChapterManifest({
        ...manifest,
        chapters: [{ ...manifest.chapters[1], start_ms: manifest.chapters[1].start_ms + 1e-8 }],
      }),
    /start_ms does not match/i
  );
  assert.doesNotThrow(() =>
    validateWorkerChapterManifest({
      ...manifest,
      chapters: [
        {
          ...manifest.chapters[1],
          start_ms: manifest.chapters[1].start_ms + 5e-10,
        },
      ],
    })
  );
  assert.throws(
    () =>
      validateWorkerChapterManifest({
        ...manifest,
        chapters: [{ ...manifest.chapters[0], start_seconds: 0.01 }],
      }),
    /start_seconds does not match/i
  );
  assert.throws(
    () =>
      validateWorkerChapterManifest({
        ...manifest,
        chapters: [
          manifest.chapters[0],
          {
            title: "Duplicate",
            start_sample: 0,
            start_ms: 0,
            start_seconds: 0,
          },
        ],
      }),
    /strictly increasing/i
  );
  assert.throws(
    () =>
      validateWorkerChapterManifest({
        ...manifest,
        chapters: [
          {
            title: "Outside",
            start_sample: manifest.total_samples,
            start_ms: manifest.total_samples,
            start_seconds: manifest.total_samples / manifest.sample_rate,
          },
        ],
      }),
    /less than total_samples/i
  );
  assert.throws(
    () =>
      validateWorkerChapterManifest({
        ...manifest,
        chapters: [{ ...manifest.chapters[0], title: "x".repeat(4_097) }],
      }),
    /exceeds 4096/i
  );
  assert.throws(
    () =>
      validateWorkerChapterManifest({
        ...manifest,
        chapters: [{ ...manifest.chapters[0], title: " \t " }],
      }),
    /null or a non-empty string/i
  );
  assert.throws(() => parseWorkerChapterManifest("{"), /not valid JSON/i);
}

function testMetadata(): void {
  assert.equal(
    escapeFfmetadataValue("one\\two;\n#three=four\r"),
    "one\\\\two\\; \\#three\\=four "
  );

  const expected = [
    ";FFMETADATA1",
    "[CHAPTER]",
    "TIMEBASE=1/1000",
    "START=0",
    "END=500",
    "title=Prologue\\; \\#1\\=\\\\",
    "[CHAPTER]",
    "TIMEBASE=1/1000",
    "START=500",
    "END=2000",
    "title=Chapter 2",
    "",
  ].join("\n");
  assert.deepEqual(buildFfmetadataDocument(manifest), {
    text: expected,
    chapterCount: 2,
  });
  assert.equal(buildFfmetadata(manifest), expected);
}

function testCommands(): void {
  assert.deepEqual(
    buildMp3FfmpegArgs({
      inputWavPath: "input.wav",
      outputMp3Path: "output.mp3",
      ffmetadataPath: "chapters.ffmeta",
      mp3Quality: 9,
    }),
    [
      "-y",
      "-hide_banner",
      "-loglevel",
      "error",
      "-i",
      "input.wav",
      "-f",
      "ffmetadata",
      "-i",
      "chapters.ffmeta",
      "-map_metadata",
      "1",
      "-map_chapters",
      "1",
      "-id3v2_version",
      "3",
      "-map",
      "0:a",
      "-vn",
      "-codec:a",
      "libmp3lame",
      "-b:a",
      "192k",
      "-write_xing",
      "0",
      "output.mp3",
    ]
  );
  assert.deepEqual(
    buildMp3FfmpegArgs({
      inputWavPath: "input.wav",
      outputMp3Path: "output.mp3",
      mp3Quality: 3,
    }),
    [
      "-y",
      "-hide_banner",
      "-loglevel",
      "error",
      "-i",
      "input.wav",
      "-map",
      "0:a",
      "-vn",
      "-codec:a",
      "libmp3lame",
      "-q:a",
      "3",
      "output.mp3",
    ]
  );
  assert.throws(
    () =>
      buildMp3FfmpegArgs({
        inputWavPath: "input.wav",
        outputMp3Path: "output.mp3",
        mp3Quality: 10,
      }),
    /integer from 0 .* through 9/i
  );

  const cleanupArgs = buildCleanupFfmpegArgs({
    inputPath: "reference.wav",
    outputWavPath: "cleaned.wav",
  });
  assert.equal(cleanupArgs.at(-1), "cleaned.wav");
  assert.equal(cleanupArgs[cleanupArgs.indexOf("-ac") + 1], "1");
  assert.equal(cleanupArgs[cleanupArgs.indexOf("-ar") + 1], "24000");
  assert.equal(cleanupArgs[cleanupArgs.indexOf("-sample_fmt") + 1], "s16");
  assert.equal(cleanupArgs[cleanupArgs.indexOf("-af") + 1], CONSERVATIVE_CLEANUP_FILTER);
  assert.match(
    CONSERVATIVE_CLEANUP_FILTER,
    /areverse,silenceremove=start_periods=1:start_threshold=-42dB,areverse/u
  );
  assert.doesNotMatch(CONSERVATIVE_CLEANUP_FILTER, /stop_periods=/u);

  assert.deepEqual(
    buildAudioSrArgs({
      inputPath: "reference.wav",
      outputDirectory: "audiosr-output",
    }),
    [
      "-i",
      "reference.wav",
      "-s",
      "audiosr-output",
      "--model_name",
      "speech",
      "-d",
      "auto",
      "--ddim_steps",
      "50",
      "--guidance_scale",
      "3.5",
      "--seed",
      "42",
      "--suffix",
      "_voiceforge_audiosr",
    ]
  );
  assert.equal(
    buildAudioSrArgs({
      inputPath: "reference.wav",
      outputDirectory: "audiosr-output",
      audioSr: { device: "CUDA:2" as "cuda:2" },
    })[7],
    "cuda:2"
  );
  assert.deepEqual(
    buildAudioSrArgs({
      inputPath: "reference.wav",
      outputDirectory: "audiosr-output",
      audioSr: {
        model: "basic",
        device: "cuda:2",
        steps: 125,
        guidance: 7.25,
        seed: -7,
      },
    }).slice(4, 14),
    [
      "--model_name",
      "basic",
      "-d",
      "cuda:2",
      "--ddim_steps",
      "125",
      "--guidance_scale",
      "7.25",
      "--seed",
      "-7",
    ]
  );
  assert.throws(
    () =>
      buildAudioSrArgs({
        inputPath: "reference.wav",
        outputDirectory: "audiosr-output",
        audioSr: { device: "cuda:9999" as "cuda:0" },
      }),
    /device must be/i
  );
  assert.throws(
    () =>
      buildAudioSrArgs({
        inputPath: "reference.wav",
        outputDirectory: "audiosr-output",
        audioSr: { steps: 9 },
      }),
    /steps must be/i
  );
  assert.throws(
    () =>
      buildAudioSrArgs({
        inputPath: "reference.wav",
        outputDirectory: "audiosr-output",
        audioSr: { guidance: 11 },
      }),
    /guidance must be/i
  );
  assert.throws(
    () =>
      buildAudioSrArgs({
        inputPath: "reference.wav",
        outputDirectory: "audiosr-output",
        audioSr: { seed: 2_147_483_648 },
      }),
    /seed must be/i
  );
}

async function testRealFfmpegPipeline(): Promise<void> {
  if (!getAudioProcessingCapabilities().ffmpegAvailable) return;

  const ffmpeg = resolveFfmpegBinary();
  const sandbox = await fsPromises.mkdtemp(
    path.join(os.tmpdir(), "voiceforge-real-ffmpeg-")
  );
  try {
    const inputPath = path.join(sandbox, "reference.wav");
    await runBinary(ffmpeg, [
      "-y",
      "-hide_banner",
      "-loglevel",
      "error",
      "-f",
      "lavfi",
      "-i",
      "anullsrc=r=24000:cl=mono:d=0.25",
      "-f",
      "lavfi",
      "-i",
      "sine=frequency=440:duration=0.4:sample_rate=24000",
      "-f",
      "lavfi",
      "-i",
      "anullsrc=r=24000:cl=mono:d=0.3",
      "-f",
      "lavfi",
      "-i",
      "sine=frequency=660:duration=0.4:sample_rate=24000",
      "-f",
      "lavfi",
      "-i",
      "anullsrc=r=24000:cl=mono:d=0.25",
      "-filter_complex",
      "[0:a][1:a][2:a][3:a][4:a]concat=n=5:v=0:a=1[out]",
      "-map",
      "[out]",
      "-codec:a",
      "pcm_s16le",
      inputPath,
    ]);

    const cleanedPath = await enhanceReferenceAudio({
      inputPath: "reference.wav",
      workingDir: sandbox,
      mode: "cleanup",
    });
    const pcm = await runBinary(ffmpeg, [
      "-hide_banner",
      "-loglevel",
      "error",
      "-i",
      cleanedPath,
      "-map",
      "0:a:0",
      "-ac",
      "1",
      "-ar",
      "24000",
      "-f",
      "s16le",
      "pipe:1",
    ]);
    assert.equal(pcm.length % 2, 0);
    const sampleRate = 24_000;
    const sampleCount = pcm.length / 2;
    const durationSeconds = sampleCount / sampleRate;
    assert.ok(
      durationSeconds > 0.9,
      `Cleanup truncated speech after an interior pause (${durationSeconds.toFixed(3)}s).`
    );
    assert.ok(
      durationSeconds < 1.35,
      `Cleanup did not reduce the silent edges (${durationSeconds.toFixed(3)}s).`
    );

    const windowSamples = 480;
    const rmsWindows: number[] = [];
    for (let start = 0; start < sampleCount; start += windowSamples) {
      const end = Math.min(sampleCount, start + windowSamples);
      let sumSquares = 0;
      for (let sample = start; sample < end; sample += 1) {
        const value = pcm.readInt16LE(sample * 2);
        sumSquares += value * value;
      }
      rmsWindows.push(Math.sqrt(sumSquares / Math.max(1, end - start)));
    }
    const active = rmsWindows.map((rms) => rms >= 250);
    const firstActive = active.indexOf(true);
    const lastActive = active.lastIndexOf(true);
    assert.ok(firstActive >= 0 && lastActive > firstActive, "Cleanup removed both tones.");

    let longestInteriorSilence = 0;
    let currentSilence = 0;
    for (let index = firstActive + 1; index < lastActive; index += 1) {
      if (active[index]) {
        longestInteriorSilence = Math.max(longestInteriorSilence, currentSilence);
        currentSilence = 0;
      } else {
        currentSilence += 1;
      }
    }
    longestInteriorSilence = Math.max(longestInteriorSilence, currentSilence);
    assert.ok(
      longestInteriorSilence * windowSamples >= sampleRate * 0.15,
      "Cleanup removed the interior pause instead of trimming only the edges."
    );
    assert.ok(
      firstActive * windowSamples < sampleRate * 0.12,
      "Cleanup retained too much leading silence."
    );
    assert.ok(
      (active.length - 1 - lastActive) * windowSamples < sampleRate * 0.12,
      "Cleanup retained too much trailing silence."
    );

    const realManifest: WorkerChapterManifest = {
      version: 1,
      sample_rate: sampleRate,
      total_samples: 38_400,
      chapters: [
        {
          title: "First",
          start_sample: 0,
          start_ms: 0,
          start_seconds: 0,
        },
        {
          title: null,
          start_sample: 19_201,
          start_ms: (19_201 * 1_000) / sampleRate,
          start_seconds: 19_201 / sampleRate,
        },
      ],
    };
    await fsPromises.writeFile(
      path.join(sandbox, "chapters.json"),
      JSON.stringify(realManifest),
      "utf8"
    );
    const finalized = await finalizeSpeechAudio({
      inputWavPath: "reference.wav",
      workingDir: sandbox,
      outputFormat: "mp3",
      chapterManifestPath: "chapters.json",
    });
    assert.equal(finalized.chapterCount, 2);

    const ffprobe = await adjacentFfprobe(ffmpeg);
    if (ffprobe) {
      const probeOutput = await runBinary(ffprobe, [
        "-v",
        "error",
        "-show_chapters",
        "-of",
        "json",
        finalized.outputPath,
      ]);
      const probed = JSON.parse(probeOutput.toString("utf8")) as {
        chapters?: Array<{
          start?: number;
          tags?: { title?: string };
        }>;
      };
      assert.equal(probed.chapters?.length, 2);
      assert.equal(probed.chapters?.[0]?.tags?.title, "First");
      assert.equal(probed.chapters?.[1]?.start, 800);
      assert.equal(probed.chapters?.[1]?.tags?.title, "Chapter 2");
    }
  } finally {
    await fsPromises.rm(sandbox, { recursive: true, force: true });
  }
}

async function testManagedApi(): Promise<void> {
  const sandbox = await fsPromises.mkdtemp(path.join(os.tmpdir(), "voiceforge-audio-postprocess-"));
  try {
    const jobDirectory = path.join(sandbox, "job");
    const outsideFile = path.join(sandbox, "outside.wav");
    const inputFile = path.join(jobDirectory, "speech.wav");
    const emptyFile = path.join(jobDirectory, "empty.wav");
    const manifestFile = path.join(jobDirectory, "chapters.json");
    const malformedManifestFile = path.join(jobDirectory, "malformed.json");
    await fsPromises.mkdir(jobDirectory);
    await Promise.all([
      fsPromises.writeFile(inputFile, Buffer.from("not-a-real-wav-but-nonempty")),
      fsPromises.writeFile(emptyFile, Buffer.alloc(0)),
      fsPromises.writeFile(outsideFile, Buffer.from("outside")),
      fsPromises.writeFile(manifestFile, JSON.stringify(manifest)),
      fsPromises.writeFile(malformedManifestFile, JSON.stringify({ ...manifest, version: 2 })),
    ]);

    assert.equal(
      await enhanceReferenceAudio({
        inputPath: "speech.wav",
        workingDir: jobDirectory,
        mode: "none",
      }),
      await fsPromises.realpath(inputFile)
    );

    const finalized = await finalizeSpeechAudio({
      inputWavPath: "speech.wav",
      workingDir: jobDirectory,
      outputFormat: "wav",
      chapterManifestPath: "chapters.json",
    });
    assert.deepEqual(finalized, {
      outputPath: await fsPromises.realpath(inputFile),
      format: "wav",
      mimeType: "audio/wav",
      chapterCount: 2,
    });

    await assert.rejects(
      finalizeSpeechAudio({
        inputWavPath: "empty.wav",
        workingDir: jobDirectory,
        outputFormat: "wav",
      }),
      /WAV output is empty/i
    );
    await assert.rejects(
      enhanceReferenceAudio({
        inputPath: outsideFile,
        workingDir: jobDirectory,
        mode: "none",
      }),
      /must be contained by workingDir/i
    );
    await assert.rejects(
      finalizeSpeechAudio({
        inputWavPath: "speech.wav",
        workingDir: jobDirectory,
        outputFormat: "wav",
        chapterManifestPath: "malformed.json",
      }),
      /version must be exactly 1/i
    );
  } finally {
    await fsPromises.rm(sandbox, { recursive: true, force: true });
  }
}

function testCapabilities(): void {
  const capabilities = getAudioProcessingCapabilities();
  assert.equal(typeof capabilities.ffmpegAvailable, "boolean");
  assert.equal(typeof capabilities.audioSrAvailable, "boolean");

  assert.deepEqual(getAudioProcessingCapabilities({ PATH: "" }), {
    ffmpegAvailable: false,
    audioSrAvailable: false,
  });
  const explicitEnvironment = {
    PATH: "",
    VOICEFORGE_FFMPEG_BIN: process.execPath,
    VOICEFORGE_AUDIOSR_BIN: process.execPath,
  };
  assert.deepEqual(getAudioProcessingCapabilities(explicitEnvironment), {
    ffmpegAvailable: true,
    audioSrAvailable: true,
  });
  assert.equal(resolveFfmpegBinary(explicitEnvironment), path.resolve(process.execPath));
  assert.equal(resolveAudioSrBinary(explicitEnvironment), path.resolve(process.execPath));
}

async function main(): Promise<void> {
  testManifestValidation();
  testMetadata();
  testCommands();
  testCapabilities();
  await testRealFfmpegPipeline();
  await testManagedApi();
  console.log("audio postprocess selftest: ok");
}

await main();
