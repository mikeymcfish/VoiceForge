import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm, symlink, truncate, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import {
  MAX_DEFAULT_VOICE_AUDIO_BYTES,
  MAX_DEFAULT_VOICE_TRANSCRIPT_BYTES,
  VoiceLibraryService,
} from "../server/voice-library-service";

const sandbox = await mkdtemp(path.join(os.tmpdir(), "voiceforge-voice-library-"));
const root = path.join(sandbox, "default_voices");
await mkdir(root);

try {
  await writeFile(path.join(root, "Alice.WAV"), Buffer.from("RIFFalice"));
  await writeFile(path.join(root, "alice.txt"), "Hello from Alice.\n", "utf8");
  await writeFile(path.join(root, "Narrator.mp3"), Buffer.from("ID3narrator"));
  await writeFile(path.join(root, "Dupe.wav"), Buffer.from("RIFFdupe-one"));
  await writeFile(path.join(root, "dupe.MP3"), Buffer.from("ID3dupe-two"));
  await writeFile(path.join(root, "Ambiguous.flac"), Buffer.from("fLaCambiguous"));
  await writeFile(path.join(root, "ambiguous.txt"), "first transcript", "utf8");
  await writeFile(path.join(root, "ＡMBIGUOUS.TXT"), "second transcript", "utf8");
  await writeFile(path.join(root, "LongTranscript.ogg"), Buffer.from("OggSvoice"));
  await writeFile(
    path.join(root, "longtranscript.txt"),
    Buffer.alloc(MAX_DEFAULT_VOICE_TRANSCRIPT_BYTES + 1, 0x61)
  );
  await writeFile(path.join(root, "ignored.exe"), Buffer.from("not audio"));
  await mkdir(path.join(root, "nested"));
  await writeFile(path.join(root, "nested", "Nested.wav"), Buffer.from("RIFFnested"));

  const oversizedAudio = path.join(root, "Oversized.webm");
  await writeFile(oversizedAudio, Buffer.from([0]));
  await truncate(oversizedAudio, MAX_DEFAULT_VOICE_AUDIO_BYTES + 1);

  const externalAudio = path.join(sandbox, "external.wav");
  await writeFile(externalAudio, Buffer.from("RIFFexternal"));
  try {
    await symlink(externalAudio, path.join(root, "Linked.wav"), "file");
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code !== "EPERM" && code !== "EACCES" && code !== "ENOSYS") throw error;
  }

  const service = new VoiceLibraryService(root);
  const catalog = await service.listVoices();

  assert.deepEqual(
    catalog.map((voice) => voice.name),
    ["Alice", "Ambiguous", "LongTranscript", "Narrator"]
  );
  assert.ok(!catalog.some((voice) => voice.name.toLocaleLowerCase("en-US") === "dupe"));
  assert.ok(!catalog.some((voice) => voice.name === "Oversized"));
  assert.ok(!catalog.some((voice) => voice.name === "Nested"));
  assert.ok(!catalog.some((voice) => voice.name === "Linked"));

  const alice = catalog.find((voice) => voice.name === "Alice");
  assert.ok(alice);
  assert.match(alice.id, /^voice_[A-Za-z0-9_-]{43}$/);
  assert.equal(alice.format, "wav");
  assert.equal(alice.mimeType, "audio/wav");
  assert.equal(alice.hasTranscript, true);
  assert.equal(alice.transcript, "Hello from Alice.\n");

  const ambiguous = catalog.find((voice) => voice.name === "Ambiguous");
  assert.ok(ambiguous);
  assert.equal(ambiguous.hasTranscript, false);
  assert.equal(ambiguous.transcript, null);

  const longTranscript = catalog.find((voice) => voice.name === "LongTranscript");
  assert.ok(longTranscript);
  assert.equal(longTranscript.hasTranscript, false);
  assert.equal(longTranscript.transcript, null);

  for (const metadata of catalog) {
    assert.ok(!Object.keys(metadata).some((key) => key.toLocaleLowerCase("en-US").includes("path")));
  }

  const secondCatalog = await service.catalog();
  assert.equal(
    secondCatalog.find((voice) => voice.name === "Alice")?.id,
    alice.id,
    "catalog IDs must remain stable"
  );

  const resolved = await service.resolveVoice(alice.id);
  assert.ok(resolved);
  assert.equal(path.dirname(resolved.audioPath), path.resolve(root));
  assert.equal(resolved.metadata.id, alice.id);

  const read = await service.readVoice(alice.id);
  assert.ok(read);
  assert.equal(read.audio.toString("utf8"), "RIFFalice");
  assert.equal(read.metadata.transcript, "Hello from Alice.\n");

  assert.equal(await service.resolveVoice("../../external.wav"), undefined);
  assert.equal(await service.readVoice("voice_not-a-valid-id"), undefined);
  assert.equal(await service.resolveVoice(""), undefined);

  // An ID from a prior catalog must fail closed after its file is replaced by
  // a non-file entry rather than resolving a stale path.
  await rm(path.join(root, "Alice.WAV"));
  await mkdir(path.join(root, "Alice.WAV"));
  assert.equal(await service.resolveVoice(alice.id), undefined);
  assert.equal(await service.readVoice(alice.id), undefined);

  // A symlink root is never treated as a voice library.
  const rootLink = path.join(sandbox, "default_voices_link");
  try {
    await symlink(root, rootLink, "junction");
    assert.deepEqual(await new VoiceLibraryService(rootLink).listVoices(), []);
  } catch (error) {
    const code = (error as NodeJS.ErrnoException).code;
    if (code !== "EPERM" && code !== "EACCES" && code !== "ENOSYS") throw error;
  }

  console.log("Voice library self-test passed.");
} finally {
  await rm(sandbox, { recursive: true, force: true });
}
