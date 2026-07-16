import assert from "node:assert/strict";
import { evaluateLocalRequest } from "../server/loopback-request-policy";

assert.equal(evaluateLocalRequest("127.0.0.1", 5000, "127.0.0.1:5000", undefined), "allowed");
assert.equal(evaluateLocalRequest("127.0.0.1", 5000, "localhost:5000", "http://localhost:5000"), "allowed");
assert.equal(evaluateLocalRequest("::1", 5000, "[::1]:5000", "http://[::1]:5000"), "allowed");
assert.equal(evaluateLocalRequest("127.0.0.1", 5000, "evil.test", undefined), "invalid-host");
assert.equal(evaluateLocalRequest("127.0.0.1", 5000, "127.0.0.1:5000", "https://evil.test"), "invalid-origin");
assert.equal(evaluateLocalRequest("0.0.0.0", 5000, "public.example:5000", "https://public.example"), "allowed");
assert.equal(evaluateLocalRequest("127.0.0.1", 80, "localhost", "http://localhost"), "allowed");

console.log("VoiceForge loopback request policy self-test passed.");
