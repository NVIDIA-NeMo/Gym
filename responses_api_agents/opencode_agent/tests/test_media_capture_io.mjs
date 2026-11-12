// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { chmod, mkdir, mkdtemp, rm, stat, symlink } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import { privateDirectory, sha256 } from "../media_capture_io.mjs";

test("shared capture hashing retains string and byte semantics", () => {
  const expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"; // pragma: allowlist secret
  assert.equal(sha256("abc"), expected);
  assert.equal(sha256(Buffer.from("abc")), expected);
});

for (const kind of ["input", "output"]) {
  test(`${kind} directory policy preserves private ownership and error text`, async () => {
    const temporary = await mkdtemp(path.join(os.tmpdir(), "video-capture-io-"));
    try {
      const directory = path.join(temporary, "capture");
      await privateDirectory(directory, kind);
      assert.equal((await stat(directory)).mode & 0o777, 0o700);
      await chmod(directory, 0o755);
      await assert.rejects(privateDirectory(directory, kind), {
        message: `${kind} capture directory must be owned, non-symlink and mode 0700`,
      });
      const target = path.join(temporary, "target");
      await mkdir(target, { mode: 0o700 });
      const linked = path.join(temporary, "linked");
      await symlink(target, linked);
      await assert.rejects(privateDirectory(linked, kind), {
        message: `${kind} capture directory must be owned, non-symlink and mode 0700`,
      });
    } finally {
      await rm(temporary, { recursive: true, force: true });
    }
  });
  test(`${kind} relative paths fail before creating directories`, async () => {
    await assert.rejects(privateDirectory("relative-capture", kind), {
      message: `${kind} capture requires an absolute private directory`,
    });
  });
}
