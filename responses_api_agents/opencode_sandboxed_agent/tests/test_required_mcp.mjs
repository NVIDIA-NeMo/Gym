// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
import assert from "node:assert/strict";
import { existsSync, readFileSync, rmSync } from "node:fs";
import test from "node:test";
import requiredMcp from "../required-mcp.js";

const marker = "/tmp/nemo-gym-mcp-setup-error";
for (const state of ["connected", "failed", "disabled", "needs_auth", "missing", "transport-error"]) {
  test(`required MCP status: ${state}`, async (t) => {
    process.env.NEMO_GYM_REQUIRED_MCP_SERVERS = '["search"]';
    rmSync(marker, { force: true });
    t.after(() => { delete process.env.NEMO_GYM_REQUIRED_MCP_SERVERS; rmSync(marker, { force: true }); });
    let calls = 0;
    const hook = await requiredMcp({client: {mcp: {status: async () => {
      calls++;
      if (state === "transport-error") throw new Error("private endpoint details");
      return {data: state === "missing" ? {} : {search: {status: state}}};
    }}}});
    if (state === "connected") {
      await hook["chat.params"]();
      await hook["chat.params"]();
      assert.equal(calls, 1);
      assert.equal(existsSync(marker), false);
    } else {
      await assert.rejects(hook["chat.params"](), /Required Gym MCP tools/);
      assert.equal(existsSync(marker), true);
      assert.equal(readFileSync(marker, "utf8").includes("private"), false);
    }
  });
}
