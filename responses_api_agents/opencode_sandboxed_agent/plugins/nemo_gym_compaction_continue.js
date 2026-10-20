// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Stops opencode inviting the model to give up right after its memory is compressed.
//
// WHY. After every compaction stock appends: "Continue if you have next steps, or stop and ask
// for clarification if you are unsure how to proceed." In `opencode run` there is nobody to ask,
// and the model has just lost its working context — which is exactly the moment it is least sure
// how to proceed. The fork replaced it with "Continue with the next steps." On the 2026-10-06
// BrowseComp-400 run 61 of 400 rows compacted at least once, some up to 22 times.
//
// The dedicated experimental.compaction.autocontinue hook only carries a boolean: it can suppress
// the message but not reword it. So we rewrite the text in the message list instead.
//
// TWO TRAPS THIS CODE WORKS AROUND.
//
// 1. Reassigning output.messages is silently dropped — prompt.ts reads its own captured binding,
//    not our replacement. Only in-place mutation of the existing objects is observed.
// 2. Rewriting the message list only changes what goes on the wire. The transcript Gym grades and
//    exports is read back out of opencode's SQLite database, so a wire-only edit leaves the two
//    disagreeing — the model would be trained or judged on text it never saw. We therefore also
//    patch the stored part through the in-process client, keeping DB, export and wire identical.

const PLUGIN = "[nemo-gym-plugin]";
const DEFAULT_TEXT = "Continue with the next steps.";

export default async function nemoGymCompactionContinue({ client }) {
  const replacement = process.env.NEMO_GYM_COMPACTION_CONTINUE_TEXT || DEFAULT_TEXT;
  if (process.env.NEMO_GYM_COMPACTION_CONTINUE === "0") return {};

  // Parts already rewritten, so a part that survives several turns in the window is not
  // re-patched on every request.
  const done = new Set();

  return {
    "experimental.chat.messages.transform": async (_input, output) => {
      for (const message of output.messages ?? []) {
        for (const part of message.parts ?? []) {
          // Match on the marker opencode stamps, not on the literal sentence: the wording is
          // upstream's and will drift between releases, the marker is structural.
          if (part?.metadata?.compaction_continue !== true) continue;
          if (part.text === replacement) continue;

          part.text = replacement; // in place — see trap 1

          const key = message.info?.id + "/" + part.id;
          if (done.has(key)) continue;
          done.add(key);

          // see trap 2. _client is an undocumented internal of the generated SDK (there is no
          // published updatePart), so this is pinned by the plugin smoke test and must be
          // re-checked on every opencode bump.
          try {
            await client.session._client.patch({
              url:
                "/session/" +
                message.info.sessionID +
                "/message/" +
                message.info.id +
                "/part/" +
                part.id,
              body: { ...part, text: replacement },
            });
          } catch (e) {
            // Non-fatal: the wire is already correct, only the stored copy still differs.
            process.stderr.write(PLUGIN + " compaction_continue_patch_failed err=" + e + "\n");
          }
        }
      }
    },
  };
}
