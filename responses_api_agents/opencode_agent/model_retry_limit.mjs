// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Holds OpenCode 1.17.11 and Kilo Code 7.4.15 to Gym's model retry rule: resend a failed model request only after
// a 408, 425, 429, 5xx or a failed connection, at most twice in a row, waiting at most 10 s before each resend.
//
// Both CLIs resend a failed model request with no time limit (OpenCode with no attempt limit either), also after a
// 409, and wait exactly as long as Retry-After says, so `Retry-After: 0` makes a hot loop. Neither has a setting
// for this, so the plugin wraps the fetch of every provider in the config:
// - a 408, 425, 429 or 5xx reaches the CLI with retry-after-ms set to the bounded wait;
// - a 409, which the CLI would resend, and the third failure in a row end the turn with an error that names the
//   status and keeps the server's error message. It is a plain Error: the CLIs resend every 5xx API error. Its text
//   and its missing cause keep both CLIs from reading it as a rate limit or a lost connection, which they would also
//   resend, so a server message with such words is left out.
// - any other answer ends a run of failures.

const MAX_RETRIES = 2;
const MAX_RETRY_DELAY_MS = 10_000;
const LIMIT = "NeMo Gym model retry limit";
const WRAPPED = Symbol.for("nemo-gym.model-retry-limit");

// Words that make OpenCode or Kilo resend an error whatever its type.
const RESENT_WORDS = /rate limit|rate increased|too many requests|fetch|socket|connect|network|load failed|time ?d? ?out/i;

const mayClear = (status) => status === 408 || status === 425 || status === 429 || status >= 500;
const bounded = (ms) => Math.min(Math.max(ms, 0), MAX_RETRY_DELAY_MS);

function retryDelayMs(headers, failures) {
  const ms = Number.parseFloat(headers.get("retry-after-ms") ?? "");
  if (Number.isFinite(ms)) return bounded(ms);
  const seconds = Number.parseFloat(headers.get("retry-after") ?? "");
  if (Number.isFinite(seconds)) return bounded(seconds * 1000);
  return 1000 * 2 ** (failures - 1);
}

async function serverMessage(response) {
  try {
    const text = (await response.text()).slice(0, 4000);
    let message = text;
    try {
      const body = JSON.parse(text);
      message = body?.error?.message ?? body?.message;
    } catch {}
    if (typeof message !== "string" || !message.trim() || RESENT_WORDS.test(message)) return "";
    return `: ${message.trim().slice(0, 1000)}`;
  } catch {
    return "";
  }
}

function limitRetries(fetchFn) {
  let failures = 0;
  const limited = async (input, init) => {
    let response;
    try {
      response = await fetchFn(input, init);
    } catch (error) {
      if (init?.signal?.aborted) throw error;
      failures += 1;
      if (failures > MAX_RETRIES) {
        const attempts = failures;
        failures = 0;
        throw new Error(`${LIMIT}: no model response after ${attempts} attempts`);
      }
      throw error;
    }
    if (response.ok || !(mayClear(response.status) || response.status === 409)) {
      failures = 0;
      return response;
    }
    if (response.status === 409) {
      failures = 0;
      throw new Error(`${LIMIT}: HTTP 409 is not sent again${await serverMessage(response)}`);
    }
    failures += 1;
    if (failures > MAX_RETRIES) {
      const attempts = failures;
      failures = 0;
      throw new Error(`${LIMIT}: HTTP ${response.status} after ${attempts} attempts${await serverMessage(response)}`);
    }
    const headers = new Headers(response.headers);
    headers.set("retry-after-ms", String(retryDelayMs(response.headers, failures)));
    headers.delete("retry-after");
    return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
  };
  limited[WRAPPED] = true;
  return limited;
}

export const NemoGymModelRetryLimit = async () => ({
  config: async (cfg) => {
    for (const provider of Object.values(cfg.provider ?? {})) {
      const options = (provider.options ??= {});
      if (!options.fetch?.[WRAPPED]) options.fetch = limitRetries(options.fetch ?? ((...args) => fetch(...args)));
    }
  },
});
