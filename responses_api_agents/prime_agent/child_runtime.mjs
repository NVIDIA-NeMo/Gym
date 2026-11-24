// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// SDK integration against Prime Agent v0.7.0 (public SDK API; no vendored implementation).
import { spawnSync } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import { join } from 'node:path';

const checking = process.argv[2] === '--check-runtime';
const args = checking ? {runtime:process.argv[3]} : JSON.parse(readFileSync(process.argv[2], 'utf8'));
if (!args.runtime) throw new Error('Prime child requires a shared runtime directory');
const packageRoot = join(args.runtime, 'lib/node_modules/prime-agent');
const aiRoot = join(packageRoot, 'node_modules/@earendil-works/pi-ai');
const sdk = await import(pathToFileURL(join(packageRoot, 'dist/index.js')));
const ai = await import(pathToFileURL(join(aiRoot, 'dist/index.js')));

function checkRuntime() {
  const sdkVersion = JSON.parse(readFileSync(join(packageRoot, 'package.json'), 'utf8')).version;
  const aiVersion = JSON.parse(readFileSync(join(aiRoot, 'package.json'), 'utf8')).version;
  if (sdkVersion !== '0.7.0' || aiVersion !== '0.7.0') throw new Error('Prime SDK and AI must both be version 0.7.0');
  const required = {
    createAgentSession:sdk.createAgentSession, DefaultResourceLoader:sdk.DefaultResourceLoader,
    ModelRegistry:sdk.ModelRegistry, 'DefaultResourceLoader.getSkills':sdk.DefaultResourceLoader?.prototype?.getSkills,
    'AuthStorage.inMemory':sdk.AuthStorage?.inMemory,
    'SettingsManager.inMemory':sdk.SettingsManager?.inMemory, 'SessionManager.create':sdk.SessionManager?.create,
    loadSkillsFromDir:sdk.loadSkillsFromDir, getPythonSkillRuntimeInfo:sdk.getPythonSkillRuntimeInfo,
    cleanupSessionResources:ai.cleanupSessionResources,
  };
  for (const [name,value] of Object.entries(required)) {
    if (typeof value !== 'function') throw new Error(`Prime SDK capability missing: ${name}`);
  }
  const skillDirectory = join(packageRoot, 'dist/skills/agent-message');
  const discovered = sdk.loadSkillsFromDir({dir:skillDirectory,source:'path'});
  const skill = discovered.skills.find(item => item.name === 'agent-message');
  if (skill?.kind !== 'python') throw new Error('Prime agent-message Python skill is unavailable');
  const runtimeInfo = sdk.getPythonSkillRuntimeInfo([skill]);
  if (runtimeInfo.length !== 1 || runtimeInfo[0].importName !== 'agent_message') {
    throw new Error('Prime agent-message Python skill has an incompatible import name');
  }
  const python = process.env.PRIME_AGENT_KERNEL_PYTHON || join(args.runtime, 'kernel/bin/python3');
  const probe = spawnSync(python, ['-c',
    'from pathlib import Path; import sys; import agent_message; from rlm import host_request; '
    + 'assert callable(agent_message.send) and callable(agent_message.list_agents) and callable(host_request); '
    + 'assert Path(agent_message.__file__).resolve() == (Path(sys.argv[1]) / "agent_message/__init__.py").resolve()',
    join(skillDirectory,'src')],
    {encoding:'utf8',timeout:30000,env:{...process.env,PYTHONDONTWRITEBYTECODE:'1'}});
  if (probe.error || probe.status !== 0) {
    throw new Error(`Prime agent-message kernel import failed: ${probe.error?.message || probe.stderr}`);
  }
  return {sdkVersion,aiVersion,skill:'agent-message',pythonImport:'agent_message'};
}

if (checking) {
  console.log(JSON.stringify(checkRuntime()));
  await ai.cleanupSessionResources();
} else {
mkdirSync(args.state, {recursive:true, mode:0o700});
if (args.captureDirectory) mkdirSync(args.captureDirectory,{recursive:true,mode:0o700});
for (const name of ['events.jsonl','parent-messages.jsonl']) {
  writeFileSync(join(args.state,name),'',{mode:0o600,flag:'wx'});
}
const events = [];
const deliveries = [];
const log = (name, value) => appendFileSync(join(args.state, name), JSON.stringify(value)+'\n');
const auth = sdk.AuthStorage.inMemory({}, {usePrimeCliConfig:false});
const modelFile = join(args.state, 'models.json');
writeFileSync(modelFile, JSON.stringify(args.models), {mode:0o600});
const registry = new sdk.ModelRegistry(auth, modelFile);
const model = registry.find('nemo', args.model);
if (!model) throw new Error('Gym model not found in Prime registry');
const settings = sdk.SettingsManager.inMemory({autoRefine:{enabled:false}, compaction:{enabled:false}, retry:{enabled:false}});
const loader = new sdk.DefaultResourceLoader({
  cwd:args.cwd, agentDir:args.state, settingsManager:settings,
  noExtensions:true, noContextFiles:true, noPromptTemplates:true, noThemes:true,
  appendSystemPrompt:args.system ? [args.system] : [],
  skillsOverride: base => ({...base, skills:base.skills.filter(skill => skill.name === 'agent-message')}),
});
await loader.reload();
if (!loader.getSkills().skills.some(skill => skill.name === 'agent-message' && skill.kind === 'python')) {
  throw new Error('Prime child requires the bundled agent-message Python skill');
}
const parent = {activeSessionId:'recording-parent', sessionId:'recording-parent', sessionName:'parent'};
let session;
let turns = 0;
let termination = 'completed';
const controller = {
  listAgents: () => ({agents:[{...parent,cwd:args.cwd,isStreaming:false,unfinishedActionCount:0}]}),
  roster: () => ({current:{name:'child',id:session?.sessionId ?? 'child',depth:1},entries:[{relationship:'parent',id:parent.sessionId,name:'parent',depth:0,status:'idle'}]}),
  sendAgentMessage: async ({target,message}) => {
    if (![parent.sessionId,parent.sessionName].includes(target)) throw new Error('Unknown parent recipient');
    const receipt = {id:randomUUID(),source:'agent_message',target:parent,message,deliveryStatus:'delivered',deliveredAt:new Date().toISOString()};
    deliveries.push(receipt);
    log('parent-messages.jsonl',receipt);
    return receipt;
  },
};
try {
  ({session} = await sdk.createAgentSession({
    cwd:args.cwd,agentDir:args.state,model,authStorage:auth,modelRegistry:registry,
    resourceLoader:loader,settingsManager:settings,sessionManager:sdk.SessionManager.create(args.cwd,join(args.state,'session')),
    tools:['ipython'],allowedToolNames:['ipython'],rlmDepth:1,rlmMaxDepth:1,
    rlmParentAgent:'parent',rlmParentNodeId:parent.sessionId,rlmSessionDir:join(args.state,'session'),
    agentMessageController:controller,includeGoals:false,includeCompactSkill:false,thinkingLevel:'off',
  }));
  if (session.getActiveToolNames().join(',') !== 'ipython') throw new Error('Expected only native IPython');
  writeFileSync(join(args.state,'effective-session.json'),JSON.stringify({rlmDepth:session.rlmDepth,systemPrompt:session.systemPrompt,tools:session.getAllTools().filter(t=>t.name==='ipython'),model:args.model,cwd:args.cwd}));
  session.subscribe(event => {
    const observed = {...event,gymObservedAt:Date.now()/1000};
    events.push(observed);
    log('events.jsonl',observed);
    if (event.type === 'message_end' && event.message.role === 'assistant') turns++;
    if (event.type === 'turn_end' && turns >= args.maxTurns && event.message.content.some(b=>b.type==='toolCall') && !['error','aborted'].includes(event.message.stopReason)) {
      termination='turn_limit';
      void session.abort();
    }
  });
  const previous = session.agent.onPayload;
  session.agent.onPayload = async (payload,selected) => {
    const adjusted = previous ? (await previous(payload,selected)) ?? payload : payload;
    const used = session.getContextUsage()?.tokens;
    const room = used == null ? args.maxTokens : Math.max(1,Math.floor(selected.contextWindow-used-8192));
    const request = {...adjusted,temperature:args.temperature,max_tokens:Math.min(args.maxTokens,room),parallel_tool_calls:false,
      tool_choice:deliveries.length ? 'auto' : {type:'function',function:{name:'ipython'}}};
    delete request.max_completion_tokens;
    log('model-requests.jsonl',request);
    return request;
  };
  await session.promptAndWait(`[task from parent]\n\n${args.prompt}`,{expandPromptTemplates:false,source:'extension'});
} catch (error) {
  termination='error';
  log('errors.jsonl',{error:String(error)});
} finally {
  try { await session?.disposeAsync(); ai.cleanupSessionResources(); }
  catch(error) {termination='error';log('errors.jsonl',{cleanup:String(error)});}
  const capture={version:1,termination,assistantTurns:turns,events,deliveries};
  writeFileSync(join(args.state,'capture.json'),JSON.stringify(capture));
  if (args.captureDirectory) {
    writeFileSync(join(args.captureDirectory,'capture.json'),JSON.stringify(capture),{mode:0o600});
  }
}

}
