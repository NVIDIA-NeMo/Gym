const elements = {
  actions: document.querySelector("#actions"),
  card0: document.querySelector("#card0"),
  card1: document.querySelector("#card1"),
  connectionDot: document.querySelector("#connection-dot"),
  connectionLabel: document.querySelector("#connection-label"),
  gameStatus: document.querySelector("#game-status"),
  handoff: document.querySelector("#handoff"),
  handoffTitle: document.querySelector("#handoff-title"),
  history: document.querySelector("#history"),
  message: document.querySelector("#message"),
  newHand: document.querySelector("#new-hand"),
  playControls: document.querySelector("#play-controls"),
  player0: document.querySelector("#player0"),
  player1: document.querySelector("#player1"),
  pot: document.querySelector("#pot"),
  reveal: document.querySelector("#reveal"),
  reward0: document.querySelector("#reward0"),
  reward1: document.querySelector("#reward1"),
  seed: document.querySelector("#seed"),
  status0: document.querySelector("#status0"),
  status1: document.querySelector("#status1"),
};

let mode = "play";
let playView = null;
let spectatorView = null;
let pendingView = null;
let eventSource = null;
let busy = false;

function playerLabel(agent) {
  return agent === "player1" ? "Player 1" : "Player 0";
}

function setConnection(label, state = "") {
  elements.connectionLabel.textContent = label;
  elements.connectionDot.className = `connection-dot ${state}`.trim();
}

function renderCard(element, card) {
  element.textContent = card || "?";
  element.classList.toggle("hidden-card", !card);
}

function renderHistory(history) {
  elements.history.replaceChildren();
  if (!history.length) {
    const empty = document.createElement("li");
    empty.className = "empty-state";
    empty.textContent = "No actions yet";
    elements.history.append(empty);
    return;
  }

  for (const turn of history) {
    const item = document.createElement("li");
    item.textContent = `${playerLabel(turn.agent)}: ${turn.action}`;
    elements.history.append(item);
  }
}

function renderActions(view, allowActions) {
  elements.actions.replaceChildren();
  if (!view) {
    const empty = document.createElement("span");
    empty.className = "empty-state";
    empty.textContent = mode === "play" ? "Deal a hand to choose an action" : "Waiting for a game";
    elements.actions.append(empty);
    return;
  }

  if (mode === "spectate") {
    const watching = document.createElement("span");
    watching.className = "empty-state";
    watching.textContent = "Spectator mode is read-only";
    elements.actions.append(watching);
    return;
  }

  if (view.status === "finished") {
    const done = document.createElement("span");
    done.className = "empty-state";
    done.textContent = "Hand complete";
    elements.actions.append(done);
    return;
  }

  if (!allowActions) {
    const hidden = document.createElement("span");
    hidden.className = "empty-state";
    hidden.textContent = "Waiting for the active player";
    elements.actions.append(hidden);
    return;
  }

  for (const action of view.legal_actions) {
    const button = document.createElement("button");
    button.className = "action-button";
    button.textContent = action;
    button.disabled = busy;
    button.addEventListener("click", () => takeAction(action));
    elements.actions.append(button);
  }
}

function resultText(view) {
  if (view.forfeited !== null && view.forfeited !== undefined) {
    return `${playerLabel(`player${view.forfeited}`)} forfeited`;
  }
  const winner = Object.entries(view.rewards).find(([, reward]) => reward > 0);
  return winner ? `${playerLabel(winner[0])} wins the hand` : "Hand complete";
}

function renderView(view, allowActions = true) {
  if (!view) {
    renderCard(elements.card0, null);
    renderCard(elements.card1, null);
    elements.player0.classList.remove("active");
    elements.player1.classList.remove("active");
    elements.status0.textContent = "Waiting";
    elements.status1.textContent = "Waiting";
    elements.reward0.textContent = "";
    elements.reward1.textContent = "";
    elements.pot.textContent = "2";
    elements.gameStatus.textContent =
      mode === "play" ? "Deal a hand to begin" : "Waiting for the current game";
    elements.message.textContent = "";
    renderHistory([]);
    renderActions(null, false);
    return;
  }

  renderCard(elements.card0, view.cards.player0);
  renderCard(elements.card1, view.cards.player1);
  elements.pot.textContent = String(view.pot);
  elements.player0.classList.toggle("active", view.active_agent === "player0");
  elements.player1.classList.toggle("active", view.active_agent === "player1");
  const activeStatus = mode === "play" ? "Your turn" : "To act";
  elements.status0.textContent = view.active_agent === "player0" ? activeStatus : "Waiting";
  elements.status1.textContent = view.active_agent === "player1" ? activeStatus : "Waiting";
  elements.reward0.textContent =
    view.rewards.player0 === undefined ? "" : `${view.rewards.player0 > 0 ? "+" : ""}${view.rewards.player0}`;
  elements.reward1.textContent =
    view.rewards.player1 === undefined ? "" : `${view.rewards.player1 > 0 ? "+" : ""}${view.rewards.player1}`;

  if (view.status === "finished") {
    elements.gameStatus.textContent = resultText(view);
    elements.status0.textContent = "Finished";
    elements.status1.textContent = "Finished";
  } else {
    elements.gameStatus.textContent = `${playerLabel(view.active_agent)} to act`;
  }

  elements.message.textContent = view.message || "";
  renderHistory(view.history);
  renderActions(view, allowActions);
}

function maskedView(view) {
  return {
    ...view,
    cards: { player0: null, player1: null },
  };
}

function queueHandoff(view) {
  pendingView = view;
  renderView(maskedView(view), false);
  elements.handoffTitle.textContent = `Pass to ${playerLabel(view.active_agent)}`;
  elements.handoff.classList.remove("hidden");
}

async function post(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || `Request failed with status ${response.status}`);
  }
  return payload;
}

function requestMetadata() {
  return {
    responses_create_params: { input: "Play one hand of Kuhn Poker." },
    verifier_metadata: { seed: Number(elements.seed.value || 0) },
  };
}

async function startHand() {
  if (busy) return;
  busy = true;
  elements.newHand.disabled = true;
  setConnection("Dealing…");
  try {
    const response = await post("/reset", requestMetadata());
    playView = response.info.view;
    setConnection("Private game", "live");
    queueHandoff(playView);
  } catch (error) {
    setConnection("Could not start game", "error");
    elements.message.textContent = error.message;
  } finally {
    busy = false;
    elements.newHand.disabled = false;
  }
}

async function takeAction(action) {
  if (busy || !playView?.active_agent) return;
  busy = true;
  renderActions(playView, true);
  elements.message.textContent = "";
  try {
    const response = await post("/step", {
      ...requestMetadata(),
      agent_id: playView.active_agent,
      action: `[${action}]`,
    });
    const nextView = response.info.view;
    const changedPlayer = nextView.active_agent && nextView.active_agent !== playView.active_agent;
    playView = nextView;
    if (nextView.status === "finished") {
      renderView(nextView);
    } else if (changedPlayer) {
      queueHandoff(nextView);
    } else {
      renderView(nextView);
    }
  } catch (error) {
    elements.message.textContent = error.message;
    renderView(playView);
  } finally {
    busy = false;
    if (elements.handoff.classList.contains("hidden")) {
      renderActions(playView, true);
    }
  }
}

function openSpectatorStream() {
  eventSource?.close();
  setConnection("Connecting to current game…");
  eventSource = new EventSource("/events");
  eventSource.addEventListener("open", () => setConnection("Watching live", "live"));
  eventSource.addEventListener("message", (event) => {
    spectatorView = JSON.parse(event.data);
    if (mode === "spectate") renderView(spectatorView);
  });
  eventSource.addEventListener("error", () => setConnection("Reconnecting…", "error"));
}

function setMode(nextMode) {
  mode = nextMode;
  pendingView = null;
  elements.handoff.classList.add("hidden");
  document.querySelectorAll(".mode-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  elements.playControls.classList.toggle("hidden", mode !== "play");

  if (mode === "spectate") {
    renderView(spectatorView);
    openSpectatorStream();
  } else {
    eventSource?.close();
    eventSource = null;
    setConnection(playView ? "Private game" : "Ready to play", playView ? "live" : "");
    if (playView?.status === "playing") {
      queueHandoff(playView);
    } else {
      renderView(playView);
    }
  }
}

document.querySelectorAll(".mode-button").forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});
elements.newHand.addEventListener("click", startHand);
elements.reveal.addEventListener("click", () => {
  elements.handoff.classList.add("hidden");
  if (pendingView) {
    playView = pendingView;
    pendingView = null;
    renderView(playView);
  }
});

renderView(null);
