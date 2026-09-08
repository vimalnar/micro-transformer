"use strict";

const colours = {
  unknown: "#121930",
  grid: "#68707a",
  outline: "#141e34",
  floor: "#f2eedb",
  wall: "#344254",
  door: "#9b8764",
  water: "#aebdc0",
  ice: "#d9e0df",
  "push-north": "#d8d1b8",
  "push-east": "#d8d1b8",
  "push-south": "#d8d1b8",
  "push-west": "#d8d1b8",
  "spin-clockwise": "#d1c9d4",
  "spin-anticlockwise": "#d1c9d4",
  "pressure-plate": "#cbbbbb",
  lamp: "#ddd4aa",
  teal: "#30b0be",
  coral: "#ee5c47",
  red: "#e0423e",
  green: "#4f953d",
  amber: "#f6b82d",
  brown: "#8f5c30",
  neutral: "#dce0e4",
};

const directions = {
  north: [0, -1],
  east: [1, 0],
  south: [0, 1],
  west: [-1, 0],
};

let sessionId = null;
let latest = null;
let playTimer = null;

const $ = (selector) => document.querySelector(selector);

function shade(hex, factor) {
  const value = Number.parseInt(hex.slice(1), 16);
  const channels = [value >> 16, (value >> 8) & 255, value & 255]
    .map((channel) => Math.max(0, Math.min(255, Math.round(channel * factor))));
  return `rgb(${channels[0]}, ${channels[1]}, ${channels[2]})`;
}

function phaseColour(kind, phase) {
  const factor = { day: 1, dusk: 0.82, night: 0.55, dawn: 0.75 }[phase] ?? 1;
  return shade(colours[kind] ?? colours.floor, factor);
}

function drawArrow(ctx, x, y, size, direction) {
  const [dx, dy] = directions[direction];
  const cx = x + size / 2;
  const cy = y + size / 2;
  ctx.strokeStyle = colours.outline;
  ctx.lineWidth = Math.max(1, size / 16);
  ctx.beginPath();
  ctx.moveTo(cx - dx * size * 0.22, cy - dy * size * 0.22);
  ctx.lineTo(cx + dx * size * 0.22, cy + dy * size * 0.22);
  ctx.stroke();
}

function drawTile(ctx, tile, x, y, size, phase, visible = true) {
  const kind = visible && tile ? tile.kind : "unknown";
  ctx.fillStyle = kind === "unknown" ? colours.unknown : phaseColour(kind, phase);
  ctx.fillRect(x, y, size, size);

  // Terrain is communicated with quiet shading plus a minimal mark. This keeps
  // large connected areas from looking like a movement trail.
  if (kind === "water") {
    ctx.strokeStyle = shade(colours.water, 0.72);
    ctx.lineWidth = Math.max(1, size / 24);
    for (const offset of [0.38, 0.62]) {
      ctx.beginPath();
      ctx.moveTo(x + size * 0.24, y + size * offset);
      ctx.lineTo(x + size * 0.76, y + size * offset);
      ctx.stroke();
    }
  }
  if (kind === "ice") {
    ctx.strokeStyle = shade(colours.ice, 0.78);
    ctx.lineWidth = Math.max(1, size / 28);
    ctx.beginPath();
    ctx.moveTo(x + size * 0.22, y + size * 0.78);
    ctx.lineTo(x + size * 0.78, y + size * 0.22);
    ctx.stroke();
  }
  if (kind.startsWith("push-")) drawArrow(ctx, x, y, size, kind.slice(5));
  if (kind.startsWith("spin-")) {
    ctx.strokeStyle = colours.outline;
    ctx.lineWidth = Math.max(1, size / 16);
    ctx.beginPath();
    ctx.arc(x + size / 2, y + size / 2, size * 0.24, 0.2, Math.PI * 1.7);
    ctx.stroke();
  }
  if (kind === "door") {
    const inset = tile.open ? size * 0.35 : size * 0.2;
    ctx.strokeStyle = colours.outline;
    ctx.lineWidth = Math.max(1, size / 16);
    ctx.strokeRect(x + inset, y + size * 0.12, size - inset * 2, size * 0.76);
  }
  if (kind === "pressure-plate") {
    ctx.strokeStyle = colours.outline;
    ctx.strokeRect(x + size * 0.25, y + size * 0.25, size * 0.5, size * 0.5);
  }
  if (kind === "lamp") {
    ctx.fillStyle = "#fff491";
    ctx.beginPath();
    ctx.arc(x + size / 2, y + size / 2, size * 0.18, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.strokeStyle = colours.grid;
  ctx.lineWidth = 1;
  ctx.strokeRect(x, y, size, size);
  return ctx.fillStyle;
}

function drawEntity(ctx, entity, x, y, size, background, viewDirection = "north") {
  const cx = x + size / 2;
  const cy = y + size / 2;
  const colour = colours[entity.colour] ?? colours.neutral;
  ctx.strokeStyle = colours.outline;
  ctx.lineWidth = Math.max(1, size / 18);

  if (entity.kind === "agent") {
    const order = ["north", "east", "south", "west"];
    const relative = (order.indexOf(entity.direction) - order.indexOf(viewDirection) + 4) % 4;
    const angle = -Math.PI / 2 + relative * Math.PI / 2;
    ctx.fillStyle = colour;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, size * 0.34, angle + 0.45, angle - 0.45 + Math.PI * 2);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  } else if (entity.kind === "orb") {
    ctx.fillStyle = colour;
    ctx.beginPath();
    ctx.arc(cx, cy, size * 0.27, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  } else if (entity.kind === "cube" || entity.kind === "crate") {
    ctx.fillStyle = colour;
    ctx.fillRect(x + size * 0.22, y + size * 0.22, size * 0.56, size * 0.56);
    ctx.strokeRect(x + size * 0.22, y + size * 0.22, size * 0.56, size * 0.56);
    if (entity.kind === "crate") {
      ctx.beginPath();
      ctx.moveTo(x + size * 0.27, y + size * 0.27);
      ctx.lineTo(x + size * 0.73, y + size * 0.73);
      ctx.moveTo(x + size * 0.73, y + size * 0.27);
      ctx.lineTo(x + size * 0.27, y + size * 0.73);
      ctx.stroke();
    }
  } else if (entity.kind === "key") {
    ctx.strokeStyle = colour;
    ctx.lineWidth = Math.max(2, size * 0.12);
    ctx.beginPath();
    ctx.arc(x + size * 0.38, y + size * 0.4, size * 0.15, 0, Math.PI * 2);
    ctx.moveTo(x + size * 0.5, y + size * 0.5);
    ctx.lineTo(x + size * 0.78, y + size * 0.7);
    ctx.stroke();
  }
}

function renderObservation(observation) {
  const canvas = $("#observation");
  const ctx = canvas.getContext("2d");
  const size = canvas.width / observation.width;
  ctx.imageSmoothingEnabled = false;
  ctx.fillStyle = colours.unknown;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  observation.cells.flat().forEach((cell) => {
    const x = cell.local_x * size;
    const y = cell.local_y * size;
    const background = drawTile(ctx, cell.tile, x, y, size, observation.phase, cell.visible);
    cell.entities.forEach((entity) =>
      drawEntity(ctx, entity, x, y, size, background, observation.view_direction));
  });
}

function visibleWorldCells(observation) {
  return new Set(
    observation.cells
      .flat()
      .filter((cell) => cell.visible && cell.world_x !== null && cell.world_y !== null)
      .map((cell) => `${cell.world_x},${cell.world_y}`),
  );
}

function renderWorld(state, observation, revealAll = false) {
  const canvas = $("#world");
  const ctx = canvas.getContext("2d");
  const size = canvas.width / state.config.width;
  const visible = visibleWorldCells(observation);
  ctx.imageSmoothingEnabled = false;
  state.tiles.forEach((row, y) => row.forEach((tile, x) => {
    const revealed = revealAll || visible.has(`${x},${y}`);
    drawTile(ctx, tile, x * size, y * size, size, state.phase, revealed);
  }));
  Object.values(state.entities).forEach((entity) => {
    if (entity.x < 0 || entity.y < 0) return;
    if (!revealAll && !visible.has(`${entity.x},${entity.y}`)) return;
    const tile = state.tiles[entity.y][entity.x];
    drawEntity(ctx, entity, entity.x * size, entity.y * size, size,
      revealAll || visible.has(`${entity.x},${entity.y}`)
        ? phaseColour(tile.kind, state.phase)
        : colours.unknown,
      "north");
  });
}

function describeEvent(event) {
  const actor = event.actor ? `${event.actor} ` : "";
  const position = event.position ? ` @ ${event.position.join(",")}` : "";
  return `${event.tick}: ${actor}${event.kind}${position}`;
}

function render(payload) {
  latest = payload;
  $("#tick").textContent = payload.state.tick;
  $("#phase").textContent = payload.state.phase;
  $("#agent-name").textContent = payload.observation.agent_id;
  renderObservation(payload.observation);
  renderWorld(payload.state, payload.observation, $("#reveal-world").checked);
  const events = payload.events ?? [];
  $("#events").replaceChildren(...events.slice(-30).reverse().map((event) => {
    const item = document.createElement("li");
    item.textContent = describeEvent(event);
    return item;
  }));
  $("#replay").textContent = JSON.stringify(payload.replay ?? {
    seed: payload.state.seed,
    map_name: payload.state.map_name,
    tick: payload.state.tick,
  }, null, 2);
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function createSession() {
  const payload = await request("/sessions", {
    method: "POST",
    body: JSON.stringify({
      seed: Number($("#seed").value),
      map_name: $("#map-name").value,
      agent_id: $("#agent-select").value,
    }),
  });
  sessionId = payload.session_id;
  render(payload);
}

async function resetSession() {
  if (!sessionId) return createSession();
  const payload = await request(`/sessions/${sessionId}/reset`, {
    method: "POST",
    body: JSON.stringify({
      seed: Number($("#seed").value),
      map_name: $("#map-name").value,
      agent_id: $("#agent-select").value,
    }),
  });
  render(payload);
}

async function step(action) {
  if (!sessionId) return;
  const payload = await request(`/sessions/${sessionId}/step`, {
    method: "POST",
    body: JSON.stringify({ action, agent_id: $("#agent-select").value }),
  });
  payload.replay = latest?.replay ?? payload.replay;
  if (payload.replay) payload.replay.actions = [
    ...(payload.replay.actions ?? []),
    { tick: payload.state.tick, agent_id: $("#agent-select").value, action },
  ];
  render(payload);
}

function togglePlay() {
  if (playTimer) {
    clearInterval(playTimer);
    playTimer = null;
    $("#play").textContent = "Play";
    return;
  }
  playTimer = setInterval(() => step("wait"), Number($("#speed").value));
  $("#play").textContent = "Pause";
}

document.querySelectorAll("[data-action]").forEach((button) =>
  button.addEventListener("click", () => step(button.dataset.action)));
$("#reset").addEventListener("click", resetSession);
$("#play").addEventListener("click", togglePlay);
$("#agent-select").addEventListener("change", async () => {
  if (!sessionId) return;
  render(await request(`/sessions/${sessionId}/state?agent_id=${$("#agent-select").value}`));
});
$("#speed").addEventListener("change", () => {
  if (playTimer) { togglePlay(); togglePlay(); }
});
$("#reveal-world").addEventListener("change", () => {
  if (latest) renderWorld(latest.state, latest.observation, $("#reveal-world").checked);
});

window.addEventListener("keydown", (event) => {
  const action = {
    ArrowUp: "forward",
    ArrowLeft: "turn-left",
    ArrowRight: "turn-right",
    " ": "wait",
  }[event.key];
  if (action) { event.preventDefault(); step(action); }
});

createSession().catch((error) => {
  $("#events").textContent = error.message;
});
