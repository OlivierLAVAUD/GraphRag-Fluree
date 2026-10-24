"use strict";

const $ = (sel) => document.querySelector(sel);
const messagesEl = $("#messages");
const form = $("#chat-form");
const input = $("#question");
const sendBtn = $("#send-btn");

// ---------------------------------------------------------------------------
// État des services (barre du haut)
// ---------------------------------------------------------------------------

async function refreshStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    // Ollama renvoie les noms avec un tag (`nomic-embed-text:latest`) alors que
    // la config peut n'en préciser aucun : on normalise avant de comparer,
    // sinon le modèle installé apparaît à tort en rouge.
    const withTag = (name) =>
      !name || name.includes(":") ? name : `${name}:latest`;
    const installed = data.ollama_models.map(withTag);
    const hasChat = installed.includes(withTag(data.chat_model));
    const hasEmbed = installed.includes(withTag(data.embed_model));
    const chip = (ok, label) => `<span class="${ok ? "ok" : "err"}">${ok ? "●" : "○"} ${label}</span>`;
    $("#header-status").innerHTML =
      `Ledger : <b>${data.ledger}</b> · ` +
      chip(true, "Fluree") +
      " " + chip(hasChat, data.chat_model) +
      " " + chip(hasEmbed, data.embed_model);
  } catch (err) {
    $("#header-status").innerHTML = `<span class="err">Services injoignables : ${err}</span>`;
  }
}

// ---------------------------------------------------------------------------
// Chat (streaming SSE via fetch + ReadableStream)
// ---------------------------------------------------------------------------

function bubble(text) {
  const div = document.createElement("div");
  div.className = "bubble";
  div.textContent = text;
  return div;
}

function addMessage(role, text) {
  const wrap = document.createElement("div");
  wrap.className = `message ${role}`;
  wrap.appendChild(bubble(text));
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return wrap;
}

function addSources(sources) {
  const row = document.createElement("div");
  row.className = "sources-row";
  row.innerHTML = sources
    .map((s, i) => `<span class="source-chip"><b>${i + 1}</b> ${s.source}#${s.index} · score ${s.score}</span>`)
    .join(" ");
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function parseSse(text) {
  return text
    .split("\n\n")
    .filter((block) => block.startsWith("data: "))
    .map((block) => JSON.parse(block.slice(6)));
}

async function sendQuestion(question) {
  addMessage("user", question);
  const wrap = addMessage("assistant", "");
  const textNode = wrap.querySelector(".bubble");
  textNode.innerHTML = '<span class="cursor"></span>';

  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) {
    const detail = (await res.json()).detail || res.statusText;
    textNode.textContent = `Erreur : ${detail}`;
    textNode.classList.add("error-text");
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let answer = "";
  let sources = null;
  let errored = false;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop();
    for (const block of blocks) {
      const events = parseSse(block);
      for (const ev of events) {
        if (ev.type === "token") {
          answer += ev.content;
          textNode.textContent = answer + "▍";
          messagesEl.scrollTop = messagesEl.scrollHeight;
        } else if (ev.type === "sources") {
          sources = ev.sources;
        } else if (ev.type === "error") {
          errored = true;
          textNode.textContent = `⚠ ${ev.message}`;
          textNode.classList.add("error-text");
        } else if (ev.type === "done") {
          sources = ev.sources;
        }
      }
    }
  }
  if (buffer.trim()) {
    for (const ev of parseSse(buffer)) {
      if (ev.type === "token") {
        answer += ev.content;
        textNode.textContent = answer;
      } else if (ev.type === "sources") {
        sources = ev.sources;
      }
    }
  }
  textNode.textContent = answer;
  if (sources && sources.length) addSources(sources);
  if (!errored) refreshStatus();
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  sendBtn.disabled = true;
  try {
    await sendQuestion(question);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
});

// ---------------------------------------------------------------------------
// Upload de documents
// ---------------------------------------------------------------------------

$("#upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const files = $("#file-input").files;
  if (!files.length) return;
  const btn = $("#upload-btn");
  btn.disabled = true;
  $("#ingest-result").innerHTML = "Indexation en cours…";
  const fd = new FormData();
  for (const f of files) fd.append("files", f);
  try {
    const res = await fetch("/api/ingest", { method: "POST", body: fd });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const data = await res.json();
    const rows = data.ingested
      .map((r) => `<div class="ok-row">✓ ${r.file} — ${r.chars} caractères → ${r.chunks} chunks</div>`)
      .join("");
    const total = data.ingested.reduce((n, r) => n + r.chunks, 0);
    $("#ingest-result").innerHTML = rows + `<div class="ok-row"><b>${total} chunks insérés dans Fluree.</b></div>`;
    refreshStatus();
  } catch (err) {
    $("#ingest-result").innerHTML = `<div class="err-row">Erreur : ${err}</div>`;
  } finally {
    btn.disabled = false;
    $("#file-input").value = "";
  }
});

// ---------------------------------------------------------------------------
// Onglets + état GPU / modèles
// ---------------------------------------------------------------------------

function switchTab(name) {
  $("#tab-docs").classList.toggle("active", name === "docs");
  $("#tab-gpu").classList.toggle("active", name === "gpu");
  $("#view-docs").classList.toggle("hidden", name !== "docs");
  $("#view-gpu").classList.toggle("hidden", name !== "gpu");
  if (name === "gpu") refreshGpu();
}
$("#tab-docs").addEventListener("click", () => switchTab("docs"));
$("#tab-gpu").addEventListener("click", () => switchTab("gpu"));

async function refreshGpu() {
  try {
    const data = await (await fetch("/api/gpu")).json();
    const rows = data.models.map((m) => {
      const vram = m.size_vram ? `${(m.size_vram / 1e9).toFixed(2)} Go VRAM` : "?";
      const gpu = m.gpus ? ` — GPU : <b>${m.gpus}</b>` : " (CPU)";
      return `<div class="${m.size_vram ? "ok-row" : "err-row"}">${m.name} · ${vram}${gpu}</div>`;
    });
    $("#gpu-status").innerHTML =
      rows.length
        ? `Modèles chargés en mémoire :<br>${rows.join("<br>")}`
        : "Aucun modèle chargé pour l'instant (ils se chargent à la première question).";
    const models = await (await fetch("/api/status")).json();
    $("#models-status").innerHTML =
      `Embedding : <b>${models.embed_model}</b><br>Génération : <b>${models.chat_model}</b><br>` +
      `Disponibles dans Ollama : ${models.ollama_models.join(", ") || "aucun"}`;
  } catch (err) {
    $("#gpu-status").innerHTML = `<div class="err-row">Ollama injoignable : ${err}</div>`;
  }
}

// ---------------------------------------------------------------------------

refreshStatus();