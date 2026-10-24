"use strict";

(() => {
  const $ = (selector, scope = document) => scope.querySelector(selector);
  const $$ = (selector, scope = document) => Array.from(scope.querySelectorAll(selector));

  const body = document.body;
  const chatEl = $("#chat");
  const messagesEl = $("#messages");
  const emptyState = $("#emptyState");
  const composer = $("#composer");
  const sendButton = $("#sendButton");
  const chatTitle = $("#chatTitle");
  const toastEl = $("#toast");
  const fileInput = $("#fileInput");
  const attachment = $("#attachment");
  const attachmentName = $("#attachmentName");
  const conversationsEl = $("#conversations");
  const suggestionsEl = $("#suggestions");
  const suggestionsSource = $("#suggestionsSource");
  const emptyHeading = $("#emptyHeading");
  const emptySubtitle = $("#emptySubtitle");

  const libraryModal = $("#libraryModal");
  const sourceModal = $("#sourceModal");
  const modelButton = $("#modelButton");
  const modelMenu = $("#modelMenu");
  const moreButton = $("#moreActions");
  const moreMenu = $("#moreMenu");

  const SUGGESTION_FILE = "/static/suggestions.json";
  const STORAGE_KEY = "graphrag.conversations.v1";
  const THEME_KEY = "graphrag-theme";
  const MAX_TITLE_LENGTH = 60;

  const SUGGESTION_ICONS = new Set([
    "sparkles", "doc", "compare", "shield", "check-search", "search", "database", "filter"
  ]);
  const DEFAULT_SUGGESTION_CONFIG = {
    heading: "Comment puis-je vous aider ?",
    subtitle: "Interrogez votre base de connaissances. Les réponses sont générées localement et citent toujours leurs sources.",
    columns: 2,
    suggestions: [
      {
        title: "Interroger la base",
        description: "Quelles entreprises fournissent des solutions de calcul quantique ?",
        prompt: "Quelles entreprises françaises fournissent des solutions de calcul quantique ?",
        icon: "check-search"
      },
      {
        title: "Résumer un document",
        description: "Résume les points clés du rapport France 2030.",
        prompt: "Résume les points clés du rapport France 2030.",
        icon: "doc"
      },
      {
        title: "Comparer des sources",
        description: "Compare les informations de La Tribune et du Figaro.",
        prompt: "Compare les informations de La Tribune et du Figaro sur le partenariat Microsoft-Mistral.",
        icon: "compare"
      },
      {
        title: "Vérifier une affirmation",
        description: "Le partenariat implique-t-il une prise de participation ?",
        prompt: "Le partenariat Microsoft-Mistral implique-t-il une prise de participation ?",
        icon: "shield"
      }
    ]
  };

  let conversations = loadConversations();
  let activeId = null;
  let isStreaming = false;
  let statusCache = null;
  let toastTimer = null;

  // ---------------------------------------------------------------------------
  // Utilitaires
  // ---------------------------------------------------------------------------

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (char) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    })[char]);
  }

  function toast(message) {
    toastEl.textContent = message;
    toastEl.classList.add("show");
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => toastEl.classList.remove("show"), 2800);
  }

  function scrollToBottom(behavior = "smooth") {
    chatEl.scrollTo({ top: chatEl.scrollHeight, behavior });
  }

  function setTitle(title) {
    chatTitle.textContent = title;
  }

  function newId() {
    if (window.crypto?.randomUUID) return window.crypto.randomUUID();
    return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  function titleFromQuestion(question) {
    const text = question.replace(/\s+/g, " ").trim();
    return text.length > MAX_TITLE_LENGTH
      ? `${text.slice(0, MAX_TITLE_LENGTH - 1).trimEnd()}…`
      : text;
  }

  function formatScore(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(2).replace(".", ",") : "—";
  }

  function startOfDay(timestamp) {
    const date = new Date(timestamp);
    date.setHours(0, 0, 0, 0);
    return date.getTime();
  }

  // ---------------------------------------------------------------------------
  // Historique local des conversations
  // ---------------------------------------------------------------------------

  function loadConversations() {
    try {
      const parsed = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || "[]");
      if (!Array.isArray(parsed)) return [];
      return parsed
        .filter((conv) => conv && typeof conv.id === "string" && Array.isArray(conv.messages))
        .map((conv) => ({
          id: conv.id,
          title: typeof conv.title === "string" && conv.title.trim() ? conv.title : "Conversation",
          createdAt: Number(conv.createdAt) || Date.now(),
          updatedAt: Number(conv.updatedAt) || Number(conv.createdAt) || Date.now(),
          messages: conv.messages
            .filter((message) => message && (message.role === "user" || message.role === "assistant"))
            .map((message) => ({
              role: message.role,
              content: typeof message.content === "string" ? message.content : "",
              sources: Array.isArray(message.sources) ? message.sources : [],
              error: Boolean(message.error)
            }))
        }));
    } catch (error) {
      return [];
    }
  }

  function saveConversations() {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
    } catch (error) {
      console.warn("Historique local indisponible", error);
    }
  }

  function getActiveConversation() {
    return conversations.find((conv) => conv.id === activeId) || null;
  }

  function createConversation(question) {
    const now = Date.now();
    const conv = {
      id: newId(),
      title: titleFromQuestion(question),
      createdAt: now,
      updatedAt: now,
      messages: []
    };
    conversations.push(conv);
    activeId = conv.id;
    return conv;
  }

  function renderConversations() {
    const query = $("#searchInput").value.trim().toLowerCase();
    const sorted = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt);
    const groups = [
      { label: "Aujourd'hui", items: [] },
      { label: "7 derniers jours", items: [] },
      { label: "Plus ancien", items: [] }
    ];
    const today = startOfDay(Date.now());

    sorted.forEach((conv) => {
      const haystack = `${conv.title} ${conv.messages.map((m) => m.content).join(" ")}`.toLowerCase();
      if (query && !haystack.includes(query)) return;
      const day = startOfDay(conv.updatedAt);
      if (day === today) groups[0].items.push(conv);
      else if (day >= today - 6 * 86400000) groups[1].items.push(conv);
      else groups[2].items.push(conv);
    });

    conversationsEl.replaceChildren();

    if (!conversations.length) {
      const empty = document.createElement("p");
      empty.className = "conv-empty";
      empty.textContent = "Aucune conversation pour l'instant.";
      conversationsEl.appendChild(empty);
      return;
    }

    if (!groups.some((group) => group.items.length)) {
      const empty = document.createElement("p");
      empty.className = "conv-empty";
      empty.textContent = "Aucune conversation ne correspond à la recherche.";
      conversationsEl.appendChild(empty);
      return;
    }

    groups.forEach((group) => {
      if (!group.items.length) return;
      const section = document.createElement("div");
      section.className = "conv-group";

      const label = document.createElement("p");
      label.className = "conv-group__label";
      label.textContent = group.label;
      section.appendChild(label);

      group.items.forEach((conv) => {
        const item = document.createElement("button");
        item.type = "button";
        item.className = "conv-item";
        if (conv.id === activeId) item.classList.add("active");
        item.innerHTML = `
          <svg class="icon"><use href="#i-sparkles"/></svg>
          <span class="conv-item__title"></span>
          <span class="conv-item__more" role="button" tabindex="0" title="Supprimer la conversation" aria-label="Supprimer la conversation">
            <svg class="icon icon--sm"><use href="#i-more"/></svg>
          </span>`;
        $(".conv-item__title", item).textContent = conv.title;
        item.addEventListener("click", (event) => {
          if (event.target.closest(".conv-item__more")) {
            deleteConversation(conv.id);
            return;
          }
          openConversation(conv.id);
        });
        const more = $(".conv-item__more", item);
        more.addEventListener("keydown", (event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            deleteConversation(conv.id);
          }
        });
        section.appendChild(item);
      });

      conversationsEl.appendChild(section);
    });
  }

  function deleteConversation(id) {
    if (isStreaming) {
      toast("Attendez la fin de la génération.");
      return;
    }
    const conv = conversations.find((entry) => entry.id === id);
    if (!conv) return;
    if (!window.confirm(`Supprimer « ${conv.title} » ?`)) return;
    conversations = conversations.filter((entry) => entry.id !== id);
    if (activeId === id) startNewConversation();
    saveConversations();
    renderConversations();
    toast("Conversation supprimée.");
  }

  function openConversation(id) {
    if (isStreaming) {
      toast("Attendez la fin de la génération.");
      return;
    }
    const conv = conversations.find((entry) => entry.id === id);
    if (!conv) return;

    activeId = conv.id;
    clearMessages();
    emptyState.hidden = true;
    setTitle(conv.title);

    conv.messages.forEach((message, index) => {
      if (message.role === "user") {
        appendUserMessage(message.content, index);
      } else {
        const article = createAssistantShell(index);
        finishAssistantMessage(article, message);
      }
    });

    renderConversations();
    closeSidebar();
    requestAnimationFrame(() => scrollToBottom("auto"));
  }

  function startNewConversation() {
    if (isStreaming) {
      toast("Attendez la fin de la génération.");
      return;
    }
    activeId = null;
    clearMessages();
    emptyState.hidden = false;
    setTitle("Nouvelle conversation");
    resetComposer();
    renderConversations();
    closeSidebar();
    composer.focus();
  }

  // ---------------------------------------------------------------------------
  // Rendu markdown et messages
  // ---------------------------------------------------------------------------

  function inlineMarkdown(value) {
    return value
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.+?)\*/g, "<em>$1</em>")
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\[(\d+)\]/g, '<span class="cite">$1</span>');
  }

  function renderMarkdown(markdown) {
    const lines = escapeHtml(markdown || "").split("\n");
    let html = "";
    let listOpen = null;
    let paragraph = [];

    const flushParagraph = () => {
      if (paragraph.length) {
        html += `<p>${inlineMarkdown(paragraph.join(" "))}</p>`;
        paragraph = [];
      }
    };
    const closeList = () => {
      if (listOpen) {
        html += `</${listOpen}>`;
        listOpen = null;
      }
    };

    lines.forEach((line) => {
      const heading = line.match(/^(#{1,3})\s+(.*)$/);
      const unordered = line.match(/^\s*[-*]\s+(.*)$/);
      const ordered = line.match(/^\s*\d+\.\s+(.*)$/);

      if (heading) {
        flushParagraph();
        closeList();
        const level = heading[1].length;
        html += `<h${level + 1}>${inlineMarkdown(heading[2])}</h${level + 1}>`;
      } else if (unordered || ordered) {
        flushParagraph();
        const kind = unordered ? "ul" : "ol";
        if (listOpen !== kind) {
          closeList();
          html += `<${kind}>`;
          listOpen = kind;
        }
        html += `<li>${inlineMarkdown((unordered || ordered)[1])}</li>`;
      } else if (!line.trim()) {
        flushParagraph();
        closeList();
      } else {
        closeList();
        paragraph.push(line.trim());
      }
    });

    flushParagraph();
    closeList();
    return html;
  }

  function clearMessages() {
    $$(".message", messagesEl).forEach((node) => node.remove());
  }

  function hideEmptyState() {
    emptyState.hidden = true;
  }

  function appendUserMessage(text, index) {
    hideEmptyState();
    const article = document.createElement("article");
    article.className = "message message--user";
    if (Number.isInteger(index)) article.dataset.messageIndex = String(index);
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = escapeHtml(text).replace(/\n/g, "<br>");
    article.appendChild(bubble);
    messagesEl.appendChild(article);
    scrollToBottom();
    return article;
  }

  function createAssistantShell(index) {
    hideEmptyState();
    const article = document.createElement("article");
    article.className = "message message--assistant";
    if (Number.isInteger(index)) article.dataset.messageIndex = String(index);
    article.innerHTML = `
      <div class="message__avatar"><svg class="icon"><use href="#i-logo"/></svg></div>
      <div class="message__body">
        <div class="message__name">GraphRag <span class="badge badge--soft">local</span></div>
        <div class="message__content"></div>
        <div class="msg-actions">
          <button class="icon-btn icon-btn--sm" data-action="copy" title="Copier la réponse" aria-label="Copier la réponse">
            <svg class="icon icon--sm"><use href="#i-copy"/></svg>
          </button>
          <button class="icon-btn icon-btn--sm" data-action="regenerate" title="Régénérer" aria-label="Régénérer">
            <svg class="icon icon--sm"><use href="#i-refresh"/></svg>
          </button>
          <span class="msg-actions__meta"></span>
        </div>
      </div>`;
    messagesEl.appendChild(article);
    return article;
  }

  function buildSources(sources) {
    const wrap = document.createElement("div");
    wrap.className = "sources";
    const threshold = formatScore(statusCache?.min_score ?? 0.3);

    wrap.innerHTML = `
      <div class="sources__head">
        <span>Sources</span>
        <span class="sources__meta">${sources.length} extraits · seuil ${threshold}</span>
      </div>
      <div class="source-list"></div>`;

    const list = $(".source-list", wrap);
    sources.forEach((raw, index) => {
      const source = {
        name: String(raw.source || raw.name || "document"),
        rank: Number.isFinite(Number(raw.index)) ? Number(raw.index) : index,
        score: Number(raw.score) || 0,
        snippet: String(raw.snippet || "").trim() || "Extrait non disponible."
      };
      const card = document.createElement("button");
      card.type = "button";
      card.className = "source-card";
      card.innerHTML = `
        <span class="source-card__icon"><svg class="icon"><use href="#i-doc"/></svg></span>
        <span class="source-card__body">
          <span class="source-card__title">
            ${escapeHtml(source.name)}
            <span class="source-card__rank">extrait #${source.rank}</span>
          </span>
          <span class="source-card__snippet">${escapeHtml(source.snippet)}</span>
        </span>
        <span class="score ${source.score < 0.55 ? "score--mid" : ""}">${formatScore(source.score)}</span>`;
      card._source = source;
      list.appendChild(card);
    });

    return wrap;
  }

  function finishAssistantMessage(article, message) {
    const content = $(".message__content", article);
    const bodyEl = $(".message__body", article);
    const meta = $(".msg-actions__meta", article);
    const sources = Array.isArray(message.sources) ? message.sources : [];

    if (message.error) content.classList.add("is-error");
    else content.classList.remove("is-error");

    content.innerHTML = renderMarkdown(
      message.content || (message.error ? "Erreur de génération." : "Aucune réponse n'a été générée.")
    );

    $(".sources", bodyEl)?.remove();
    if (sources.length) {
      bodyEl.appendChild(buildSources(sources));
      const best = Math.max(...sources.map((source) => Number(source.score) || 0));
      const label = sources.length > 1 ? "sources" : "source";
      meta.textContent = `${sources.length} ${label} · meilleur score ${formatScore(best)}`;
    } else {
      meta.textContent = message.error ? "Erreur" : "Aucune source";
    }

    $(".msg-actions", article).classList.add("visible");
    scrollToBottom();
  }

  function setAssistantTyping(article) {
    const content = $(".message__content", article);
    content.classList.remove("is-error");
    content.innerHTML = '<div class="typing"><span></span><span></span><span></span></div>';
  }

  // ---------------------------------------------------------------------------
  // Chat réel (SSE)
  // ---------------------------------------------------------------------------

  function parseSseBlock(block) {
    return block
      .split("\n")
      .filter((line) => line.startsWith("data: "))
      .map((line) => {
        try {
          return JSON.parse(line.slice(6));
        } catch (error) {
          return null;
        }
      })
      .filter(Boolean);
  }

  async function streamAnswer(question, conv, message, article) {
    isStreaming = true;
    updateSendState();
    setAssistantTyping(article);
    $(".sources", article)?.remove();
    $(".msg-actions", article).classList.remove("visible");
    $(".msg-actions__meta", article).textContent = "";
    scrollToBottom();

    let answer = "";
    let sources = [];
    let receivedToken = false;

    const applyEvent = (event) => {
      if (event.type === "sources") {
        sources = Array.isArray(event.sources) ? event.sources : [];
        message.sources = sources;
      } else if (event.type === "token") {
        if (!receivedToken) {
          receivedToken = true;
          $(".message__content", article).innerHTML = "";
        }
        answer += event.content || "";
        message.content = answer;
        $(".message__content", article).innerHTML =
          renderMarkdown(answer) + '<span class="cursor"></span>';
        scrollToBottom();
      } else if (event.type === "done") {
        if (Array.isArray(event.sources)) {
          sources = event.sources;
          message.sources = sources;
        }
      } else if (event.type === "error") {
        throw new Error(event.message || "Erreur de génération.");
      }
    };

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question })
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || response.statusText || "Requête refusée.");
      }
      if (!response.body) throw new Error("Réponse streaming indisponible.");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const blocks = buffer.split("\n\n");
        buffer = blocks.pop();
        blocks.forEach((block) => parseSseBlock(block).forEach(applyEvent));
      }
      if (buffer.trim()) parseSseBlock(buffer).forEach(applyEvent);

      if (!answer.trim()) throw new Error("Le modèle n'a renvoyé aucun texte.");

      message.content = answer;
      message.sources = sources;
      message.error = false;
      finishAssistantMessage(article, message);

      conv.updatedAt = Date.now();
      saveConversations();
      renderConversations();
      refreshStatus();
    } catch (error) {
      message.content = `Erreur : ${error.message || error}`;
      message.sources = [];
      message.error = true;
      finishAssistantMessage(article, message);
      conv.updatedAt = Date.now();
      saveConversations();
      renderConversations();
      toast(message.content);
    } finally {
      isStreaming = false;
      updateSendState();
      composer.focus();
    }
  }

  async function sendQuestion(rawQuestion) {
    const question = String(rawQuestion || "").trim();
    if (!question) return;
    if (isStreaming) {
      toast("Une génération est déjà en cours.");
      return;
    }

    let conv = getActiveConversation();
    if (!conv) conv = createConversation(question);
    if (!conv.messages.length) conv.title = titleFromQuestion(question);
    conv.updatedAt = Date.now();

    const userMessage = { role: "user", content: question, sources: [] };
    const assistantMessage = { role: "assistant", content: "", sources: [], error: false };
    conv.messages.push(userMessage, assistantMessage);

    appendUserMessage(question, conv.messages.length - 2);
    const article = createAssistantShell(conv.messages.length - 1);
    setTitle(conv.title);
    resetComposer();
    saveConversations();
    renderConversations();
    scrollToBottom();

    await streamAnswer(question, conv, assistantMessage, article);
  }

  function regenerateAnswer(article) {
    if (isStreaming) {
      toast("Une génération est déjà en cours.");
      return;
    }
    const conv = getActiveConversation();
    const index = Number(article.dataset.messageIndex);
    if (!conv || !Number.isInteger(index)) return;

    const message = conv.messages[index];
    const previous = conv.messages[index - 1];
    if (!message || message.role !== "assistant" || !previous || previous.role !== "user") {
      toast("Question d'origine introuvable.");
      return;
    }

    message.content = "";
    message.sources = [];
    message.error = false;
    streamAnswer(previous.content, conv, message, article);
  }

  // ---------------------------------------------------------------------------
  // Actions sur les messages
  // ---------------------------------------------------------------------------

  messagesEl.addEventListener("click", (event) => {
    const sourceCard = event.target.closest(".source-card");
    if (sourceCard && sourceCard._source) {
      openSource(sourceCard._source);
      return;
    }

    const action = event.target.closest("[data-action]");
    if (!action) return;
    const article = action.closest(".message--assistant");
    if (!article) return;
    const conv = getActiveConversation();
    const index = Number(article.dataset.messageIndex);
    const message = conv?.messages[index];

    if (action.dataset.action === "copy") {
      const text = $(".message__content", article).innerText;
      navigator.clipboard?.writeText(text).then(
        () => toast("Réponse copiée dans le presse-papiers."),
        () => toast("Copie impossible dans ce contexte.")
      );
    } else if (action.dataset.action === "regenerate") {
      regenerateAnswer(article);
    }
  });

  function openSource(source) {
    $("#sourceTitle").textContent = source.name;
    $("#sourceMeta").textContent = `Extrait #${source.rank}`;
    $("#sourceSnippet").textContent = source.snippet;
    $("#sourceScore").textContent = `Score ${formatScore(source.score)}`;
    openModal(sourceModal);
  }

  // ---------------------------------------------------------------------------
  // Suggestions configurables
  // ---------------------------------------------------------------------------

  function normalizeSuggestionConfig(data) {
    const defaults = DEFAULT_SUGGESTION_CONFIG;
    const rawList = Array.isArray(data?.suggestions) ? data.suggestions : [];
    const suggestions = rawList
      .filter((item) => item && item.enabled !== false)
      .filter((item) => typeof item.prompt === "string" && item.prompt.trim())
      .slice(0, 8)
      .map((item) => ({
        title: String(item.title || "Question suggérée").trim(),
        description: String(item.description || item.prompt).trim(),
        prompt: String(item.prompt).trim(),
        icon: SUGGESTION_ICONS.has(String(item.icon || "")) ? String(item.icon) : "sparkles"
      }));

    const heading = String(data?.heading || "").trim();
    const subtitle = String(data?.subtitle || "").trim();
    const columns = Math.min(4, Math.max(1, Number(data?.columns) || defaults.columns));

    return {
      heading: heading || defaults.heading,
      subtitle: subtitle || defaults.subtitle,
      columns,
      suggestions: suggestions.length ? suggestions : defaults.suggestions
    };
  }

  function renderSuggestionConfig(config) {
    emptyHeading.textContent = config.heading;
    emptySubtitle.textContent = config.subtitle;
    suggestionsEl.style.setProperty("--suggestion-columns", String(config.columns));
    suggestionsEl.replaceChildren();

    config.suggestions.forEach((item) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "suggestion";
      button.innerHTML = `
        <span class="suggestion__icon"><svg class="icon"><use href="#i-${item.icon}"/></svg></span>
        <span class="suggestion__text"><strong></strong><span></span></span>`;
      $("strong", button).textContent = item.title;
      $(".suggestion__text span", button).textContent = item.description;
      button.addEventListener("click", () => sendQuestion(item.prompt));
      suggestionsEl.appendChild(button);
    });
  }

  async function loadSuggestionConfig() {
    try {
      const response = await fetch(`${SUGGESTION_FILE}?v=${Date.now()}`, { cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      renderSuggestionConfig(normalizeSuggestionConfig(await response.json()));
      suggestionsSource.textContent = "suggestions.json";
      suggestionsSource.title = "Cartes chargées depuis /static/suggestions.json";
    } catch (error) {
      renderSuggestionConfig(DEFAULT_SUGGESTION_CONFIG);
      suggestionsSource.textContent = "valeurs par défaut";
      suggestionsSource.title = "JSON inaccessible";
    }
  }

  // ---------------------------------------------------------------------------
  // Ingestion de documents
  // ---------------------------------------------------------------------------

  async function ingestFiles(files) {
    const list = Array.from(files || []).filter(Boolean);
    if (!list.length) throw new Error("Aucun fichier sélectionné.");

    const formData = new FormData();
    list.forEach((file) => formData.append("files", file));
    const response = await fetch("/api/ingest", { method: "POST", body: formData });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.detail || response.statusText || "Ingestion impossible.");
    }
    return payload;
  }

  function renderIngestResult(data) {
    const result = $("#ingestResult");
    result.replaceChildren();
    const rows = Array.isArray(data.ingested) ? data.ingested : [];
    let total = 0;

    rows.forEach((row) => {
      total += Number(row.chunks) || 0;
      const line = document.createElement("div");
      line.className = "ok-row";
      line.textContent = `${row.file} — ${row.chars} caractères → ${row.chunks} chunks`;
      result.appendChild(line);
    });

    const summary = document.createElement("div");
    summary.className = "ok-row";
    summary.innerHTML = `<b>${total} chunks insérés dans Fluree.</b>`;
    result.appendChild(summary);
  }

  function showAttachment(label, busy = false) {
    attachmentName.textContent = label;
    attachment.classList.add("visible");
    attachment.classList.toggle("is-busy", busy);
  }

  function clearAttachment() {
    fileInput.value = "";
    attachment.classList.remove("visible", "is-busy");
    attachmentName.textContent = "";
  }

  async function ingestAttachment(file) {
    if (!file) return;
    if (isStreaming) {
      toast("Attendez la fin de la génération.");
      clearAttachment();
      return;
    }

    showAttachment(`${file.name} · indexation…`, true);
    try {
      const data = await ingestFiles([file]);
      const total = (data.ingested || []).reduce((sum, row) => sum + (Number(row.chunks) || 0), 0);
      toast(`${total} chunk(s) indexé(s) depuis ${file.name}.`);
      refreshStatus();
    } catch (error) {
      toast(`Indexation impossible : ${error.message}`);
    } finally {
      clearAttachment();
      composer.focus();
    }
  }

  // ---------------------------------------------------------------------------
  // État des services et modèles
  // ---------------------------------------------------------------------------

  function withTag(name) {
    if (!name) return "";
    return name.includes(":") ? name : `${name}:latest`;
  }

  function setServiceState(ok, label) {
    $("#serviceStatus").classList.toggle("is-off", !ok);
    $("#serviceStatusText").textContent = label;
    $("#statusDot").classList.toggle("is-off", !ok);
    $("#modelDot").classList.toggle("is-off", !ok);
  }

  function renderModelMenu(data, hasChat, hasEmbed) {
    modelMenu.replaceChildren();

    const addLabel = (text) => {
      const label = document.createElement("p");
      label.className = "menu__label";
      label.textContent = text;
      modelMenu.appendChild(label);
    };
    const addRow = (icon, name, meta) => {
      const row = document.createElement("div");
      row.className = "menu-item is-static";
      row.innerHTML = `
        <svg class="icon"><use href="#i-${icon}"/></svg>
        <span class="model-menu__name"></span>
        <span class="menu-item__meta"></span>`;
      $(".model-menu__name", row).textContent = name || "—";
      $(".menu-item__meta", row).textContent = meta;
      modelMenu.appendChild(row);
    };
    const addSeparator = () => {
      const separator = document.createElement("div");
      separator.className = "menu-sep";
      modelMenu.appendChild(separator);
    };

    addLabel("Génération");
    addRow("sparkles", data.chat_model, hasChat ? "installé" : "absent");
    addLabel("Embeddings");
    addRow("database", data.embed_model, hasEmbed ? "installé" : "absent");
    addSeparator();

    const manage = document.createElement("button");
    manage.type = "button";
    manage.className = "menu-item";
    manage.innerHTML = `
      <svg class="icon"><use href="#i-settings"/></svg>
      GPU &amp; modèles`;
    manage.addEventListener("click", () => {
      closeMenus();
      openLibrary("gpu");
    });
    modelMenu.appendChild(manage);

    const refresh = document.createElement("button");
    refresh.type = "button";
    refresh.className = "menu-item";
    refresh.innerHTML = `
      <svg class="icon"><use href="#i-refresh"/></svg>
      Actualiser l'état`;
    refresh.addEventListener("click", () => {
      closeMenus();
      refreshStatus();
      toast("État des services actualisé.");
    });
    modelMenu.appendChild(refresh);
  }

  async function refreshStatus() {
    try {
      const response = await fetch("/api/status", { cache: "no-store" });
      if (!response.ok) throw new Error(response.statusText);
      const data = await response.json();
      statusCache = data;

      const installed = (data.ollama_models || []).map(withTag);
      const hasChat = installed.includes(withTag(data.chat_model));
      const hasEmbed = installed.includes(withTag(data.embed_model));
      const ok = hasChat && hasEmbed;

      setServiceState(
        ok,
        ok ? "Services prêts" : hasChat || hasEmbed ? "Services partiels" : "Services indisponibles"
      );

      $("#modelName").textContent = data.chat_model || "—";
      $("#modelCardName").textContent = data.chat_model || "Modèle inconnu";
      $("#modelCardMeta").textContent = `${data.embed_model || "embedding"} · ${data.ledger || "Fluree"}`;
      $("#instanceMeta").textContent = `Fluree · ${data.ledger || "kb/main"}`;
      $("#libraryMeta").textContent = `Fluree · ${data.ledger || "kb/main"} · ${data.chat_model || "modèle inconnu"}`;
      $("#ledgerAvatar").textContent = String(data.ledger || "F").slice(0, 1).toUpperCase();
      renderModelMenu(data, hasChat, hasEmbed);
    } catch (error) {
      setServiceState(false, "Hors ligne");
      $("#modelName").textContent = "hors ligne";
      $("#modelCardName").textContent = "Services injoignables";
      $("#modelCardMeta").textContent = "Vérifiez Fluree et Ollama";
      modelMenu.replaceChildren();
      const row = document.createElement("div");
      row.className = "menu-item is-static";
      row.textContent = "Impossible de joindre les services.";
      modelMenu.appendChild(row);
    }
  }

  async function refreshGpu() {
    const gpuStatus = $("#gpuStatus");
    const modelsStatus = $("#modelsStatus");
    gpuStatus.textContent = "Chargement…";
    modelsStatus.textContent = "Chargement…";

    try {
      const response = await fetch("/api/gpu", { cache: "no-store" });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || response.statusText);
      }
      const data = await response.json();
      gpuStatus.replaceChildren();

      if (!Array.isArray(data.models) || !data.models.length) {
        gpuStatus.textContent = "Aucun modèle chargé pour l'instant (ils se chargent à la première question).";
      } else {
        data.models.forEach((model) => {
          const row = document.createElement("div");
          row.className = model.size_vram ? "ok-row" : "err-row";
          const vram = model.size_vram ? `${(model.size_vram / 1e9).toFixed(2)} Go VRAM` : "mémoire inconnue";
          const gpu = model.gpus ? ` · GPU ${model.gpus}` : " · CPU";
          row.textContent = `${model.name} · ${vram}${gpu}`;
          gpuStatus.appendChild(row);
        });
      }
    } catch (error) {
      gpuStatus.replaceChildren();
      const row = document.createElement("div");
      row.className = "err-row";
      row.textContent = `Ollama injoignable : ${error.message}`;
      gpuStatus.appendChild(row);
    }

    try {
      const data = statusCache || await (await fetch("/api/status", { cache: "no-store" })).json();
      modelsStatus.replaceChildren();
      const lines = [
        `Embedding : ${data.embed_model || "—"}`,
        `Génération : ${data.chat_model || "—"}`,
        `Disponibles : ${(data.ollama_models || []).join(", ") || "aucun"}`
      ];
      lines.forEach((text) => {
        const line = document.createElement("div");
        line.textContent = text;
        modelsStatus.appendChild(line);
      });
    } catch (error) {
      modelsStatus.textContent = "État des modèles indisponible.";
    }
  }

  // ---------------------------------------------------------------------------
  // Modales, menus et navigation
  // ---------------------------------------------------------------------------

  function openModal(modal) {
    modal.hidden = false;
    body.classList.add("modal-open");
  }

  function closeModal(modal) {
    modal.hidden = true;
    if (!$$(".modal").some((entry) => !entry.hidden)) {
      body.classList.remove("modal-open");
    }
  }

  function openLibrary(tab = "docs") {
    closeSidebar();
    openModal(libraryModal);
    switchModalTab(tab);
  }

  function switchModalTab(name) {
    $$(".modal-tab").forEach((tab) => {
      tab.classList.toggle("active", tab.dataset.tab === name);
    });
    $("#modalDocs").hidden = name !== "docs";
    $("#modalGpu").hidden = name !== "gpu";
    if (name === "gpu") refreshGpu();
  }

  function closeMenus() {
    modelMenu.classList.remove("open");
    modelButton.setAttribute("aria-expanded", "false");
    moreMenu.classList.remove("open");
    moreButton.setAttribute("aria-expanded", "false");
  }

  function toggleMenu(button, menu) {
    const open = !menu.classList.contains("open");
    closeMenus();
    menu.classList.toggle("open", open);
    button.setAttribute("aria-expanded", String(open));
  }

  function closeSidebar() {
    body.classList.remove("sidebar-open");
  }

  function toggleSidebar() {
    if (window.matchMedia("(max-width: 900px)").matches) {
      body.classList.toggle("sidebar-open");
    } else {
      body.classList.toggle("sidebar-closed");
    }
  }

  function exportConversation() {
    const conv = getActiveConversation();
    if (!conv || !conv.messages.length) {
      toast("Aucune conversation à exporter.");
      return;
    }

    const blocks = conv.messages.map((message) => {
      if (message.role === "user") return `## Question\n\n${message.content}`;
      const sources = (message.sources || [])
        .map((source, index) => `${index + 1}. ${source.source} — extrait #${source.index} — score ${formatScore(source.score)}`)
        .join("\n");
      return `## Réponse\n\n${message.content}${sources ? `\n\n### Sources\n\n${sources}` : ""}`;
    });

    const markdown = `# ${conv.title}\n\n${blocks.join("\n\n")}\n`;
    const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${conv.title.replace(/[^a-z0-9à-ÿ]+/gi, "-").replace(/^-|-$/g, "").toLowerCase() || "conversation"}.md`;
    link.click();
    URL.revokeObjectURL(url);
    closeMenus();
    toast("Conversation exportée.");
  }

  // ---------------------------------------------------------------------------
  // Composer
  // ---------------------------------------------------------------------------

  function autoGrow() {
    composer.style.height = "auto";
    composer.style.height = `${Math.min(composer.scrollHeight, 200)}px`;
  }

  function updateSendState() {
    sendButton.disabled = isStreaming || !composer.value.trim();
  }

  function resetComposer() {
    composer.value = "";
    autoGrow();
    updateSendState();
    clearAttachment();
  }

  // ---------------------------------------------------------------------------
  // Thème
  // ---------------------------------------------------------------------------

  const themeToggle = $("#themeToggle");
  const themeIcon = $("#themeIcon");
  const root = document.documentElement;
  const savedTheme = window.localStorage.getItem(THEME_KEY);
  if (savedTheme === "dark" || savedTheme === "light") {
    root.dataset.theme = savedTheme;
  }
  const syncThemeIcon = () => {
    themeIcon.setAttribute("href", root.dataset.theme === "dark" ? "#i-sun" : "#i-moon");
  };
  syncThemeIcon();
  themeToggle.addEventListener("click", () => {
    root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
    window.localStorage.setItem(THEME_KEY, root.dataset.theme);
    syncThemeIcon();
  });

  // ---------------------------------------------------------------------------
  // Événements
  // ---------------------------------------------------------------------------

  $("#newChat").addEventListener("click", startNewConversation);
  $("#newChatMenu").addEventListener("click", () => {
    closeMenus();
    startNewConversation();
  });
  $("#openSidebar").addEventListener("click", toggleSidebar);
  $("#closeSidebar").addEventListener("click", () => {
    if (window.matchMedia("(max-width: 900px)").matches) closeSidebar();
    else body.classList.toggle("sidebar-closed");
  });
  $("#overlay").addEventListener("click", closeSidebar);

  sendButton.addEventListener("click", () => sendQuestion(composer.value));
  composer.addEventListener("input", () => {
    autoGrow();
    updateSendState();
  });
  composer.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendQuestion(composer.value);
    }
  });

  $("#searchInput").addEventListener("input", renderConversations);

  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      $("#searchInput").focus();
    }
    if (event.key === "Escape") {
      closeMenus();
      closeSidebar();
      closeModal(libraryModal);
      closeModal(sourceModal);
    }
  });

  modelButton.addEventListener("click", (event) => {
    event.stopPropagation();
    toggleMenu(modelButton, modelMenu);
  });
  moreButton.addEventListener("click", (event) => {
    event.stopPropagation();
    toggleMenu(moreButton, moreMenu);
  });
  document.addEventListener("click", (event) => {
    if (!event.target.closest("#modelPicker")) {
      modelMenu.classList.remove("open");
      modelButton.setAttribute("aria-expanded", "false");
    }
    if (!event.target.closest("#morePicker")) {
      moreMenu.classList.remove("open");
      moreButton.setAttribute("aria-expanded", "false");
    }
  });

  $("#exportConversation").addEventListener("click", exportConversation);
  $("#openDocsMenu").addEventListener("click", () => {
    closeMenus();
    openLibrary("docs");
  });
  $("#openModelsMenu").addEventListener("click", () => {
    closeMenus();
    openLibrary("gpu");
  });

  $("#openIngest").addEventListener("click", () => openLibrary("docs"));
  $("#openLibraryChip").addEventListener("click", () => openLibrary("docs"));
  $("#openModels").addEventListener("click", (event) => {
    event.stopPropagation();
    openLibrary("gpu");
  });
  $("#modelCard").addEventListener("click", () => openLibrary("gpu"));
  $("#modelCard").addEventListener("keydown", (event) => {
    if (event.target !== event.currentTarget) return;
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openLibrary("gpu");
    }
  });
  $("#openInstance").addEventListener("click", () => openLibrary("gpu"));

  $$("#libraryModal [data-close-library], #closeLibrary").forEach((button) => {
    button.addEventListener("click", () => closeModal(libraryModal));
  });
  $$("#sourceModal [data-close-source], #closeSourceModal").forEach((button) => {
    button.addEventListener("click", () => closeModal(sourceModal));
  });
  $$(".modal-tab").forEach((tab) => {
    tab.addEventListener("click", () => switchModalTab(tab.dataset.tab));
  });
  $("#refreshGpu").addEventListener("click", refreshGpu);

  $("#attachButton").addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => ingestAttachment(fileInput.files[0]));
  $("#removeAttachment").addEventListener("click", clearAttachment);

  $("#uploadForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const files = $("#libraryFiles").files;
    if (!files.length) {
      toast("Sélectionnez au moins un fichier.");
      return;
    }

    const button = $("#uploadBtn");
    const result = $("#ingestResult");
    button.disabled = true;
    result.textContent = "Indexation en cours…";

    try {
      const data = await ingestFiles(files);
      renderIngestResult(data);
      const total = (data.ingested || []).reduce((sum, row) => sum + (Number(row.chunks) || 0), 0);
      toast(`${total} chunk(s) indexé(s).`);
      $("#libraryFiles").value = "";
      refreshStatus();
    } catch (error) {
      result.replaceChildren();
      const row = document.createElement("div");
      row.className = "err-row";
      row.textContent = `Erreur : ${error.message}`;
      result.appendChild(row);
    } finally {
      button.disabled = false;
    }
  });

  // ---------------------------------------------------------------------------
  // Démarrage
  // ---------------------------------------------------------------------------

  renderSuggestionConfig(DEFAULT_SUGGESTION_CONFIG);
  loadSuggestionConfig();
  renderConversations();
  refreshStatus();
  autoGrow();
  updateSendState();
  setTimeout(() => composer.focus(), 120);
})();
