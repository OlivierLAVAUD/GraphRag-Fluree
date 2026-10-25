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
  const uploadTitle = $("#uploadTitle");
  const uploadHint = $("#uploadHint");
  const uploadProgress = $("#uploadProgress");
  const uploadFill = $("#uploadFill");
  const uploadProgressLabel = $("#uploadProgressLabel");
  const libraryFiles = $("#libraryFiles");
  const libraryDir = $("#libraryDir");
  const uploadBtn = $("#uploadBtn");

  const kbModal = $("#kbModal");
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
  let librarySelection = [];
  let ingestRunning = false;

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

  let sourcesSeq = 0;

  // Bloc de sources d'une réponse : replié par défaut, la barre d'en-tête
  // (un vrai bouton, donc accessible au clavier) le déplie et le replie.
  function buildSources(sources) {
    const wrap = document.createElement("div");
    wrap.className = "sources is-collapsed";
    const threshold = formatScore(statusCache?.min_score ?? 0.3);
    const listId = `sourceList-${++sourcesSeq}`;
    const count = sources.length > 1 ? "extraits" : "extrait";

    wrap.innerHTML = `
      <button class="sources__head" type="button" aria-expanded="false" aria-controls="${listId}">
        <span class="sources__label">
          <svg class="icon icon--sm sources__chevron"><use href="#i-chevron-right"/></svg>
          <span>Sources</span>
        </span>
        <span class="sources__meta">${sources.length} ${count} · seuil ${threshold}</span>
      </button>
      <div class="source-list" id="${listId}"></div>`;

    const list = $(".source-list", wrap);
    const head = $(".sources__head", wrap);
    const chevron = $(".sources__chevron use", wrap);
    head.addEventListener("click", () => {
      const collapsed = !wrap.classList.contains("is-collapsed");
      wrap.classList.toggle("is-collapsed", collapsed);
      head.setAttribute("aria-expanded", String(!collapsed));
      chevron.setAttribute("href", collapsed ? "#i-chevron-right" : "#i-chevron-down");
    });
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
        body: JSON.stringify({
          question,
          source: $("#chatSourceFilter")?.value || null,
          entity: $("#chatEntityFilter")?.value || null,
          top_k: Number($("#chatTopK")?.value) || undefined
        })
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

  const SUPPORTED_EXTENSIONS = new Set([".md", ".markdown", ".txt", ".pdf"]);
  const FALLBACK_LIMITS = {
    max_upload_files: 20,
    max_total_upload_size: 200 * 1024 * 1024
  };

  function uploadLimits() {
    return { ...FALLBACK_LIMITS, ...(statusCache?.limits || {}) };
  }

  function basename(path) {
    return String(path || "").split("/").filter(Boolean).pop() || "";
  }

  function formatBytes(bytes) {
    const value = Number(bytes) || 0;
    if (value < 1024) return `${value} o`;
    if (value < 1024 * 1024) return `${Math.round(value / 1024)} Ko`;
    return `${(value / (1024 * 1024)).toFixed(1).replace(".", ",")} Mo`;
  }

  function isSupportedDocument(name) {
    const dot = String(name || "").lastIndexOf(".");
    return dot >= 0 && SUPPORTED_EXTENSIONS.has(String(name).slice(dot).toLowerCase());
  }

  function selectionItems(fileList) {
    return Array.from(fileList || [])
      .filter((file) => file && typeof file.name === "string")
      .map((file) => ({
        file,
        // `webkitRelativePath` porte le chemin dans le dossier choisi ; il est
        // vide pour une sélection de fichiers, d'où le repli sur le nom.
        path: (file.webkitRelativePath || file.name).replace(/^\/+/, "")
      }));
  }

  function selectedDirectory(items) {
    const path = items.find((item) => item.path.includes("/"))?.path || "";
    return path.split("/")[0] || "";
  }

  function previewSelection(items) {
    if (!items.length) {
      uploadTitle.textContent = "Sélectionner des documents ou un dossier";
      uploadHint.textContent =
        "Markdown, texte ou PDF · sélection multiple, ou dossier complet";
      return;
    }

    const totalBytes = items.reduce((sum, item) => sum + (item.file.size || 0), 0);
    const directory = selectedDirectory(items);
    if (directory) {
      uploadTitle.textContent = directory;
      uploadHint.textContent = `Dossier · ${items.length} fichier(s) · ${formatBytes(totalBytes)}`;
      return;
    }
    uploadTitle.textContent =
      items.length === 1 ? items[0].path : `${items.length} fichiers sélectionnés`;
    uploadHint.textContent = items.length === 1
      ? `Document · ${formatBytes(totalBytes)}`
      : `${items.slice(0, 3).map((item) => basename(item.path)).join(" · ")}${
          items.length > 3 ? " · …" : ""
        } · ${formatBytes(totalBytes)}`;
  }

  function setLibrarySelection(items) {
    librarySelection = items;
    previewSelection(items);
  }

  function resetIngestProgress() {
    uploadProgress.hidden = true;
    uploadProgress.classList.remove("is-running");
    uploadFill.style.width = "0%";
    uploadProgressLabel.textContent = "";
  }

  function setIngestProgress({ index = 0, total = 1, item = null, fraction = 0 } = {}) {
    const ratio = total ? Math.min(1, (index + (fraction || 0)) / total) : 0;
    const done = Math.min(index, total);
    uploadProgress.hidden = false;
    uploadProgress.classList.add("is-running");
    uploadFill.style.width = `${Math.round(ratio * 100)}%`;
    uploadProgressLabel.textContent =
      `${done < total ? done + 1 : total}/${total} fichier(s) · ${Math.round(ratio * 100)} %`;
    // Le nom du document en cours remplace l'invite de sélection.
    if (item) uploadTitle.textContent = basename(item.path);
  }

  function finishIngestProgress(total, failures) {
    uploadProgress.hidden = false;
    uploadProgress.classList.remove("is-running");
    uploadFill.style.width = failures ? uploadFill.style.width : "100%";
    uploadProgressLabel.textContent = failures
      ? `Terminé · ${total - failures}/${total} fichier(s) indexé(s), ${failures} en échec`
      : `Terminé · ${total} fichier(s) indexé(s)`;
  }

  function postIngestFile(item, onUpload) {
    return new Promise((resolve, reject) => {
      const formData = new FormData();
      // 3e argument : nom envoyé au serveur, chemin relatif compris, pour que
      // deux fichiers homonymes de dossiers différents restent distincts.
      formData.append("files", item.file, item.path);
      const request = new XMLHttpRequest();
      request.open("POST", "/api/ingest");
      request.upload.addEventListener("progress", (event) => {
        if (event.lengthComputable) onUpload(event.loaded / event.total);
      });
      request.addEventListener("load", () => {
        let payload = {};
        try {
          payload = JSON.parse(request.responseText);
        } catch (error) {
          payload = {};
        }
        if (request.status >= 200 && request.status < 300) {
          resolve(payload);
        } else {
          reject(new Error(payload.detail || `HTTP ${request.status}`));
        }
      });
      request.addEventListener("error", () => reject(new Error("Backend injoignable.")));
      request.send(formData);
    });
  }

  async function ingestFiles(files, { onProgress } = {}) {
    const items = (Array.isArray(files) ? files : Array.from(files || []))
      .filter(Boolean)
      .map((entry) =>
        entry instanceof File
          ? { file: entry, path: entry.webkitRelativePath || entry.name }
          : entry
      );
    if (!items.length) throw new Error("Aucun fichier sélectionné.");

    const inserted = [];
    const failures = [];
    // Un fichier par requête : la progression reflète l'ingestion réelle
    // (embeddings + écriture Fluree) et un fichier en échec n'annule pas les
    // autres. Les limites du serveur sont vérifiées avant l'envoi.
    for (let index = 0; index < items.length; index += 1) {
      const item = items[index];
      try {
        const payload = await postIngestFile(item, (fraction) =>
          onProgress?.({ index, total: items.length, item, fraction, inserted, failures })
        );
        if (Array.isArray(payload.ingested)) inserted.push(...payload.ingested);
      } catch (error) {
        failures.push({ file: item.path, message: error.message || String(error) });
      }
      onProgress?.({
        index: index + 1,
        total: items.length,
        item,
        fraction: 1,
        inserted,
        failures
      });
    }
    return { ingested: inserted, errors: failures };
  }

  function renderIngestResult(data) {
    const result = $("#ingestResult");
    result.replaceChildren();
    const rows = Array.isArray(data.ingested) ? data.ingested : [];
    const failures = Array.isArray(data.errors) ? data.errors : [];
    let total = 0;

    rows.forEach((row) => {
      total += Number(row.chunks) || 0;
      const line = document.createElement("div");
      line.className = "ok-row";
      line.textContent = `${row.file} — ${row.chars} caractères → ${row.chunks} chunks`;
      result.appendChild(line);
    });

    failures.forEach((failure) => {
      const line = document.createElement("div");
      line.className = "err-row";
      line.textContent = `${failure.file} — échec : ${failure.message}`;
      result.appendChild(line);
    });

    const summary = document.createElement("div");
    summary.className = failures.length ? "err-row" : "ok-row";
    summary.innerHTML = `<b>${total} chunks insérés dans Fluree.</b>${
      failures.length ? ` ${failures.length} fichier(s) en échec.` : ""
    }`;
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
      const failure = (data.errors || [])[0];
      if (failure) throw new Error(failure.message);
      toast(`${total} chunk(s) indexé(s) depuis ${file.name}.`);
      refreshStatus();
      refreshSourceList();
      refreshEntityList();
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
      openKB("gpu");
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
      $("#kbMeta").textContent = `Fluree · ${data.ledger || "kb/main"} · ${data.chat_model || "modèle inconnu"}`;
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
  // Refonte Base de connaissances
  // ---------------------------------------------------------------------------

  async function fetchSources() {
    try {
      const response = await fetch("/api/sources", { cache: "no-store" });
      if (!response.ok) throw new Error(response.statusText);
      return (await response.json()).sources || [];
    } catch (error) {
      return [];
    }
  }

  async function fetchEntities() {
    try {
      const response = await fetch("/api/entities", { cache: "no-store" });
      if (!response.ok) throw new Error(response.statusText);
      return (await response.json()).entities || [];
    } catch (error) {
      return [];
    }
  }

  async function refreshSourceList() {
    const sources = await fetchSources();
    const list = $("#sourceList");
    if (!list) return;

    list.replaceChildren();
    if (!sources.length) {
      const item = document.createElement("li");
      item.textContent = "Aucune source indexée.";
      list.appendChild(item);
    } else {
      sources.forEach((source) => {
        const item = document.createElement("li");

        const label = document.createElement("span");
        label.className = "source-list-kb__label";
        const name = document.createElement("span");
        name.textContent = source.name;
        name.title = source.name;
        const chunks = document.createElement("span");
        chunks.className = "chunks";
        chunks.textContent = `${source.chunks} chunks`;
        label.append(name, chunks);

        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "icon-btn icon-btn--sm source-list-kb__delete";
        remove.title = `Désindexer ${source.name}`;
        remove.setAttribute("aria-label", `Désindexer ${source.name}`);
        remove.innerHTML = '<svg class="icon icon--sm"><use href="#i-close"/></svg>';
        remove.addEventListener("click", () => deindexSource(source.name));

        item.append(label, remove);
        list.appendChild(item);
      });
    }

    ["#kbSourceFilter", "#chatSourceFilter"].forEach((selector) => {
      const select = $(selector);
      if (!select) return;
      const current = select.value;
      select.querySelectorAll("option:not(:first-child)").forEach((option) => option.remove());
      sources.forEach((source) => {
        const option = document.createElement("option");
        option.value = source.name;
        option.textContent = source.name;
        select.appendChild(option);
      });
      select.value = current;
    });
  }

  async function deindexSource(name) {
    if (
      !window.confirm(
        `Désindexer « ${name} » ?\n\nSes chunks ne seront plus utilisés pour répondre. ` +
          "Le document n'est pas effacé du ledger : une nouvelle indexation le réactive."
      )
    ) {
      return;
    }

    try {
      const response = await fetch(`/api/sources?source=${encodeURIComponent(name)}`, {
        method: "DELETE",
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.detail || response.statusText || "Désindexation impossible.");
      }
      toast(`« ${name} » désindexée (${payload.deindexed} chunks).`);
      await Promise.all([refreshSourceList(), refreshEntityList()]);
      refreshStatus();
    } catch (error) {
      toast(`Désindexation impossible : ${error.message}`);
    }
  }

  async function resetKnowledgeBase() {
    const confirmed = window.confirm(
      "Réinitialiser la base de connaissances ?\n\nTous les documents seront désindexés. " +
        "Le ledger Fluree et son historique sont conservés."
    );
    if (!confirmed) return;

    const button = $("#resetKb");
    if (button) button.disabled = true;
    try {
      const response = await fetch("/api/reset", { method: "POST" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.detail || response.statusText || "Réinitialisation impossible.");
      }
      toast(`Base réinitialisée (${payload.deindexed} chunks désindexés).`);
      const result = $("#ingestResult");
      if (result) result.replaceChildren();
      await Promise.all([refreshSourceList(), refreshEntityList()]);
      refreshStatus();
    } catch (error) {
      toast(`Réinitialisation impossible : ${error.message}`);
    } finally {
      if (button) button.disabled = false;
    }
  }

  async function refreshEntityList() {
    const entities = await fetchEntities();
    const grid = $("#kbEntitiesList");
    if (grid) {
      grid.replaceChildren();
      entities.forEach((entity) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "entity-chip";
        button.innerHTML = `<strong></strong><span></span>`;
        $("strong", button).textContent = entity.name;
        $("span", button).textContent = `${entity.type || "entité"} · ${entity.chunks} chunks`;
        button.addEventListener("click", () => {
          const select = $("#kbEntityFilter");
          if (select) select.value = entity.name;
          switchKbTab("search");
          const input = $("#kbSearchInput");
          if (input) input.value = `Entité : ${entity.name}`;
          kbSearch(entity.name, entity.name);
        });
        grid.appendChild(button);
      });
      if (!entities.length) grid.textContent = "Aucune entité détectée.";
    }

    ["#kbEntityFilter", "#chatEntityFilter"].forEach((selector) => {
      const select = $(selector);
      if (!select) return;
      const current = select.value;
      select.querySelectorAll("option:not(:first-child)").forEach((option) => option.remove());
      entities.forEach((entity) => {
        const option = document.createElement("option");
        option.value = entity.name;
        option.textContent = entity.name;
        select.appendChild(option);
      });
      select.value = current;
    });
  }

  async function refreshHistory() {
    const list = $("#kbHistoryList");
    if (!list) return;
    list.textContent = "Chargement…";

    try {
      const response = await fetch("/api/history", { cache: "no-store" });
      if (!response.ok) throw new Error(response.statusText);
      const data = await response.json();
      const commits = data.history || [];
      list.replaceChildren();

      if (!commits.length) {
        list.textContent = "Aucun commit.";
        return;
      }

      commits.forEach((commit) => {
        const item = document.createElement("li");
        if (typeof commit === "string") {
          const code = document.createElement("code");
          code.textContent = commit;
          item.appendChild(code);
          list.appendChild(item);
          return;
        }

        const time = commit.time || commit.timestamp;
        const parsed = time ? new Date(time) : null;
        const date =
          parsed && !Number.isNaN(parsed.valueOf())
            ? parsed.toLocaleString("fr-FR")
            : "commit";

        const code = document.createElement("code");
        code.textContent = commit.commit_id
          ? String(commit.commit_id).slice(0, 12)
          : `t=${commit.t ?? "?"}`;
        item.append(document.createTextNode(`${date} — `), code);

        const details = [];
        if (commit.t !== undefined) details.push(`t=${commit.t}`);
        if (commit.asserts !== undefined) details.push(`${commit.asserts} asserts`);
        if (commit.retracts) details.push(`${commit.retracts} retracts`);
        if (details.length) {
          const detail = document.createElement("span");
          detail.className = "commit-detail";
          detail.textContent = details.join(" · ");
          item.appendChild(detail);
        }

        list.appendChild(item);
      });
    } catch (error) {
      list.textContent = `Erreur : ${error.message}`;
    }
  }

  async function kbSearch(presetEntity, presetQuestion = null) {
    const question = (presetQuestion || $("#kbSearchInput")?.value || "").trim();
    if (!question) {
      toast("Saisissez une requête.");
      return;
    }

    const body = {
      question,
      top_k: Number($("#kbTopK")?.value) || 7,
      source: $("#kbSourceFilter")?.value || null,
      entity: presetEntity || $("#kbEntityFilter")?.value || null
    };
    const minScore = Number($("#kbMinScore")?.value);
    const output = $("#kbSearchResults");
    output.textContent = "Recherche…";

    try {
      const response = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || response.statusText);

      const results = (data.results || []).filter((hit) =>
        !Number.isFinite(minScore) || Number(hit.score) >= minScore
      );
      output.replaceChildren();

      results.forEach((hit) => {
        const card = document.createElement("div");
        card.className = "kb-result";
        card.innerHTML = `
          <div class="kb-result__head">
            <strong></strong>
            <span class="score"></span>
          </div>
          <div class="kb-result__snippet"></div>
          <div class="kb-result__actions">
            <button class="btn" data-act="tochat">Envoyer au chat</button>
          </div>`;
        $("strong", card).textContent = `${hit.source} · extrait #${hit.index}`;
        $(".score", card).textContent = formatScore(hit.score);
        $(".kb-result__snippet", card).textContent = (hit.content || "").slice(0, 240);
        $("[data-act='tochat']", card).addEventListener("click", () => {
          $("#composer").value = body.question;
          if ($("#chatSourceFilter")) $("#chatSourceFilter").value = body.source || "";
          if ($("#chatEntityFilter")) $("#chatEntityFilter").value = body.entity || "";
          if ($("#chatTopK")) {
            $("#chatTopK").value = String(body.top_k);
            const out = $("#chatTopK").nextElementSibling;
            if (out) out.textContent = String(body.top_k);
          }
          autoGrow();
          updateSendState();
          closeModal(kbModal);
          composer.focus();
        });
        output.appendChild(card);
      });

      if (!output.children.length) output.textContent = "Aucun résultat au-dessus du seuil.";
    } catch (error) {
      output.textContent = `Erreur : ${error.message}`;
    }
  }

  function switchKbTab(name) {
    $$(".modal-tab").forEach((tab) => {
      tab.classList.toggle("active", tab.dataset.kbtab === name);
    });
    const views = {
      docs: "#kbDocs",
      search: "#kbSearch",
      entities: "#kbEntities",
      history: "#kbHistory",
      gpu: "#kbGpu"
    };
    Object.entries(views).forEach(([key, selector]) => {
      const view = $(selector);
      if (view) view.hidden = key !== name;
    });

    if (name === "docs") refreshSourceList();
    if (name === "entities") refreshEntityList();
    if (name === "history") refreshHistory();
    if (name === "gpu") refreshGpu();
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

  function openKB(tab = "docs") {
    closeSidebar();
    closeMenus();
    openModal(kbModal);
    switchKbTab(tab);
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
      closeModal(kbModal);
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
    openKB("docs");
  });
  $("#openModelsMenu").addEventListener("click", () => {
    closeMenus();
    openKB("gpu");
  });

  $("#openIngest").addEventListener("click", () => openKB("docs"));
  $("#openLibraryChip").addEventListener("click", () => openKB("docs"));
  $("#openModels").addEventListener("click", (event) => {
    event.stopPropagation();
    openKB("gpu");
  });
  $("#modelCard").addEventListener("click", () => openKB("gpu"));
  $("#modelCard").addEventListener("keydown", (event) => {
    if (event.target !== event.currentTarget) return;
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openKB("gpu");
    }
  });
  $("#openInstance").addEventListener("click", () => openKB("gpu"));

  $$("#kbModal [data-close-kb], #closeKb").forEach((button) => {
    button.addEventListener("click", () => closeModal(kbModal));
  });
  $$("#sourceModal [data-close-source], #closeSourceModal").forEach((button) => {
    button.addEventListener("click", () => closeModal(sourceModal));
  });
  $$(".modal-tab").forEach((tab) => {
    tab.addEventListener("click", () => switchKbTab(tab.dataset.kbtab));
  });
  $("#resetKb")?.addEventListener("click", resetKnowledgeBase);
  $("#refreshGpu").addEventListener("click", refreshGpu);
  $("#kbRefreshHistory").addEventListener("click", refreshHistory);
  $("#kbSearchBtn").addEventListener("click", () => kbSearch());
  $("#kbSearchInput").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      kbSearch();
    }
  });

  ["#chatTopK", "#kbTopK", "#kbMinScore"].forEach((selector) => {
    const input = $(selector);
    if (!input) return;
    const output = input.nextElementSibling;
    const sync = () => {
      if (!output) return;
      output.textContent = selector === "#kbMinScore"
        ? Number(input.value).toFixed(2)
        : String(input.value);
    };
    input.addEventListener("input", sync);
    sync();
  });

  $("#attachButton").addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => ingestAttachment(fileInput.files[0]));
  $("#removeAttachment").addEventListener("click", clearAttachment);

  function selectFrom(fileList) {
    const all = selectionItems(fileList);
    const items = all.filter((item) => isSupportedDocument(item.file.name));
    const ignored = all.length - items.length;
    const limits = uploadLimits();

    const kept = items.slice(0, limits.max_upload_files);
    resetIngestProgress();
    setLibrarySelection(kept);

    // Un seul toast : `toast()` ne garde que le dernier message.
    const notices = [];
    if (ignored) {
      notices.push(`${ignored} fichier(s) ignoré(s) (formats : .md, .markdown, .txt, .pdf)`);
    }
    if (items.length > limits.max_upload_files) {
      notices.push(`limité aux ${limits.max_upload_files} premiers fichiers`);
    }
    const totalBytes = kept.reduce((sum, item) => sum + (item.file.size || 0), 0);
    if (totalBytes > limits.max_total_upload_size) {
      notices.push(`dépasse la limite de ${formatBytes(limits.max_total_upload_size)}`);
    }
    if (notices.length) toast(`Sélection ajustée : ${notices.join(" · ")}.`);
  }

  libraryFiles.addEventListener("change", () => {
    libraryDir.value = "";
    selectFrom(libraryFiles.files);
  });
  libraryDir.addEventListener("change", () => {
    libraryFiles.value = "";
    selectFrom(libraryDir.files);
  });
  $("#pickDirBtn").addEventListener("click", () => libraryDir.click());

  $("#uploadForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (ingestRunning) return;

    const items = librarySelection;
    if (!items.length) {
      toast("Sélectionnez au moins un fichier ou un dossier.");
      return;
    }

    const limits = uploadLimits();
    const totalBytes = items.reduce((sum, item) => sum + (item.file.size || 0), 0);
    if (totalBytes > limits.max_total_upload_size) {
      toast(
        `Sélection trop volumineuse : ${formatBytes(totalBytes)} ` +
          `(limite ${formatBytes(limits.max_total_upload_size)}).`
      );
      return;
    }

    const result = $("#ingestResult");
    ingestRunning = true;
    uploadBtn.disabled = true;
    result.replaceChildren();
    setIngestProgress({ index: 0, total: items.length, item: items[0], fraction: 0 });

    try {
      const data = await ingestFiles(items, {
        onProgress: ({ index, total, item, fraction }) => {
          setIngestProgress({
            index,
            total,
            item: index < total ? item : null,
            fraction
          });
        }
      });
      const failures = (data.errors || []).length;
      const total = (data.ingested || []).reduce((sum, row) => sum + (Number(row.chunks) || 0), 0);
      renderIngestResult(data);
      finishIngestProgress(items.length, failures);

      if (failures) {
        toast(`${total} chunk(s) indexé(s), ${failures} fichier(s) en échec.`);
        previewSelection(items);
      } else {
        toast(`${total} chunk(s) indexé(s) depuis ${items.length} fichier(s).`);
        setLibrarySelection([]);
        libraryFiles.value = "";
        libraryDir.value = "";
      }
      refreshStatus();
      refreshSourceList();
      refreshEntityList();
    } catch (error) {
      result.replaceChildren();
      const row = document.createElement("div");
      row.className = "err-row";
      row.textContent = `Erreur : ${error.message}`;
      result.appendChild(row);
      previewSelection(items);
      resetIngestProgress();
    } finally {
      ingestRunning = false;
      uploadBtn.disabled = false;
    }
  });

  // ---------------------------------------------------------------------------
  // Démarrage
  // ---------------------------------------------------------------------------

  renderSuggestionConfig(DEFAULT_SUGGESTION_CONFIG);
  loadSuggestionConfig();
  renderConversations();
  refreshStatus();
  refreshSourceList();
  refreshEntityList();
  autoGrow();
  updateSendState();
  setTimeout(() => composer.focus(), 120);
})();
