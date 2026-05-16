const API_URL = "http://127.0.0.1:5000/generate";
const HEALTH_URL = "http://127.0.0.1:5000/health";

const chatListEl = document.querySelector("#chat-list");
const messagesEl = document.querySelector("#messages");
const composerEl = document.querySelector("#composer");
const promptInputEl = document.querySelector("#prompt-input");
const sendButtonEl = document.querySelector("#send-button");
const newChatButtonEl = document.querySelector("#new-chat-button");
const chatTitleEl = document.querySelector("#chat-title");
const statusEl = document.querySelector("#api-status");
const temperatureInputEl = document.querySelector("#temperature-input");
const tokensInputEl = document.querySelector("#tokens-input");
const highlightToggleEl = document.querySelector("#highlight-toggle");


let chats = [];
let activeChatId = null;
let isGenerating = false;
let activeTypingTimer = null;
let highlightPromptEnabled = false;
let currentInputType = null; // Armazena o tipo da entrada atual

function createChat() {
  const chat = {
    id: createId(),
    title: "Novo chat",
    messages: [],
    createdAt: new Date().toISOString(),
  };

  chats.unshift(chat);
  activeChatId = chat.id;
  render();
  promptInputEl.focus();
}

function createId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }

  return `chat-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function getActiveChat() {
  return chats.find((chat) => chat.id === activeChatId);
}

function updateChatTitle(chat, prompt) {
  if (chat.title !== "Novo chat") {
    return;
  }

  const clean = prompt.replace(/\s+/g, " ").trim();
  chat.title = clean.length > 34 ? `${clean.slice(0, 34)}...` : clean || "Novo chat";
}

function escapeHtml(value) {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function looksLikeCodeRequest(prompt) {
  return /\b(c[oó]digo|funcao|fun[cç][aã]o|classe|python|script|programa|implemente|crie|def |class )\b/i.test(prompt);
}

function looksLikePythonCode(text) {
  return /(^|\n)\s*(def|class|for|while|if|elif|else|try|except|return|import|from|with)\b/.test(text)
    || /(^|\n)\s*[a-zA-Z_]\w*\s*=/.test(text)
    || text.includes("self.");
}

function formatPythonLikeCode(text) {
  const trimmed = text.trim();
  if (trimmed.includes("\n")) {
    return trimmed;
  }

  return trimmed
    .replace(/\s+(class\s+[A-Za-z_]\w*)/g, "\n$1")
    .replace(/\s+(def\s+[A-Za-z_]\w*)/g, "\n$1")
    .replace(/\s+(elif\b|else:|except\b|finally:)/g, "\n$1")
    .replace(/\s+(if\b|for\b|while\b|try:|with\b)/g, "\n    $1")
    .replace(/\s+(return\b|import\b|from\b|pass\b|break\b|continue\b|raise\b)/g, "\n    $1")
    .replace(/:\s+(?=(self\.|[A-Za-z_]\w*\s*=|return\b|if\b|for\b|while\b))/g, ":\n    ")
    .replace(/;\s*/g, "\n    ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function ensureCodeBlockWhenNeeded(prompt, response) {
  const hasFence = /```/.test(response);
  if (hasFence || (!looksLikeCodeRequest(prompt) && !looksLikePythonCode(response))) {
    return response;
  }

  return `\`\`\`py\n${formatPythonLikeCode(response)}\n\`\`\``;
}

function classifyInput(prompt) {
  const trimmed = prompt.trim().toLowerCase();
  
  // Verifica se é uma pergunta
  if (trimmed.includes("?") || trimmed.startsWith("<pergunta>")) {
    return "pergunta";
  }
  
  // Verifica se é código Python
  if (/\bdef\s+\w+\s*\(|^class\s+\w+|^\s*(import|from)\s+\w+|\breturn\b|\bif\s+\w+|for\s+\w+\s+in/.test(prompt)) {
    return "codigo";
  }
  
  // Se não é pergunta nem código, pode ser história
  if (prompt.length > 50) {
    return "historia";
  }
  
  return "pergunta"; // padrão
}

function extractResponseSection(prompt, modelOutput, inputType) {
  // Usa o tipo passado como parâmetro (classificação do input)
  const type = inputType || classifyInput(prompt);
  let sectionContent = "";
  
  // Remove quebras de linha extras e normaliza espaços
  let cleaned = modelOutput.replace(/\s+/g, " ").trim();
  
  if (type === "pergunta") {
    // Procura por <RESPOSTA> ou <resposta>
    const respMatch = cleaned.match(/<[Rr][Ee][Ss][Pp][Oo][Ss][Tt][Aa]>\s*([^<]*?)\s*<[Ee][Oo][Ss]>/i);
    if (respMatch) {
      sectionContent = respMatch[1].trim();
    }
  } 
  else if (type === "codigo") {
    // Procura por padrão de código ou seção "codigo:"
    const codeMatch = cleaned.match(/(?:<codigo>|codigo\s*:)\s*([^<]*?)(?:<[Ee][Oo][Ss]>|$)/i);
    if (codeMatch) {
      sectionContent = codeMatch[1].trim();
    } else {
      // Tenta extrair o primeiro bloco de código
      const pyMatch = cleaned.match(/^([^<]*?(?:def|class)\s+\w+[^<]*?)(?:<[Ee][Oo][Ss]>|<|$)/i);
      if (pyMatch) {
        sectionContent = pyMatch[1].trim();
      }
    }
  } 
  else if (type === "historia") {
    // Procura por "historia:" ou <HISTORIA>
    const histMatch = cleaned.match(/(?:<[Hh][Ii][Ss][Tt][Oo][Rr][Ii][Aa]>|historia\s*:)\s*([^<]*?)(?:<[Ee][Oo][Ss]>|<|$)/i);
    if (histMatch) {
      sectionContent = histMatch[1].trim();
    } else if (!/</.test(cleaned)) {
      // Se não tem tags, assume que é tudo história
      sectionContent = cleaned.replace(/<[Ee][Oo][Ss]>/g, "").trim();
    }
  }
  
  // Se não conseguiu extrair, retorna a saída limpa sem tags
  if (!sectionContent) {
    sectionContent = cleaned
      .replace(/<[^>]+>/g, "") // remove todas as tags
      .replace(/\s+/g, " ")
      .trim();
  }
  
  return {
    content: sectionContent,
    type: type,
    raw: modelOutput
  };
}

function highlightPromptInResponse(prompt, response) {
  if (!highlightPromptEnabled) {
    return response;
  }

  // Encontra o prefixo do prompt na resposta
  const promptLength = prompt.length;
  
  // Verifica se a resposta começa com o prompt
  if (response.startsWith(prompt)) {
    const highlightedPrompt = `<span class="prompt-highlight">${escapeHtml(prompt)}</span>`;
    const rest = escapeHtml(response.substring(promptLength));
    return highlightedPrompt + rest;
  }

  return escapeHtml(response);
}

function highlightPython(escapedCode) {
  const protectedParts = [];
  const protect = (html) => {
    const key = `@@TOKEN_${protectedParts.length}@@`;
    protectedParts.push(html);
    return key;
  };

  const highlighted = escapedCode
    .replace(/(&quot;.*?&quot;|&#039;.*?&#039;)/g, (match) => protect(`<span class="token-string">${match}</span>`))
    .replace(/(#.*)$/gm, (match) => protect(`<span class="token-comment">${match}</span>`))
    .replace(/\b(class|def|return|if|elif|else|for|while|in|import|from|as|try|except|finally|with|pass|break|continue|self|None|True|False)\b/g, '<span class="token-keyword">$1</span>')
    .replace(/\b(\d+(?:\.\d+)?)\b/g, '<span class="token-number">$1</span>')
    .replace(/\b([a-zA-Z_]\w*)(?=\()/g, '<span class="token-function">$1</span>');

  return highlighted.replace(/@@TOKEN_(\d+)@@/g, (_, index) => protectedParts[Number(index)]);
}

function renderInlineMarkdown(text) {
  return escapeHtml(text)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>");
}

function renderMarkdown(markdown) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let inCode = false;
  let codeLines = [];
  let codeLanguage = "";
  let listType = null;

  function closeList() {
    if (listType) {
      html.push(`</${listType}>`);
      listType = null;
    }
  }

  function closeCode() {
    if (inCode) {
      const languageLabel = codeLanguage || "text";
      const escapedCode = escapeHtml(codeLines.join("\n"));
      const highlightedCode = ["py", "python"].includes(languageLabel.toLowerCase())
        ? highlightPython(escapedCode)
        : escapedCode;

      html.push(`
        <div class="code-card">
          <div class="code-card-header">
            <span>${escapeHtml(languageLabel)}</span>
          </div>
          <pre><code>${highlightedCode}</code></pre>
        </div>
      `);
      codeLines = [];
      codeLanguage = "";
      inCode = false;
    }
  }

  for (const line of lines) {
    if (line.trim().startsWith("```")) {
      if (inCode) {
        closeCode();
      } else {
        closeList();
        inCode = true;
        codeLanguage = line.trim().replace(/^```/, "").trim() || "text";
        codeLines = [];
      }
      continue;
    }

    if (inCode) {
      codeLines.push(line);
      continue;
    }

    const trimmed = line.trim();
    if (!trimmed) {
      closeList();
      continue;
    }

    const unordered = trimmed.match(/^[-*]\s+(.+)$/);
    const ordered = trimmed.match(/^\d+\.\s+(.+)$/);

    if (unordered) {
      if (listType !== "ul") {
        closeList();
        listType = "ul";
        html.push("<ul>");
      }
      html.push(`<li>${renderInlineMarkdown(unordered[1])}</li>`);
      continue;
    }

    if (ordered) {
      if (listType !== "ol") {
        closeList();
        listType = "ol";
        html.push("<ol>");
      }
      html.push(`<li>${renderInlineMarkdown(ordered[1])}</li>`);
      continue;
    }

    closeList();
    html.push(`<p>${renderInlineMarkdown(trimmed)}</p>`);
  }

  closeCode();
  closeList();
  return html.join("");
}

function messageTemplate(message) {
  const roleLabel = message.role === "user" ? "Eu" : message.role === "error" ? "!" : "AI";
  const visibleContent = message.visibleContent ?? message.content;
  
  let content;
  if (message.loading) {
    content = '<div class="typing" aria-label="Gerando resposta"><span></span><span></span><span></span></div>';
  } else if (message.role === "assistant" && highlightPromptEnabled && message.originalPrompt) {
    // Para mensagens do assistente com destaque ativado
    const highlighted = highlightPromptInResponse(message.originalPrompt, visibleContent);
    content = `<div class="markdown">${highlighted}</div>`;
  } else {
    content = `<div class="markdown">${renderMarkdown(visibleContent)}</div>`;
  }

  // Adiciona badge do tipo de entrada para mensagens do usuário
  let inputTypeBadge = "";
  if (message.role === "user" && message.inputType) {
    const typeBadges = {
      pergunta: "❓",
      codigo: "💻",
      historia: "📖"
    };
    const badge = typeBadges[message.inputType] || "•";
    inputTypeBadge = `<div class="input-type-badge" title="Tipo: ${message.inputType}">${badge}</div>`;
  }

  return `
    <article class="message ${message.role}">
      <div class="avatar">${roleLabel}${inputTypeBadge}</div>
      <div class="bubble">${content}</div>
    </article>
  `;
}

function renderEmptyState() {
  messagesEl.innerHTML = `
    <div class="empty-state">
      <h3>Converse com seu modelo local</h3>
      <p>Digite um trecho de Python, uma pergunta ou o inicio de uma classe para gerar a continuacao pela API Flask.</p>
      <div class="quick-prompts">
        <button type="button" data-prompt="def soma(a, b):">def soma(a, b):</button>
        <button type="button" data-prompt="class Carro:">class Carro:</button>
        <button type="button" data-prompt="Explique listas em Python com exemplo.">Explicar listas</button>
        <button type="button" data-prompt="def calcular_media(valores):">def calcular_media(valores):</button>
      </div>
    </div>
  `;

  messagesEl.querySelectorAll("[data-prompt]").forEach((button) => {
    button.addEventListener("click", () => {
      promptInputEl.value = button.dataset.prompt;
      promptInputEl.focus();
    });
  });
}

function renderMessages(chat) {
  if (!chat || chat.messages.length === 0) {
    renderEmptyState();
    return;
  }

  messagesEl.innerHTML = chat.messages.map(messageTemplate).join("");
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function renderChatList() {
  chatListEl.innerHTML = chats
    .map((chat) => {
      const active = chat.id === activeChatId ? "active" : "";
      return `<button class="chat-item ${active}" type="button" data-chat-id="${chat.id}">${escapeHtml(chat.title)}</button>`;
    })
    .join("");

  chatListEl.querySelectorAll("[data-chat-id]").forEach((button) => {
    button.addEventListener("click", () => {
      activeChatId = button.dataset.chatId;
      render();
    });
  });
}

function render() {
  const chat = getActiveChat();
  chatTitleEl.textContent = chat?.title || "Novo chat";
  sendButtonEl.disabled = isGenerating;
  sendButtonEl.textContent = isGenerating ? "Gerando..." : "Enviar";
  renderChatList();
  renderMessages(chat);
}

async function checkApiHealth() {
  try {
    const response = await fetch(HEALTH_URL);
    if (!response.ok) {
      throw new Error("API indisponivel");
    }

    const data = await response.json();
    statusEl.textContent = `API online (${data.device})`;
  } catch (error) {
    statusEl.textContent = "API offline";
  }
}

async function sendPrompt(prompt) {
  const chat = getActiveChat();
  if (!chat || isGenerating) {
    return;
  }

  const temperature = Number(temperatureInputEl.value || 1);
  const maxToken = Number(tokensInputEl.value || 100);

  // Classifica o prompt do usuário
  currentInputType = classifyInput(prompt);

  updateChatTitle(chat, prompt);
  chat.messages.push({ role: "user", content: prompt, inputType: currentInputType });
  chat.messages.push({ role: "assistant", content: "", loading: true, originalPrompt: prompt, inputType: currentInputType });
  isGenerating = true;
  render();

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        prompt,
        temperature,
        max_token: maxToken,
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Erro ao gerar resposta.");
    }

    const loadingIndex = chat.messages.findIndex((message) => message.loading);
    if (loadingIndex >= 0) {
      // Limpa a resposta do modelo baseado na classificação do input
      const cleanedResponse = extractResponseSection(prompt, data.response || "", currentInputType);
      const finalContent = ensureCodeBlockWhenNeeded(prompt, cleanedResponse.content);
      
      chat.messages[loadingIndex].loading = false;
      chat.messages[loadingIndex].content = finalContent;
      chat.messages[loadingIndex].visibleContent = "";
      chat.messages[loadingIndex].originalPrompt = prompt;
      chat.messages[loadingIndex].responseType = cleanedResponse.type;
      chat.messages[loadingIndex].inputType = currentInputType;
      animateAssistantMessage(chat.id, loadingIndex, finalContent);
    }
    statusEl.textContent = `API online (${data.device})`;
  } catch (error) {
    const loadingIndex = chat.messages.findIndex((message) => message.loading);
    if (loadingIndex >= 0) {
      chat.messages.splice(loadingIndex, 1);
    }

    chat.messages.push({
      role: "error",
      content: `Erro ao chamar a API: ${error.message}`,
    });
    statusEl.textContent = "API com erro";
  } finally {
    isGenerating = false;
    render();
  }
}

function animateAssistantMessage(chatId, messageIndex, finalContent) {
  if (activeTypingTimer) {
    clearInterval(activeTypingTimer);
  }

  let cursor = 0;
  const charsPerTick = finalContent.length > 700 ? 6 : 3;
  const tickMs = 18;

  activeTypingTimer = setInterval(() => {
    const chat = chats.find((item) => item.id === chatId);
    const message = chat?.messages[messageIndex];

    if (!message) {
      clearInterval(activeTypingTimer);
      activeTypingTimer = null;
      return;
    }

    cursor = Math.min(finalContent.length, cursor + charsPerTick);
    message.visibleContent = finalContent.slice(0, cursor);

    if (chat.id === activeChatId) {
      render();
    }

    if (cursor >= finalContent.length) {
      message.visibleContent = finalContent;
      clearInterval(activeTypingTimer);
      activeTypingTimer = null;
      render();
    }
  }, tickMs);
}

composerEl.addEventListener("submit", (event) => {
  event.preventDefault();
  const prompt = promptInputEl.value.trim();
  if (!prompt) {
    return;
  }

  promptInputEl.value = "";
  sendPrompt(prompt);
});

promptInputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    composerEl.requestSubmit();
  }
});

newChatButtonEl.addEventListener("click", createChat);

highlightToggleEl.addEventListener("click", () => {
  highlightPromptEnabled = !highlightPromptEnabled;
  highlightToggleEl.classList.toggle("active");
  render();
});

createChat();
checkApiHealth();