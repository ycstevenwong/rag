(() => {
  const messagesEl = document.getElementById("messages");
  const formEl = document.getElementById("chat-form");
  const inputEl = document.getElementById("message-input");
  const sendBtn = document.getElementById("send-btn");
  const sessionIdEl = document.getElementById("session-id");
  const newSessionBtn = document.getElementById("new-session");
  const docListEl = document.getElementById("doc-list");
  const uploadForm = document.getElementById("upload-form");
  const fileInput = document.getElementById("file-input");
  const uploadStatus = document.getElementById("upload-status");
  const uploadProgress = document.getElementById("upload-progress");
  const sourcesListEl = document.getElementById("sources-list");
  const rewrittenQueryEl = document.getElementById("rewritten-query");
  const uploadSourceTypeEl = document.getElementById("upload-source-type");
  const uploadAppCodeEl = document.getElementById("upload-app-code");
  const uploadTagsEl = document.getElementById("upload-tags");
  const filterSourceTypeEl = document.getElementById("filter-source-type");
  const filterAppCodeEl = document.getElementById("filter-app-code");
  const filterTagsEl = document.getElementById("filter-tags");

  let sessionId = localStorage.getItem("card-rag.session") || newSessionId();
  let currentSources = [];
  sessionIdEl.textContent = sessionId.slice(0, 8) + "…";
  localStorage.setItem("card-rag.session", sessionId);

  function newSessionId() {
    return crypto.randomUUID().replace(/-/g, "");
  }

  newSessionBtn.addEventListener("click", () => {
    sessionId = newSessionId();
    localStorage.setItem("card-rag.session", sessionId);
    sessionIdEl.textContent = sessionId.slice(0, 8) + "…";
    messagesEl.innerHTML = "";
    sourcesListEl.innerHTML = '<p class="empty">Cited sources will appear here.</p>';
    rewrittenQueryEl.hidden = true;
    rewrittenQueryEl.innerHTML = "";
    currentSources = [];
  });

  // ---------- documents ----------
  async function refreshDocs() {
    const r = await fetch("/docs");
    const docs = await r.json();
    docListEl.innerHTML = "";
    if (!docs.length) {
      const li = document.createElement("li");
      li.innerHTML = '<span class="name" style="color:var(--muted)">No documents yet</span>';
      docListEl.appendChild(li);
      return;
    }
    for (const d of docs) {
      const li = document.createElement("li");
      const typeBadge = d.source_type && d.source_type !== "other"
        ? `<span class="type-badge">${escapeHtml(d.source_type)}</span>` : "";
      li.innerHTML = `<span class="name" title="${escapeHtml(d.filename)}">${escapeHtml(d.filename)}</span>
                     ${typeBadge}
                     <span class="count">${d.n_chunks}</span>
                     <button data-id="${d.doc_id}" title="Delete">✕</button>`;
      li.querySelector("button").addEventListener("click", async () => {
        if (!confirm(`Delete "${d.filename}"?`)) return;
        await fetch(`/docs/${d.doc_id}`, { method: "DELETE" });
        refreshDocs();
      });
      docListEl.appendChild(li);
    }
  }
  refreshDocs();

  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    fd.append("source_type", uploadSourceTypeEl.value || "other");
    fd.append("app_code", uploadAppCodeEl.value || "");
    fd.append("tags", uploadTagsEl.value || "");
    uploadStatus.classList.remove("error");
    uploadStatus.textContent = `Uploading ${file.name}…`;
    uploadProgress.hidden = false;
    uploadProgress.value = 0;

    const xhr = new XMLHttpRequest();
    let parsedIdx = 0;
    let buffer = "";
    let finalResult = null;
    let finalError = null;

    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) uploadProgress.value = (e.loaded / e.total) * 100;
    });
    xhr.upload.addEventListener("load", () => {
      uploadProgress.removeAttribute("value");
      uploadStatus.textContent = "Indexing…";
    });

    xhr.addEventListener("progress", () => {
      const newChunk = xhr.responseText.slice(parsedIdx);
      parsedIdx = xhr.responseText.length;
      buffer += newChunk;
      let idx;
      while ((idx = buffer.indexOf("\n\n")) >= 0) {
        const raw = buffer.slice(0, idx).trim();
        buffer = buffer.slice(idx + 2);
        if (!raw.startsWith("data:")) continue;
        const payload = raw.slice(5).trim();
        if (!payload) continue;
        let evt;
        try { evt = JSON.parse(payload); } catch { continue; }
        if (evt.type === "stage") {
          uploadStatus.textContent = `${evt.stage}…`;
          if (evt.stage === "Embedding") uploadProgress.value = 0;
          else uploadProgress.removeAttribute("value");
        } else if (evt.type === "progress") {
          uploadProgress.value = (evt.done / evt.total) * 100;
          uploadStatus.textContent = `Embedding batch ${evt.done}/${evt.total}…`;
        } else if (evt.type === "done") {
          finalResult = evt.result;
        } else if (evt.type === "error") {
          finalError = new Error(evt.error);
        }
      }
    });

    xhr.addEventListener("load", () => {
      if (finalError) {
        uploadStatus.classList.add("error");
        uploadStatus.textContent = finalError.message;
      } else if (finalResult) {
        uploadStatus.textContent = finalResult.duplicate
          ? "Already indexed."
          : finalResult.linked
            ? `Linked to existing content (${finalResult.n_chunks} chunks).`
            : `Indexed ${finalResult.n_chunks} chunks.`;
        uploadProgress.value = 100;
        uploadForm.reset();
        refreshDocs();
      } else {
        uploadStatus.classList.add("error");
        uploadStatus.textContent = "Ingest ended without result.";
      }
      setTimeout(() => { uploadProgress.hidden = true; }, 800);
    });

    xhr.addEventListener("error", () => {
      uploadStatus.classList.add("error");
      uploadStatus.textContent = "Network error";
      setTimeout(() => { uploadProgress.hidden = true; }, 800);
    });

    xhr.open("POST", "/upload");
    xhr.send(fd);
  });

  // ---------- chat ----------
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      formEl.requestSubmit();
    }
  });

  formEl.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = inputEl.value.trim();
    if (!text) return;
    inputEl.value = "";
    sendBtn.disabled = true;

    appendMessage("user", text);
    const assistantEl = appendMessage("assistant", "", { streaming: true });

    let buffer = "";
    let assembled = "";

    try {
      const filterTags = (filterTagsEl.value || "")
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean);
      const filters = {};
      if (filterSourceTypeEl.value) filters.source_type = filterSourceTypeEl.value;
      if (filterAppCodeEl.value) filters.app_code = filterAppCodeEl.value;
      if (filterTags.length) filters.tags = filterTags;

      const r = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: text, filters }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const reader = r.body.getReader();
      const decoder = new TextDecoder();
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let idx;
        while ((idx = buffer.indexOf("\n\n")) >= 0) {
          const raw = buffer.slice(0, idx).trim();
          buffer = buffer.slice(idx + 2);
          if (!raw.startsWith("data:")) continue;
          const payload = raw.slice(5).trim();
          if (!payload) continue;
          let evt;
          try { evt = JSON.parse(payload); } catch { continue; }
          handleEvent(evt, assistantEl, (delta) => {
            assembled += delta;
            assistantEl.innerHTML = renderMarkdown(assembled);
            attachCitationHandlers(assistantEl);
            messagesEl.scrollTop = messagesEl.scrollHeight;
          });
        }
      }
    } catch (err) {
      assistantEl.classList.remove("cursor");
      assistantEl.innerHTML = `<span style="color:var(--danger)">${escapeHtml(err.message)}</span>`;
    } finally {
      assistantEl.classList.remove("cursor");
      sendBtn.disabled = false;
      inputEl.focus();
    }
  });

  function handleEvent(evt, assistantEl, onDelta) {
    if (evt.type === "session") {
      sessionId = evt.session_id;
      localStorage.setItem("card-rag.session", sessionId);
      sessionIdEl.textContent = sessionId.slice(0, 8) + "…";
    } else if (evt.type === "sources") {
      currentSources = evt.sources || [];
      renderRewrittenQuery(evt.rewritten_query);
      renderSources(currentSources);
    } else if (evt.type === "delta") {
      onDelta(evt.text);
    } else if (evt.type === "error") {
      assistantEl.classList.remove("cursor");
      assistantEl.innerHTML += `\n<span style="color:var(--danger)">${escapeHtml(evt.error)}</span>`;
    } else if (evt.type === "done") {
      assistantEl.classList.remove("cursor");
    }
  }

  function appendMessage(role, text, { streaming = false } = {}) {
    const el = document.createElement("div");
    el.className = `message ${role}` + (streaming ? " cursor" : "");
    el.innerHTML = renderMarkdown(text);
    messagesEl.appendChild(el);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return el;
  }

  function renderMarkdown(text) {
    let html = escapeHtml(text);
    html = html.replace(/```([\s\S]*?)```/g, (_, c) => `<pre><code>${c.trim()}</code></pre>`);
    html = html.replace(/`([^`\n]+)`/g, "<code>$1</code>");
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\[(\d+(?:,\s*\d+)*)\]/g, (match, group) => {
      const nums = group.split(/,\s*/);
      return nums.map((n) =>
        `<span class="citation" data-n="${n}">${n}</span>`
      ).join("");
    });
    html = html.replace(/\n/g, "<br>");
    return html;
  }

  function attachCitationHandlers(root) {
    root.querySelectorAll(".citation").forEach((el) => {
      el.onclick = () => {
        const n = parseInt(el.dataset.n, 10);
        const card = document.querySelector(`.source-card[data-n="${n}"]`);
        if (card) {
          document.querySelectorAll(".source-card").forEach((c) => c.classList.remove("highlight"));
          card.classList.add("highlight");
          card.scrollIntoView({ behavior: "smooth", block: "center" });
        }
      };
    });
  }

  function renderRewrittenQuery(q) {
    if (!q) {
      rewrittenQueryEl.hidden = true;
      rewrittenQueryEl.innerHTML = "";
      return;
    }
    rewrittenQueryEl.hidden = false;
    rewrittenQueryEl.innerHTML = `<span class="label">Searched for</span>${escapeHtml(q)}`;
  }

  function renderSources(sources) {
    sourcesListEl.classList.remove("flash");
    void sourcesListEl.offsetWidth;
    sourcesListEl.classList.add("flash");
    sourcesListEl.innerHTML = "";
    if (!sources.length) {
      sourcesListEl.innerHTML = '<p class="empty">No sources retrieved.</p>';
      return;
    }
    for (const s of sources) {
      const card = document.createElement("div");
      card.className = "source-card";
      card.dataset.n = s.n;
      card.innerHTML = `
        <div><span class="n">${s.n}</span><strong>${escapeHtml(s.filename)}</strong></div>
        <div class="locator">${escapeHtml(s.locator)}</div>
        <div class="snippet">${escapeHtml(s.snippet)}</div>
      `;
      sourcesListEl.appendChild(card);
    }
  }

  function escapeHtml(s) {
    return (s || "").replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }[c]));
  }
})();
