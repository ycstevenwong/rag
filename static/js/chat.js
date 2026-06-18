(() => {
  const messagesEl = document.getElementById("messages");
  const formEl = document.getElementById("chat-form");
  const inputEl = document.getElementById("message-input");
  const sendBtn = document.getElementById("send-btn");
  const sessionIdEl = document.getElementById("session-id");
  const newSessionBtn = document.getElementById("new-session");
  const docListEl = document.getElementById("doc-list");
  const managedSummaryEl = document.getElementById("managed-summary");
  const uploadForm = document.getElementById("upload-form");
  const fileInput = document.getElementById("file-input");
  const uploadStatus = document.getElementById("upload-status");
  const uploadProgress = document.getElementById("upload-progress");
  const sourcesListEl = document.getElementById("sources-list");
  const rewrittenQueryEl = document.getElementById("rewritten-query");
  const adminToggleEl = document.getElementById("admin-toggle");
  const adminModalEl = document.getElementById("admin-modal");
  const adminLoginFormEl = document.getElementById("admin-login-form");
  const adminUsernameEl = document.getElementById("admin-username");
  const adminPasswordEl = document.getElementById("admin-password");
  const adminCancelEl = document.getElementById("admin-cancel");
  const adminErrorEl = document.getElementById("admin-error");
  const uploadAppCodeEl = document.getElementById("upload-app-code");
  const uploadSourceTypeEl = document.getElementById("upload-source-type");
  const uploadVersionEl = document.getElementById("upload-version");
  const uploadFunctionalityEl = document.getElementById("upload-functionality");
  const uploadRequesterEl = document.getElementById("upload-requester");
  const anonymousOnlyFieldsEl = document.getElementById("anonymous-only-fields");
  const uploadHintUserEl = document.getElementById("upload-hint-user");
  const uploadHintAdminEl = document.getElementById("upload-hint-admin");
  const fileDropEl = document.getElementById("file-drop");
  const fileDropLabelEl = document.getElementById("file-drop-label");
  const pendingSectionEl = document.getElementById("pending-section");
  const pendingListEl = document.getElementById("pending-list");
  const openDocsModalBtn = document.getElementById("open-docs-modal");
  const docsModalEl = document.getElementById("docs-modal");
  const docsModalCloseBtn = document.getElementById("docs-modal-close");
  const docsSearchEl = document.getElementById("docs-search");
  const docsFilterTypeEl = document.getElementById("docs-filter-type");
  const docsFilterAppEl = document.getElementById("docs-filter-app");
  const docsCountEl = document.getElementById("docs-count");
  const docsTbodyEl = document.getElementById("docs-tbody");
  const uploadTagsEl = document.getElementById("upload-tags");
  const filterSourceTypeEl = document.getElementById("filter-source-type");
  const filterAppCodeEl = document.getElementById("filter-app-code");
  const filterTagsEl = document.getElementById("filter-tags");

  let sessionId = localStorage.getItem("card-rag.session") || newSessionId();
  let currentSources = [];
  let isAdmin = false;
  let allDocs = [];

  function setAdminButton(enabled, admin) {
    if (!enabled) {
      adminToggleEl.hidden = true;
      isAdmin = false;
      anonymousOnlyFieldsEl.hidden = false;
      uploadHintUserEl.hidden = false;
      uploadHintAdminEl.hidden = true;
      return;
    }
    adminToggleEl.hidden = false;
    isAdmin = !!admin;
    adminToggleEl.textContent = admin ? "🔒 admin" : "🔓 Login";
    adminToggleEl.classList.toggle("is-admin", !!admin);
    anonymousOnlyFieldsEl.hidden = !!admin;
    uploadHintUserEl.hidden = !!admin;
    uploadHintAdminEl.hidden = !admin;
    if (typeof updateFileDropState === "function") updateFileDropState();
  }

  async function refreshAdminState() {
    try {
      const r = await fetch("/admin/me");
      const data = await r.json();
      setAdminButton(data.enabled, data.is_admin);
      if (data.is_admin) refreshPending();
      else pendingSectionEl.hidden = true;
    } catch {
      setAdminButton(false, false);
    }
  }

  async function refreshPending() {
    if (!isAdmin) {
      pendingSectionEl.hidden = true;
      return;
    }
    let items = [];
    try {
      const r = await fetch("/admin/pending");
      const data = await r.json();
      items = data.items || [];
    } catch {
      pendingSectionEl.hidden = true;
      return;
    }
    if (!items.length) {
      pendingSectionEl.hidden = true;
      return;
    }
    pendingSectionEl.hidden = false;
    pendingListEl.innerHTML = "";
    for (const item of items) {
      const li = document.createElement("li");
      li.className = "pending-item";
      const metaBits = [
        item.source_type ? `type:${escapeHtml(item.source_type)}` : "",
        item.app_code ? `app:${escapeHtml(item.app_code)}` : "",
        item.version ? `ver:${escapeHtml(item.version)}` : "",
        item.functionality ? `func:${escapeHtml(item.functionality)}` : "",
        item.tags && item.tags.length ? `tags:${item.tags.map(escapeHtml).join(",")}` : "",
      ].filter(Boolean).join(" · ");
      const requesterLine = item.requester
        ? `<div class="pending-requester">requested by <strong>${escapeHtml(item.requester)}</strong></div>` : "";
      const sourceTypeOptions = ["other", "manual", "spec"].map(v =>
        `<option value="${v}"${item.source_type === v ? " selected" : ""}>${v[0].toUpperCase() + v.slice(1)}</option>`
      ).join("");
      const appCodeOptions = Array.from(uploadAppCodeEl.options).map(opt =>
        `<option value="${escapeHtml(opt.value)}"${item.app_code === opt.value ? " selected" : ""}>${escapeHtml(opt.text)}</option>`
      ).join("");
      li.innerHTML = `
        <div class="pending-head">
          <span class="name" title="${escapeHtml(item.filename)}">${escapeHtml(item.filename)}</span>
          <button data-action="reject" data-id="${item.pending_id}" class="reject" title="Reject">✕</button>
        </div>
        ${requesterLine}
        ${metaBits ? `<div class="pending-meta">${metaBits}</div>` : ""}
        <div class="pending-approve" data-id="${item.pending_id}" hidden>
          <div class="meta-row">
            <select data-field="source_type">${sourceTypeOptions}</select>
            <select data-field="app_code">${appCodeOptions}</select>
          </div>
          <div class="meta-row">
            <input data-field="version" type="text" placeholder="version" value="${escapeHtml(item.version || "")}" />
            <input data-field="functionality" type="text" placeholder="functionality" value="${escapeHtml(item.functionality || "")}" />
          </div>
          <div class="meta-row">
            <input data-field="tags" type="text" placeholder="tags" value="${escapeHtml((item.tags || []).join(","))}" />
          </div>
          <div class="pending-approve-actions">
            <button type="button" data-action="cancel-approve">Cancel</button>
            <button type="button" data-action="confirm-approve" class="primary">Approve &amp; ingest</button>
          </div>
        </div>
        <div class="pending-status status" hidden></div>
        <button class="approve" data-action="approve" data-id="${item.pending_id}">Approve…</button>
      `;
      pendingListEl.appendChild(li);
    }
  }

  pendingListEl.addEventListener("click", async (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    const action = btn.dataset.action;
    const li = btn.closest(".pending-item");
    if (!li) return;
    const pendingId = btn.dataset.id || li.querySelector("[data-id]").dataset.id;

    if (action === "reject") {
      if (!confirm("Reject and delete this upload?")) return;
      await fetch(`/admin/pending/${pendingId}/reject`, { method: "POST" });
      refreshPending();
      return;
    }
    if (action === "approve") {
      li.querySelector(".pending-approve").hidden = false;
      btn.hidden = true;
      return;
    }
    if (action === "cancel-approve") {
      li.querySelector(".pending-approve").hidden = true;
      li.querySelector("[data-action='approve']").hidden = false;
      return;
    }
    if (action === "confirm-approve") {
      const approveBlock = li.querySelector(".pending-approve");
      const statusEl = li.querySelector(".pending-status");
      const payload = {
        source_type: approveBlock.querySelector("[data-field='source_type']").value,
        app_code: approveBlock.querySelector("[data-field='app_code']").value,
        version: approveBlock.querySelector("[data-field='version']").value,
        functionality: approveBlock.querySelector("[data-field='functionality']").value,
        tags: approveBlock.querySelector("[data-field='tags']").value,
      };
      btn.disabled = true;
      statusEl.hidden = false;
      statusEl.classList.remove("error");
      statusEl.textContent = "Ingesting…";
      try {
        const r = await fetch(`/admin/pending/${pendingId}/approve`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const reader = r.body.getReader();
        const decoder = new TextDecoder();
        let buf = "";
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          let idx;
          while ((idx = buf.indexOf("\n\n")) >= 0) {
            const raw = buf.slice(0, idx).trim();
            buf = buf.slice(idx + 2);
            if (!raw.startsWith("data:")) continue;
            try {
              const evt = JSON.parse(raw.slice(5).trim());
              if (evt.type === "stage") statusEl.textContent = `${evt.stage}…`;
              else if (evt.type === "progress") statusEl.textContent = `Embedding ${evt.done}/${evt.total}…`;
              else if (evt.type === "error") { statusEl.classList.add("error"); statusEl.textContent = evt.error; }
            } catch {}
          }
        }
        refreshPending();
        refreshDocs();
      } catch (err) {
        statusEl.classList.add("error");
        statusEl.textContent = err.message;
        btn.disabled = false;
      }
    }
  });

  function openLoginModal() {
    adminErrorEl.hidden = true;
    adminErrorEl.textContent = "";
    adminModalEl.hidden = false;
    setTimeout(() => adminUsernameEl.focus(), 0);
  }

  function closeLoginModal() {
    adminModalEl.hidden = true;
    adminLoginFormEl.reset();
  }

  adminToggleEl.addEventListener("click", async () => {
    if (isAdmin) {
      if (!confirm("Log out of admin?")) return;
      await fetch("/admin/logout", { method: "POST" });
      isAdmin = false;
      setAdminButton(true, false);
      docsModalEl.hidden = true;
      refreshDocs();
      refreshPending();
      return;
    }
    openLoginModal();
  });

  adminCancelEl.addEventListener("click", closeLoginModal);

  adminLoginFormEl.addEventListener("submit", async (e) => {
    e.preventDefault();
    adminErrorEl.hidden = true;
    try {
      const r = await fetch("/admin/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: adminUsernameEl.value,
          password: adminPasswordEl.value,
        }),
      });
      const data = await r.json();
      if (!r.ok) {
        adminErrorEl.textContent = data.error || `Error ${r.status}`;
        adminErrorEl.hidden = false;
        return;
      }
      isAdmin = !!data.is_admin;
      setAdminButton(true, isAdmin);
      closeLoginModal();
      refreshDocs();
      refreshPending();
    } catch (err) {
      adminErrorEl.textContent = err.message;
      adminErrorEl.hidden = false;
    }
  });

  refreshAdminState();

  function getMissingUploadFields() {
    const missing = [];
    if (!uploadSourceTypeEl.value) missing.push("source type");
    if (!uploadAppCodeEl.value) missing.push("app code");
    if (!uploadVersionEl.value.trim()) missing.push("version");
    if (!uploadFunctionalityEl.value.trim()) missing.push("functionality");
    if (!isAdmin && !uploadRequesterEl.value.trim()) missing.push("your name");
    return missing;
  }

  function updateFileDropState() {
    const missing = getMissingUploadFields();
    if (missing.length) {
      fileInput.disabled = true;
      fileDropEl.classList.add("disabled");
      fileDropLabelEl.textContent = `Fill in: ${missing.join(", ")}`;
    } else {
      fileInput.disabled = false;
      fileDropEl.classList.remove("disabled");
      fileDropLabelEl.textContent = "Drop file or click";
    }
  }

  [
    uploadSourceTypeEl, uploadAppCodeEl, uploadVersionEl,
    uploadFunctionalityEl, uploadRequesterEl,
  ].forEach((el) => {
    el.addEventListener("input", updateFileDropState);
    el.addEventListener("change", updateFileDropState);
  });
  updateFileDropState();
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
    const payload = await r.json();
    allDocs = payload.items || [];
    const managedCount = payload.managed_count || 0;
    const totalCount = managedCount + allDocs.filter(d => !d.managed).length;

    if (!isAdmin) {
      managedSummaryEl.hidden = totalCount <= 0;
      managedSummaryEl.textContent = totalCount > 0
        ? `${totalCount.toLocaleString()} document${totalCount === 1 ? "" : "s"} indexed`
        : "";
      openDocsModalBtn.hidden = true;
      docListEl.innerHTML = "";
      docListEl.hidden = true;
      return;
    }

    // Admin: managed-summary collapses into the manage button (count + action).
    managedSummaryEl.hidden = true;
    const adminTotal = allDocs.length;
    if (adminTotal > 0) {
      openDocsModalBtn.hidden = false;
      openDocsModalBtn.textContent = `Manage all docs (${adminTotal.toLocaleString()})`;
    } else {
      openDocsModalBtn.hidden = true;
    }

    // Inline sidebar list shows ONLY unmanaged docs (user uploads).
    const unmanaged = allDocs.filter(d => !d.managed);
    docListEl.hidden = false;
    docListEl.innerHTML = "";
    if (!unmanaged.length) {
      const li = document.createElement("li");
      li.innerHTML = '<span class="name" style="color:var(--muted)">No user uploads</span>';
      docListEl.appendChild(li);
      // Update modal if it's open.
      if (!docsModalEl.hidden) renderDocsTable();
      return;
    }
    for (const d of unmanaged) {
      const li = document.createElement("li");
      const typeBadge = d.source_type && d.source_type !== "other"
        ? `<span class="type-badge">${escapeHtml(d.source_type)}</span>` : "";
      const control = (d.managed && !isAdmin)
        ? `<span class="locked" title="Managed by ingest script — delete on disk and re-run">🔒</span>`
        : `<button data-id="${d.doc_id}" title="${d.managed ? 'Delete managed doc (admin)' : 'Delete'}">✕</button>`;
      li.innerHTML = `<span class="name" title="${escapeHtml(d.filename)}">${escapeHtml(d.filename)}</span>
                     ${typeBadge}
                     <span class="count">${d.n_chunks}</span>
                     ${control}`;
      const delBtn = li.querySelector("button");
      if (delBtn) {
        delBtn.addEventListener("click", async () => {
          if (!confirm(`Delete "${d.filename}"?`)) return;
          await fetch(`/docs/${d.doc_id}`, { method: "DELETE" });
          refreshDocs();
        });
      }
      docListEl.appendChild(li);
    }
    if (!docsModalEl.hidden) renderDocsTable();
  }
  refreshDocs();

  // ---------- docs management modal ----------
  function renderDocsTable() {
    const q = (docsSearchEl.value || "").toLowerCase();
    const typeFilter = docsFilterTypeEl.value;
    const appFilter = docsFilterAppEl.value;
    const filtered = allDocs.filter((d) => {
      if (q && !(d.filename || "").toLowerCase().includes(q)) return false;
      if (typeFilter && d.source_type !== typeFilter) return false;
      if (appFilter && d.app_code !== appFilter) return false;
      return true;
    });
    docsCountEl.textContent = `${filtered.length} of ${allDocs.length}`;
    docsTbodyEl.innerHTML = "";
    for (const d of filtered) {
      const tr = document.createElement("tr");
      tr.dataset.docId = d.doc_id;
      tr.innerHTML = `
        <td class="filename" title="${escapeHtml(d.filename || "")}">${escapeHtml(d.filename || "")}</td>
        <td>${escapeHtml(d.source_type || "-")}</td>
        <td>${escapeHtml(d.app_code || "-")}</td>
        <td>${escapeHtml(d.version || "-")}</td>
        <td>${escapeHtml(d.functionality || "-")}</td>
        <td>${d.n_chunks || 0}</td>
        <td class="actions">
          <button data-action="edit" title="Edit">✏</button>
          <button data-action="delete" title="Delete">✕</button>
        </td>
      `;
      docsTbodyEl.appendChild(tr);
    }
  }

  openDocsModalBtn.addEventListener("click", () => {
    docsModalEl.hidden = false;
    docsSearchEl.value = "";
    docsFilterTypeEl.value = "";
    docsFilterAppEl.value = "";
    renderDocsTable();
  });
  docsModalCloseBtn.addEventListener("click", () => { docsModalEl.hidden = true; });
  docsModalEl.addEventListener("click", (e) => {
    if (e.target === docsModalEl) docsModalEl.hidden = true;
  });
  docsSearchEl.addEventListener("input", renderDocsTable);
  docsFilterTypeEl.addEventListener("change", renderDocsTable);
  docsFilterAppEl.addEventListener("change", renderDocsTable);

  function buildEditRow(doc) {
    const appCodeOptions = Array.from(uploadAppCodeEl.options).map((opt) => {
      const val = opt.value;
      const text = opt.text;
      return `<option value="${escapeHtml(val)}"${doc.app_code === val ? " selected" : ""}>${escapeHtml(text)}</option>`;
    }).join("");
    const editRow = document.createElement("tr");
    editRow.className = "edit-row";
    editRow.innerHTML = `
      <td colspan="7">
        <div class="edit-form">
          <select data-field="source_type">
            <option value="other"${doc.source_type === "other" ? " selected" : ""}>Other</option>
            <option value="manual"${doc.source_type === "manual" ? " selected" : ""}>Manual</option>
            <option value="spec"${doc.source_type === "spec" ? " selected" : ""}>Spec</option>
          </select>
          <select data-field="app_code">${appCodeOptions}</select>
          <input data-field="version" type="text" placeholder="version" value="${escapeHtml(doc.version || "")}" />
          <input data-field="functionality" type="text" placeholder="functionality" value="${escapeHtml(doc.functionality || "")}" />
          <input data-field="tags" type="text" placeholder="tags" value="${escapeHtml((doc.tags || []).join(","))}" />
          <button type="button" data-action="cancel-edit">Cancel</button>
          <button type="button" data-action="save-edit" class="primary">Save</button>
        </div>
      </td>
    `;
    return editRow;
  }

  docsTbodyEl.addEventListener("click", async (e) => {
    const btn = e.target.closest("button");
    if (!btn) return;
    const action = btn.dataset.action;
    const tr = btn.closest("tr");
    if (!tr) return;

    if (action === "edit") {
      const docId = tr.dataset.docId;
      const doc = allDocs.find((d) => d.doc_id === docId);
      if (!doc) return;
      const next = tr.nextElementSibling;
      if (next && next.classList.contains("edit-row")) {
        next.remove();
        return;
      }
      tr.parentNode.insertBefore(buildEditRow(doc), tr.nextSibling);
      return;
    }
    if (action === "cancel-edit") {
      btn.closest("tr.edit-row").remove();
      return;
    }
    if (action === "save-edit") {
      const editRow = btn.closest("tr.edit-row");
      const docTr = editRow.previousElementSibling;
      const docId = docTr.dataset.docId;
      const payload = {
        source_type: editRow.querySelector("[data-field='source_type']").value,
        app_code: editRow.querySelector("[data-field='app_code']").value,
        version: editRow.querySelector("[data-field='version']").value,
        functionality: editRow.querySelector("[data-field='functionality']").value,
        tags: editRow.querySelector("[data-field='tags']").value,
      };
      btn.disabled = true;
      const r = await fetch(`/docs/${docId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (r.ok) {
        await refreshDocs();
      } else {
        btn.disabled = false;
        alert("Update failed");
      }
      return;
    }
    if (action === "delete") {
      const docId = tr.dataset.docId;
      const doc = allDocs.find((d) => d.doc_id === docId);
      if (!doc) return;
      if (!confirm(`Delete "${doc.filename}"?`)) return;
      const r = await fetch(`/docs/${docId}`, { method: "DELETE" });
      if (r.ok) await refreshDocs();
    }
  });

  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) return;

    const missing = [];
    if (!uploadSourceTypeEl.value) missing.push("source type");
    if (!uploadAppCodeEl.value) missing.push("app code");
    if (!uploadVersionEl.value.trim()) missing.push("version");
    if (!uploadFunctionalityEl.value.trim()) missing.push("functionality");
    if (!isAdmin && !uploadRequesterEl.value.trim()) missing.push("your name");
    if (missing.length) {
      uploadStatus.classList.add("error");
      uploadStatus.textContent = `Please fill in: ${missing.join(", ")}.`;
      fileInput.value = "";
      return;
    }
    uploadStatus.classList.remove("error");

    const fd = new FormData();
    fd.append("file", file);
    fd.append("app_code", uploadAppCodeEl.value);
    fd.append("tags", uploadTagsEl.value || "");
    fd.append("source_type", uploadSourceTypeEl.value);
    fd.append("version", uploadVersionEl.value.trim());
    fd.append("functionality", uploadFunctionalityEl.value.trim());
    if (!isAdmin) {
      fd.append("requester", uploadRequesterEl.value.trim());
    }
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
        } else if (evt.type === "queued") {
          finalResult = { queued: true, filename: evt.filename };
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
        if (finalResult.queued) {
          uploadStatus.textContent = `Uploaded — awaiting admin approval.`;
        } else {
          uploadStatus.textContent = finalResult.duplicate
            ? "Already indexed."
            : finalResult.linked
              ? `Linked to existing content (${finalResult.n_chunks} chunks).`
              : `Indexed ${finalResult.n_chunks} chunks.`;
        }
        uploadProgress.value = 100;
        uploadForm.reset();
        updateFileDropState();
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
