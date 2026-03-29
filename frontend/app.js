/**
 * ReceiptAI Frontend — app.js
 * NEXA/AURA Dark-Theme Assistant Logic
 */

// Use window.location.origin to handle both localhost and production (Vercel) automatically
const API = window.location.origin;

// ── State ─────────────────────────────────────────────────────────────────────
let sessionId = localStorage.getItem("receipt_ia_sid") || uuidv4();
localStorage.setItem("receipt_ia_sid", sessionId);

let selectedDocId = null;
let documents = [];

function uuidv4() {
    return "10000000-1000-4000-8000-100000000000".replace(/[018]/g, c =>
        (+c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> +c / 4).toString(16)
    );
}

// ── DOM refs ──────────────────────────────────────────────────────────────────
const docList = document.getElementById("doc-list");
const chatMessages = document.getElementById("chat-messages");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");
const uploadBtn = document.getElementById("upload-btn");
const fileInput = document.getElementById("file-input");
const chatTitle = document.getElementById("chat-title");
const statusDot = document.getElementById("status-dot");
const apiStatusText = document.getElementById("api-status-text");
const toastCont = document.getElementById("toast-container");

// ── Interaction Logic ──────────────────────────────────────────────────────────
chatInput.addEventListener("input", () => {
    sendBtn.disabled = chatInput.value.trim().length === 0;
});

// ── Toast ─────────────────────────────────────────────────────────────────────
function toast(message, type = "info") {
    const el = document.createElement("div");
    el.className = `flex items-center gap-4 px-6 py-4 bg-[#111111] border border-white/5 rounded-2xl shadow-2xl animate-slide-in mb-3 relative overflow-hidden`;

    const icon = type === "success"
        ? `<iconify-icon icon="solar:check-circle-bold" class="text-emerald-500 text-xl"></iconify-icon>`
        : type === "error"
            ? `<iconify-icon icon="solar:danger-circle-bold" class="text-red-500 text-xl"></iconify-icon>`
            : `<iconify-icon icon="solar:info-circle-bold" class="text-indigo-500 text-xl"></iconify-icon>`;

    el.innerHTML = `
    <div class="absolute left-0 top-0 bottom-0 w-1 bg-${type === 'success' ? 'emerald' : type === 'error' ? 'red' : 'indigo'}-500 shadow-[0_0_8px_${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#6366f1'}]"></div>
    ${icon}
    <div class="flex flex-col">
        <span class="text-[10px] font-black text-white/40 uppercase tracking-widest leading-none mb-1">${type.toUpperCase()}</span>
        <span class="text-[11px] font-bold text-white/80">${message}</span>
    </div>
  `;
    toastCont.appendChild(el);
    setTimeout(() => {
        el.style.opacity = "0";
        setTimeout(() => el.remove(), 400);
    }, 4000);
}

// ── Health ───────────────────────────────────────────────────────────────────
async function checkHealth() {
    try {
        const r = await fetch(`${API}/health`);
        if (r.ok) {
            statusDot.className = "w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_8px_#10b981]";
            apiStatusText.textContent = "Assistant Online";
            apiStatusText.className = "text-[10px] font-bold text-emerald-500 uppercase tracking-widest";
        } else throw new Error();
    } catch {
        statusDot.className = "w-1.5 h-1.5 rounded-full bg-red-400";
        apiStatusText.textContent = "Assistant Offline";
        apiStatusText.className = "text-[10px] font-bold text-red-500 uppercase tracking-widest";
    }
}

// ── Messages ──────────────────────────────────────────────────────────────────
function appendMessage(content, role = "assistant") {
    const wrap = document.createElement("div");
    wrap.className = `flex ${role === 'user' ? 'justify-end' : 'items-start'} gap-4 max-w-[85%] animate-slide-in ${role === 'user' ? 'ml-auto' : ''}`;

    const iconHtml = role === 'assistant'
        ? `<div class="w-9 h-9 bg-indigo-600 rounded-xl flex items-center justify-center text-white shrink-0 shadow-lg shadow-indigo-600/20">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="w-5 h-5"><path d="M12 8V4H8"></path><rect width="16" height="12" x="4" y="8" rx="2"></rect><path d="M2 14h2"></path><path d="M20 14h2"></path><path d="M15 13v2"></path><path d="M9 13v2"></path></svg>
      </div>`
        : `<div class="w-9 h-9 bg-white/5 border border-white/5 rounded-xl flex items-center justify-center text-gray-500 shrink-0">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="w-5 h-5"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
      </div>`;

    wrap.innerHTML = role === 'assistant' ? `
    ${iconHtml}
    <div class="bubble-assistant p-4 text-sm leading-relaxed">${content.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>").replace(/\n/g, "<br/>")}</div>
  ` : `
    <div class="bubble-user p-4 text-sm leading-relaxed font-bold">${content}</div>
    ${iconHtml}
  `;

    chatMessages.appendChild(wrap);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return wrap;
}

function appendThinking() {
    const wrap = appendMessage(`<div class="flex items-center gap-3 py-1 font-bold text-indigo-400 uppercase tracking-[0.2em] text-[10px]"><div class="spinner"></div> Assistant Processing...</div>`, "assistant");
    wrap.id = "thinking-msg";
    return wrap;
}

// ── Rendering Data Card ───────────────────────────────────────────────────────
function formatExtractionCard(record) {
    const ex = record.extracted || {};

    const imageSection = record.image_data 
        ? `<div class="mb-5 rounded-2xl overflow-hidden border border-white/5 cursor-zoom-in group/img relative">
             <img src="${record.image_data}" class="w-full h-auto max-h-64 object-cover transition-transform duration-500 group-hover/img:scale-105" onclick="window.open('${record.image_data}', '_blank')" />
             <div class="absolute inset-0 bg-black/40 opacity-0 group-hover/img:opacity-100 transition-opacity flex items-center justify-center">
                <iconify-icon icon="solar:magnifer-zoom-in-bold" class="text-white text-2xl"></iconify-icon>
             </div>
           </div>` 
        : '';

    return `
        <div class="space-y-6" data-doc-id="${record.doc_id}">
            <p>I've extracted the core fields from **${record.filename || 'your document'}**. Here is the report:</p>
            
            ${imageSection}

            <div class="bg-gray-950/50 border border-white/5 rounded-2xl p-5 space-y-3 font-inter">
                <div class="flex items-center justify-between pb-3 border-b border-white/[0.03]">
                    <span class="text-[9px] font-black text-gray-600 uppercase tracking-widest flex items-center gap-2">
                         <iconify-icon icon="solar:shop-2-bold" class="text-indigo-500"></iconify-icon> Vendor
                    </span>
                    <span class="text-xs font-bold text-gray-300 font-inter">${ex.vendor_name || '—'}</span>
                </div>
                <div class="flex items-center justify-between pb-3 border-b border-white/[0.03]">
                    <span class="text-[9px] font-black text-gray-600 uppercase tracking-widest flex items-center gap-2">
                         <iconify-icon icon="solar:calendar-bold" class="text-indigo-500"></iconify-icon> Date
                    </span>
                    <span class="text-xs font-bold text-gray-300 font-mono">${ex.date || '—'}</span>
                </div>
                <div class="flex items-center justify-between pb-1">
                    <span class="text-[10px] font-black text-emerald-500 uppercase tracking-[0.15em] flex items-center gap-2">
                         <iconify-icon icon="solar:banknote-bold" class="text-emerald-500 text-xs"></iconify-icon> Total Amount
                    </span>
                    <span class="text-base font-black text-emerald-500 tracking-tighter">${ex.total_amount || '—'}</span>
                </div>
            </div>
            
            <div class="flex items-center gap-2 mt-4">
                <span class="text-[8px] font-black uppercase text-indigo-400 bg-indigo-500/10 border border-indigo-500/20 px-2.5 py-1 rounded">Method: ${record.method === 'model' ? 'DistilBERT NER' : 'Regex'}</span>
                <span class="text-[8px] font-black uppercase text-emerald-500 bg-emerald-500/10 border border-emerald-500/20 px-2.5 py-1 rounded">Analysis Successful</span>
            </div>
        </div>
    `;
}

// ── App Logic ─────────────────────────────────────────────────────────────────
async function loadDocuments() {
    try {
        const r = await fetch(`${API}/documents`);
        if (!r.ok) return;
        const data = await r.json();
        documents = data.documents || [];
        renderDocList();
    } catch { /* silence */ }
}

function renderDocList() {
    // Group documents by session_id
    const sessions = {};
    documents.forEach(doc => {
        const sid = doc.session_id || 'default_session';
        if (!sessions[sid]) {
            sessions[sid] = {
                id: sid,
                docs: [],
                latest: doc.timestamp
            };
        }
        sessions[sid].docs.push(doc);
    });

    const sessionList = Object.values(sessions).sort((a, b) => new Date(b.latest) - new Date(a.latest));

    const overviewHeader = document.querySelector('.text-\\[10px\\].font-bold.text-gray-700.uppercase.tracking-widest');
    if (overviewHeader) {
        overviewHeader.textContent = `Chat History (${sessionList.length})`;
    }

    if (sessionList.length === 0) {
        docList.innerHTML = `<p class="px-6 text-[10px] text-gray-700 italic font-bold uppercase tracking-widest py-10 text-center">No active context</p>`;
        return;
    }

    docList.innerHTML = "";
    sessionList.forEach(sn => {
        const active = sn.id === sessionId;
        const item = document.createElement("div");
        item.className = `group flex items-center justify-between gap-3 px-5 py-3 rounded-lg cursor-pointer transition-all border-r-2 ${active ? 'bg-white/[0.03] text-white border-indigo-600' : 'text-gray-500 hover:text-gray-300 hover:bg-white/[0.01] border-transparent'}`;

        const label = sn.docs[0]?.filename || 'Session Analysis';
        const docCount = sn.docs.length > 1 ? `<span class="ml-1 text-[9px] opacity-40">+${sn.docs.length - 1}</span>` : '';

        item.innerHTML = `
            <div class="flex items-center gap-3 truncate flex-1">
                 <iconify-icon icon="solar:chat-line-linear" class="text-lg ${active ? 'text-indigo-500' : 'text-gray-700 group-hover:text-indigo-500 transition-colors'}"></iconify-icon>
                 <span class="truncate text-[11px] font-bold leading-none mt-0.5">${label}${docCount}</span>
            </div>
            <button class="opacity-0 group-hover:opacity-100 text-gray-700 hover:text-red-500 transition-all p-1 delete-session-btn" data-id="${sn.id}">
                <iconify-icon icon="solar:trash-bin-trash-linear" class="text-base"></iconify-icon>
            </button>
        `;

        item.addEventListener("click", (e) => {
            if (e.target.closest(".delete-session-btn")) return;
            switchSession(sn);
        });

        item.querySelector(".delete-session-btn").addEventListener("click", (e) => {
            e.stopPropagation();
            deleteSession(sn.id);
        });

        docList.appendChild(item);
    });
}

async function deleteSession(sid) {
    if (!confirm("Are you sure you want to delete this entire session and all its documents?")) return;

    try {
        const sessionDocs = documents.filter(d => (d.session_id || 'default_session') === sid);

        // Delete all docs in this session
        await Promise.all(sessionDocs.map(d => fetch(`${API}/documents/${d.doc_id}`, { method: "DELETE" })));

        toast("Session removed", "success");
        if (sessionId === sid) {
            resetChat();
        } else {
            await loadDocuments();
        }
    } catch { toast("Deletion failed", "error"); }
}

function switchSession(sn) {
    sessionId = sn.id;
    localStorage.setItem("receipt_ia_sid", sessionId);

    chatMessages.innerHTML = "";
    chatTitle.textContent = "Chat Session Reloaded";

    // Rebuild chat from session documents
    sn.docs.forEach(doc => {
        appendMessage(formatExtractionCard(doc), "assistant");
    });

    appendMessage(`**Session Summary**: I've reloaded **${sn.docs.length}** document(s) for this conversation. You can continue asking questions about any of them.`, "assistant");

    renderDocList();
}

function openConversation(record) {
    // Check if we already have this card in the chat
    const existingCard = document.querySelector(`[data-doc-id="${record.doc_id}"]`);
    if (existingCard) {
        existingCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
        existingCard.classList.add('ring-2', 'ring-indigo-500', 'ring-offset-4', 'ring-offset-black');
        setTimeout(() => existingCard.classList.remove('ring-2', 'ring-indigo-500', 'ring-offset-4', 'ring-offset-black'), 2000);
    } else {
        appendMessage(formatExtractionCard(record), "assistant");
        appendMessage(`Loaded context for **${record.filename}**. Asking about "this document" will now focus on its data.`, "assistant");
    }

    selectedDocId = record.doc_id;
    chatTitle.textContent = `${record.filename || 'Session'} Context`;
    renderDocList();
}

async function deleteDocument(docId) {
    try {
        const r = await fetch(`${API}/documents/${docId}`, { method: "DELETE" });
        if (r.ok) {
            toast("Removed analysis", "success");
            if (selectedDocId === docId) {
                selectedDocId = null;
                chatTitle.textContent = "New Conversation";
                chatMessages.innerHTML = `
                    <div class="flex items-start gap-4 max-w-3xl animate-slide-in">
                        <div class="w-9 h-9 bg-indigo-600 rounded-xl flex items-center justify-center text-white shrink-0 shadow-lg shadow-indigo-600/20">
                            <iconify-icon icon="solar:robot-bold" class="text-lg"></iconify-icon>
                        </div>
                        <div class="bubble-assistant p-4 text-sm font-medium leading-relaxed">
                            Session context cleared. Ready for your next document.
                        </div>
                    </div>
                `;
            }
            await loadDocuments();
        }
    } catch { toast("Process error", "error"); }
}

async function doUpload(file) {
    toast(`Syncing ${file.name}`, "info");
    const thinking = appendThinking();

    try {
        const form = new FormData();
        form.append("file", file);
        form.append("session_id", sessionId);
        const r = await fetch(`${API}/extract`, { method: "POST", body: form });

        thinking.remove();
        if (!r.ok) {
            const errData = await r.json().catch(() => ({ detail: "Network error" }));
            const errMsg = errData.detail || "Extraction Failed";
            toast(errMsg, "error");
            appendMessage(`⚠️ **Extraction Error**: ${errMsg}`, "assistant");
            return;
        }

        const data = await r.json();
        toast("Analysis complete", "success");
        await loadDocuments();

        // Instead of opening and clearing, we append and focus
        appendMessage(formatExtractionCard(data), "assistant");
        appendMessage(`Synchronized **${data.filename}**. It has been added to your session context.`, "assistant");

        selectedDocId = data.doc_id;
        chatTitle.textContent = `${data.filename} Context`;
        renderDocList();

    } catch (err) {
        thinking.remove();
        toast("API Connectivity Lost", "error");
    }
}

async function sendQuestion() {
    const q = chatInput.value.trim();
    if (!q) return;

    appendMessage(q, "user");
    chatInput.value = "";
    sendBtn.disabled = true;

    const thinking = appendThinking();
    try {
        const body = { question: q };
        if (selectedDocId) body.doc_id = selectedDocId;

        const r = await fetch(`${API}/query`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
        });

        thinking.remove();
        if (r.ok) {
            const data = await r.json();
            appendMessage(data.answer, "assistant");
        } else {
            appendMessage("⚠️ Contextual retrieval failed. My extraction scope is limited for this document.", "assistant");
        }

    } catch {
        thinking.remove();
        appendMessage("⚠️ Network failure. Is uvicorn running?", "assistant");
    }
}

// ── Events ────────────────────────────────────────────────────────────────────
const inlineUploadBtn = document.getElementById("inline-upload-btn");
if (inlineUploadBtn) inlineUploadBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => {
    if (fileInput.files[0]) doUpload(fileInput.files[0]);
    fileInput.value = "";
});

sendBtn.addEventListener("click", sendQuestion);
chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendQuestion();
});

const newChatBtn = document.getElementById("new-chat-btn");

function resetChat() {
    sessionId = uuidv4();
    localStorage.setItem("receipt_ia_sid", sessionId);
    selectedDocId = null;
    chatTitle.textContent = "New Conversation";
    chatMessages.innerHTML = `
        <div class="flex items-start gap-4 max-w-3xl animate-slide-in">
            <div class="w-9 h-9 bg-indigo-600 rounded-xl flex items-center justify-center text-white shrink-0 shadow-lg shadow-indigo-600/20">
                <iconify-icon icon="solar:robot-bold" class="text-lg"></iconify-icon>
            </div>
            <div class="bubble-assistant p-4 text-sm font-medium leading-relaxed">
                New session started. Upload any receipt to begin analysis in this clear workspace.
            </div>
        </div>
    `;
    loadDocuments();
}

if (newChatBtn) newChatBtn.addEventListener("click", resetChat);

// ── Data Management Events ───────────────────────────────────────────────────
const viewDocsBtn = document.getElementById("view-json-btn"); // Note: mismatched IDs in HTML maybe? Fixing here.
const viewJsonBtn = document.getElementById("view-json-btn");
const realViewDocsBtn = document.getElementById("view-docs-btn");

if (realViewDocsBtn) {
    realViewDocsBtn.addEventListener("click", (e) => {
        e.preventDefault();
        const activeDocs = documents.filter(d => (d.session_id || 'default_session') === sessionId);
        if (activeDocs.length === 0) {
            toast("No documents in session", "info");
            return;
        }

        const list = activeDocs.map(d => `• **${d.filename}** (ID: ${d.doc_id})`).join("\n");
        appendMessage(`### 📂 Current Session Documents\n\n${list}`, "assistant");
    });
}

if (viewJsonBtn) {
    viewJsonBtn.addEventListener("click", (e) => {
        e.preventDefault();
        const activeDocs = documents.filter(d => (d.session_id || 'default_session') === sessionId);
        if (activeDocs.length === 0) {
            toast("No structured data available", "info");
            return;
        }

        const json = JSON.stringify(activeDocs.map(d => ({
            id: d.doc_id,
            file: d.filename,
            extracted: d.extracted
        })), null, 4);

        appendMessage(`### 🤖 Structured JSON Data\n\n\`\`\`json\n${json}\n\`\`\``, "assistant");
    });
}

// ── Init ──────────────────────────────────────────────────────────────────────
(async () => {
    await checkHealth();
    await loadDocuments();
    setInterval(checkHealth, 15000);
})();
