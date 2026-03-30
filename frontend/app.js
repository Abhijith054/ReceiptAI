/**
 * ReceiptAI Frontend — app.js
 * NEXA/AURA Dark-Theme Assistant Logic
 */

// Use window.location.origin to handle both localhost and production (Vercel) automatically
const API = window.location.origin;

// ── Auth State ────────────────────────────────────────────────────────────────
let authToken = localStorage.getItem("receipt_ia_token");
let userEmail = localStorage.getItem("receipt_ia_email");

// ── State ─────────────────────────────────────────────────────────────────────
let sessionId = userEmail || "guest";
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
const fileInput = document.getElementById("file-input");
const statusDot = document.getElementById("status-dot");
const apiStatusText = document.getElementById("api-status-text");
const toastCont = document.getElementById("toast-container");

// New Panel Refs
const previewPlaceholder = document.getElementById("preview-placeholder");
const activeImageWrapper = document.getElementById("active-image-wrapper");
const activeImage = document.getElementById("active-image");
const previewLoader = document.getElementById("preview-loader");
const analysisStatus = document.getElementById("analysis-status");
const dataShimmer = document.getElementById("data-shimmer");
const extractedFields = document.getElementById("extracted-fields");
const fieldVendor = document.getElementById("field-vendor");
const fieldDate = document.getElementById("field-date");
const fieldTotal = document.getElementById("field-total");
const reExtractBtn = document.getElementById("re-extract-btn");
const viewJsonHeader = document.getElementById("view-json-header");
const headerUploadBtn = document.getElementById("header-upload-btn");

// ── DOM Refs (Auth) ──────────────────────────────────────────────────────────
const loginOverlay = document.getElementById("login-overlay");
const loginEmailView = document.getElementById("login-email-view");
const loginOtpView = document.getElementById("login-otp-view");
const loginEmailInput = document.getElementById("login-email");
const loginOtpInput = document.getElementById("login-otp");
const sendOtpBtn = document.getElementById("send-otp-btn");
const verifyOtpBtn = document.getElementById("verify-otp-btn");
const loginError = document.getElementById("login-error");
const logoutBtn = document.getElementById("logout-btn");
const resendOtpBtn = document.getElementById("resend-otp-btn");
const backToEmailBtn = document.getElementById("back-to-email");

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
            if (statusDot) statusDot.className = "w-1.5 h-1.5 rounded-full bg-emerald-500 shadow-[0_0_8px_#10b981]";
            if (apiStatusText) {
                apiStatusText.textContent = "Assistant Online";
                apiStatusText.className = "text-[10px] font-bold text-emerald-500 uppercase tracking-widest";
            }
        } else throw new Error();
    } catch {
        if (statusDot) statusDot.className = "w-1.5 h-1.5 rounded-full bg-red-400";
        if (apiStatusText) {
            apiStatusText.textContent = "Assistant Offline";
            apiStatusText.className = "text-[10px] font-bold text-red-500 uppercase tracking-widest";
        }
    }
}

// ── Messages ──────────────────────────────────────────────────────────────────
function appendMessage(content, role = "assistant") {
    const wrap = document.createElement("div");
    wrap.className = `flex ${role === 'user' ? 'justify-end' : 'items-start'} gap-4 max-w-[90%] animate-slide-in ${role === 'user' ? 'ml-auto' : ''}`;

    const iconHtml = role === 'assistant'
        ? `<div class="ai-avatar-premium">
             <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                 <rect x="3" y="6" width="18" height="13" rx="4" fill="white" fill-opacity="0.1" stroke="white" stroke-width="1.5"/>
                 <circle cx="8" cy="12" r="1.5" fill="#22c55e" class="robot-eye" style="transform-origin: 8px 12px;"></circle>
                 <circle cx="16" cy="12" r="1.5" fill="#22c55e" class="robot-eye" style="transform-origin: 16px 12px;"></circle>
                 <path d="M10 16H14" stroke="white" stroke-width="1" stroke-linecap="round" opacity="0.5"></path>
                 <path d="M12 6V4M10 4H14" stroke="white" stroke-width="1.5" stroke-linecap="round"></path>
             </svg>
           </div>`
        : `<div class="w-8 h-8 bg-white/5 border border-white/5 rounded-xl flex items-center justify-center text-gray-500 shrink-0">
             <iconify-icon icon="solar:user-bold" class="text-lg"></iconify-icon>
           </div>`;

    wrap.innerHTML = role === 'assistant' ? `
    ${iconHtml}
    <div class="bubble-assistant p-4 text-[11px] leading-relaxed max-w-[85%]">${content.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>").replace(/\n/g, "<br/>")}</div>
  ` : `
    <div class="bubble-user p-4 text-[11px] leading-relaxed font-bold">${content}</div>
    ${iconHtml}
  `;

    chatMessages.appendChild(wrap);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return wrap;
}

function appendThinking() {
    const wrap = appendMessage(`<div class="flex items-center gap-3 py-1 font-bold text-indigo-400 uppercase tracking-[0.2em] text-[9px]"><div class="spinner"></div> Assistant Processing...</div>`, "assistant");
    wrap.id = "thinking-msg";
    return wrap;
}

// ── Panel Management ─────────────────────────────────────────────────────────
function updateExtractionPanel(record) {
    const ex = record.extracted || {};
    const imgData = record.image_data || ex.image_data || record.image_url || ex.image_url;

    // 1. Update Preview
    if (imgData) {
        activeImage.src = imgData;
        previewPlaceholder.classList.add("hidden");
        activeImageWrapper.classList.remove("hidden");
        if (headerUploadBtn) headerUploadBtn.classList.remove("hidden");
        // Reset opacity for transition
        activeImage.classList.remove("opacity-100");
        activeImage.classList.add("opacity-0");
    } else {
        previewPlaceholder.classList.remove("hidden");
        activeImageWrapper.classList.add("hidden");
        const previewText = document.getElementById("preview-text");
        if (previewText) previewText.textContent = "Upload a receipt to begin";
    }

    // 2. Update Data Fields
    dataShimmer.classList.add("hidden");
    extractedFields.classList.remove("hidden");
    analysisStatus.classList.remove("hidden");

    fieldVendor.textContent = ex.vendor || "Not detected";
    fieldVendor.classList.toggle("italic", !ex.vendor);
    fieldVendor.classList.toggle("text-gray-500", !ex.vendor);

    fieldDate.textContent = ex.date || "Not detected";
    fieldDate.classList.toggle("italic", !ex.date);
    fieldDate.classList.toggle("text-gray-500", !ex.date);

    if (typeof ex.total_amount === 'number') {
        fieldTotal.textContent = ex.total_amount.toLocaleString(undefined, { minimumFractionDigits: 2 });
    } else {
        fieldTotal.textContent = "0.00";
    }
}

function showLoadingPanel() {
    previewLoader.classList.remove("hidden");
    dataShimmer.classList.remove("hidden");
    extractedFields.classList.add("hidden");
    analysisStatus.classList.add("hidden");
}

function hideLoadingPanel() {
    previewLoader.classList.add("hidden");
}

// ── App Logic ─────────────────────────────────────────────────────────────────
async function loadDocuments() {
    try {
        const r = await fetch(`${API}/documents`);
        if (!r.ok) return;
        const data = await r.json();
        documents = data.documents || [];
        renderDocList();

        // Ensure consistent state on initial load
        if (documents.length > 0 && !selectedDocId) {
            const latest = [...documents].sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))[0];
            selectDocument(latest);
        }
    } catch { /* silence */ }
}

function renderDocList() {
    docList.innerHTML = "";

    // Add "Sessions" Label if not first item
    const label = document.createElement("div");
    label.className = "px-4 pt-4 mb-2 text-[9px] font-black text-gray-700 uppercase tracking-[0.2em]";
    label.innerText = "View Documents";
    docList.appendChild(label);

    if (documents.length === 0) {
        docList.innerHTML += `<p class="px-6 text-[9px] text-gray-700 italic font-bold uppercase tracking-widest py-10 text-center">Empty Vault</p>`;
        return;
    }

    // Sort by timestamp
    const sortedDocs = [...documents].sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

    sortedDocs.forEach(record => {
        const active = selectedDocId === record.doc_id;
        const item = document.createElement("div");
        item.className = `group flex items-center justify-between gap-3 px-5 py-3 rounded-xl cursor-pointer transition-all border shrink-0 mx-2 mb-1 ${active ? 'bg-indigo-600/10 border-indigo-500/30 text-white shadow-[0_0_15px_rgba(99,102,241,0.1)]' : 'border-transparent text-gray-500 hover:text-gray-300 hover:bg-white/[0.02]'}`;

        item.innerHTML = `
            <div class="flex items-center gap-3 truncate flex-1">
                 <iconify-icon icon="solar:document-text-bold" class="text-xl ${active ? 'text-indigo-400' : 'text-gray-800 transition-colors'}"></iconify-icon>
                 <div class="flex flex-col truncate">
                    <span class="truncate text-[10px] font-black uppercase tracking-tight">${record.filename || 'Unnamed Doc'}</span>
                    <span class="text-[8px] text-gray-700 font-bold">${new Date(record.timestamp).toLocaleDateString()}</span>
                 </div>
            </div>
            <button class="opacity-0 group-hover:opacity-100 text-gray-700 hover:text-red-500 transition-all p-1 delete-doc-btn" data-id="${record.doc_id}">
                <iconify-icon icon="solar:trash-bin-trash-linear" class="text-base"></iconify-icon>
            </button>
        `;

        item.addEventListener("click", (e) => {
            if (e.target.closest(".delete-doc-btn")) return;
            selectDocument(record);
        });

        item.querySelector(".delete-doc-btn").addEventListener("click", (e) => {
            e.stopPropagation();
            deleteDocument(record.doc_id);
        });

        docList.appendChild(item);
    });
}

function selectDocument(record) {
    selectedDocId = record.doc_id;
    updateExtractionPanel(record);
    renderDocList();

    chatMessages.innerHTML = "";
    // Rule 6: Show conversational invite
    appendMessage(`Document analyzed. Ask me anything about this receipt.`, "assistant");
}

async function deleteDocument(docId) {
    if (!confirm("Are you sure you want to delete this document?")) return;
    try {
        const r = await fetch(`${API}/documents/${docId}`, { method: "DELETE" });
        if (r.ok) {
            toast("Analysis removed", "success");
            if (selectedDocId === docId) {
                selectedDocId = null;
                previewPlaceholder.classList.remove("hidden");
                activeImageWrapper.classList.add("hidden");
                chatMessages.innerHTML = "";
            }
            await loadDocuments();
        }
    } catch { toast("Process error", "error"); }
}

async function doUpload(file) {
    // Instant UI Preview before server sync
    const previewUrl = URL.createObjectURL(file);
    activeImage.src = previewUrl;
    previewPlaceholder.classList.add("hidden");
    activeImageWrapper.classList.remove("hidden");
    activeImage.classList.remove("opacity-0");
    activeImage.classList.add("opacity-100");

    toast(`Syncing ${file.name}`, "info");
    showLoadingPanel();

    try {
        const form = new FormData();
        form.append("file", file);
        form.append("session_id", sessionId);
        const r = await fetch(`${API}/extract`, { method: "POST", body: form });

        hideLoadingPanel();
        if (!r.ok) {
            const errData = await r.json().catch(() => ({ detail: "Network error" }));
            toast(errData.detail || "Extraction Failed", "error");
            return;
        }

        const data = await r.json();
        toast("Analysis complete", "success");
        await loadDocuments();
        selectDocument(data);

    } catch (err) {
        hideLoadingPanel();
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
        appendMessage("⚠️ Network failure. Is server running?", "assistant");
    }
}

// ── Events ────────────────────────────────────────────────────────────────────
if (headerUploadBtn) headerUploadBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => {
    if (fileInput.files[0]) doUpload(fileInput.files[0]);
    fileInput.value = "";
});

sendBtn.addEventListener("click", sendQuestion);
chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendQuestion();
});

// (new-chat-btn removed)

reExtractBtn.addEventListener("click", () => {
    if (selectedDocId) {
        const doc = documents.find(d => d.doc_id === selectedDocId);
        if (doc) toast(`Retrying extraction for ${doc.filename}...`, "info");
    }
});

// ── Auth Logic ────────────────────────────────────────────────────────────────
async function initAuth() {
    if (!authToken) {
        showLogin();
    } else {
        hideLogin();
        // Update user profile in sidebar
        const emailSpan = document.getElementById("sidebar-user-email");
        const avatarChar = document.getElementById("avatar-char");
        if (emailSpan && userEmail) emailSpan.textContent = userEmail;
        if (avatarChar && userEmail) avatarChar.textContent = userEmail.charAt(0).toUpperCase();
    }
}

function showLogin() {
    loginOverlay.classList.remove("hidden");
    loginEmailView.classList.remove("hidden");
    loginOtpView.classList.add("hidden");
}

function hideLogin() {
    loginOverlay.classList.add("hidden");
}

function logout() {
    localStorage.removeItem("receipt_ia_token");
    localStorage.removeItem("receipt_ia_email");
    authToken = null;
    userEmail = null;
    location.reload();
}

sendOtpBtn.addEventListener("click", async () => {
    const email = loginEmailInput.value.trim();
    if (!email || !email.includes("@")) {
        showLoginError("Please enter a valid email");
        return;
    }

    sendOtpBtn.disabled = true;
    sendOtpBtn.textContent = "Sending...";
    loginError.classList.add("hidden");

    try {
        const r = await fetch(`${API}/send-otp`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email })
        });

        if (r.ok) {
            const data = await r.json();
            userEmail = email;
            loginEmailView.classList.add("hidden");
            loginOtpView.classList.remove("hidden");
            
            if (data.dev) {
                // Email delivery failed on server - do NOT auto-fill, show error
                showLoginError("Email delivery failed. Check server email configuration.");
                loginOtpView.classList.add("hidden");
                loginEmailView.classList.remove("hidden");
            } else {
                toast("Verification code sent to " + email, "success");
            }
        } else {
            const data = await r.json();
            showLoginError(data.detail || "Failed to send code");
        }
    } catch (e) {
        showLoginError("Network error. Try again.");
    } finally {
        sendOtpBtn.disabled = false;
        sendOtpBtn.textContent = "Send Access Code";
    }
});

verifyOtpBtn.addEventListener("click", async () => {
    const otp = loginOtpInput.value.trim();
    if (otp.length !== 6) {
        showLoginError("Enter 6-digit code");
        return;
    }

    verifyOtpBtn.disabled = true;
    verifyOtpBtn.textContent = "Verifying...";

    try {
        const r = await fetch(`${API}/verify-otp`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email: userEmail, otp })
        });

        if (r.ok) {
            const data = await r.json();
            authToken = data.access_token;
            localStorage.setItem("receipt_ia_token", authToken);
            localStorage.setItem("receipt_ia_email", userEmail);
            hideLogin();
            toast("Access Granted", "success");
            location.reload();
        } else {
            const data = await r.json();
            showLoginError(data.detail || "Invalid code");
        }
    } catch (e) {
        showLoginError("Sync failed. Check connection.");
    } finally {
        verifyOtpBtn.disabled = false;
        verifyOtpBtn.textContent = "Verify & Enter";
    }
});

if (backToEmailBtn) {
    backToEmailBtn.addEventListener("click", () => {
        loginOtpView.classList.add("hidden");
        loginEmailView.classList.remove("hidden");
    });
}

if (resendOtpBtn) {
    resendOtpBtn.addEventListener("click", () => sendOtpBtn.click());
}

if (logoutBtn) {
    logoutBtn.addEventListener("click", logout);
}

function showLoginError(msg) {
    loginError.textContent = msg;
    loginError.classList.remove("hidden");
    toast(msg, "error");
}

// ── Init ──────────────────────────────────────────────────────────────────────
(async () => {
    await initAuth();
    if (authToken) {
        await checkHealth();
        await loadDocuments();
        setInterval(checkHealth, 15000);
    }
})();
