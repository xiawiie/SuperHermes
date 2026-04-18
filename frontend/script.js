const { createApp } = Vue;

createApp({
    data() {
        return {
            messages: [],
            userInput: "",
            isLoading: false,
            activeView: "chat",
            abortController: null,
            sessionId: "session_" + Date.now(),
            sessions: [],
            showHistoryPanel: false,
            isComposing: false,
            documents: [],
            documentsLoading: false,
            selectedFiles: [],
            pendingContextFiles: [],
            activeUploadResolvers: [],
            isUploading: false,
            uploadProgress: "",
            currentUploadPercent: 0,
            completedUploads: 0,
            currentUploadName: "",
            maxUploadFiles: 5,
            isPageDragActive: false,
            pageDragDepth: 0,
            token: localStorage.getItem("accessToken") || "",
            currentUser: null,
            authMode: "login",
            authForm: {
                username: "",
                password: "",
                role: "user",
                admin_code: "",
            },
            authLoading: false,
            streamBuffer: "",
            streamFlushScheduled: false,
            streamFlushTimer: null,
            streamFlushIntervalMs: 48,
            scrollScheduled: false,
            historyEditSessionId: null,
            historyEditDraft: "",
            historyEditBlurTimer: null,
        };
    },

    computed: {
        isAuthenticated() {
            return !!this.token && !!this.currentUser;
        },

        isAdmin() {
            return this.currentUser?.role === "admin";
        },

        userInitial() {
            return (this.currentUser?.username || "U").slice(0, 1);
        },

        historyEditSizerText() {
            const t = this.historyEditDraft ?? "";
            return t === "" ? "\u00a0" : t;
        },

        canAcceptKnowledgeUpload() {
            return this.isAuthenticated && this.isAdmin;
        },

        uploadProgressPercent() {
            if (!this.selectedFiles.length) {
                return 0;
            }
            const partial = Math.min(Math.max(this.currentUploadPercent, 0), 100) / 100;
            const combined = ((this.completedUploads + partial) / this.selectedFiles.length) * 100;
            return Math.max(0, Math.min(100, Math.round(combined)));
        },
        uploadStatusText() {
            if (this.isUploading && this.selectedFiles.length) {
                const currentIndex = Math.min(this.completedUploads + 1, this.selectedFiles.length);
                const currentName = this.currentUploadName || this.selectedFiles[currentIndex - 1]?.name || "";
                return `正在上传 ${currentIndex}/${this.selectedFiles.length}${currentName ? `：${currentName}` : ""}`;
            }
            if (this.selectedFiles.length) {
                return `已选择 ${this.selectedFiles.length} 个文件，最多 ${this.maxUploadFiles} 个`;
            }
            return this.uploadProgress;
        },

        showUploadStatusBar() {
            return this.isUploading || !!this.uploadProgress || this.selectedFiles.length > 0 || this.pendingContextFiles.length > 0;
        },

        composerUploadTitle() {
            if (this.isUploading) {
                return this.uploadStatusText;
            }
            if (this.pendingContextFiles.length) {
                return `将作为本轮对话上下文：${this.pendingContextFiles.length} 个文件`;
            }
            return this.uploadStatusText;
        },
    },

    watch: {
        showHistoryPanel(val) {
            if (!val) {
                this.cancelHistoryRename();
            }
        },
    },

    async mounted() {
        this.configureMarked();
        this.bindPageUploadEvents();
        if (this.token) {
            try {
                await this.fetchMe();
            } catch (_) {
                this.handleLogout();
            }
        }
    },

    beforeUnmount() {
        this.unbindPageUploadEvents();
    },

    methods: {
        configureMarked() {
            if (!window.marked) return;
            marked.setOptions({
                highlight(code, lang) {
                    if (!window.hljs) return code;
                    const language = hljs.getLanguage(lang) ? lang : "plaintext";
                    return hljs.highlight(code, { language }).value;
                },
                langPrefix: "hljs language-",
                breaks: true,
                gfm: true,
            });
        },

        renderMarkdown(text) {
            const source = text || "";
            return window.marked ? marked.parse(source) : this.escapeHtml(source);
        },

        escapeHtml(text) {
            const div = document.createElement("div");
            div.textContent = text || "";
            return div.innerHTML;
        },

        createUserMessage(text, contextFiles = []) {
            return {
                id: "msg_" + Date.now() + "_" + Math.random().toString(16).slice(2),
                text,
                html: "",
                isUser: true,
                contextFiles,
            };
        },

        createBotMessage(text = "", ragTrace = null) {
            return {
                id: "msg_" + Date.now() + "_" + Math.random().toString(16).slice(2),
                text,
                html: this.renderMarkdown(text),
                isUser: false,
                isThinking: !text,
                ragTrace,
                ragSteps: [],
            };
        },

        authHeaders(extra = {}) {
            const headers = { ...extra };
            if (this.token) {
                headers.Authorization = `Bearer ${this.token}`;
            }
            return headers;
        },

        async authFetch(url, options = {}) {
            const opts = { ...options };
            opts.headers = this.authHeaders(opts.headers || {});
            const response = await fetch(url, opts);
            if (response.status === 401) {
                this.handleLogout();
                throw new Error("登录已过期，请重新登录");
            }
            return response;
        },

        async fetchMe() {
            const response = await this.authFetch("/auth/me");
            if (!response.ok) {
                throw new Error("认证失败");
            }
            this.currentUser = await response.json();
            this.activeView = "chat";
        },

        toggleAuthMode() {
            this.authMode = this.authMode === "login" ? "register" : "login";
            this.authForm.password = "";
            this.authForm.admin_code = "";
        },

        async handleAuthSubmit() {
            if (this.authLoading) return;
            const username = this.authForm.username.trim();
            const password = this.authForm.password.trim();
            if (!username || !password) {
                alert("用户名和密码不能为空");
                return;
            }

            this.authLoading = true;
            try {
                const endpoint = this.authMode === "login" ? "/auth/login" : "/auth/register";
                const payload = { username, password };
                if (this.authMode === "register") {
                    payload.role = this.authForm.role;
                    payload.admin_code = this.authForm.admin_code || null;
                }

                const response = await fetch(endpoint, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                });

                const data = await response.json().catch(() => ({}));
                if (!response.ok) {
                    throw new Error(data.detail || "认证失败");
                }

                this.token = data.access_token;
                this.currentUser = { username: data.username, role: data.role };
                localStorage.setItem("accessToken", this.token);
                this.authForm.password = "";
                this.authForm.admin_code = "";
                this.messages = [];
                this.sessionId = "session_" + Date.now();
                this.activeView = "chat";
                this.showHistoryPanel = false;
            } catch (error) {
                alert(error.message);
            } finally {
                this.authLoading = false;
            }
        },

        handleLogout() {
            if (this.abortController) {
                this.abortController.abort();
            }
            this.token = "";
            this.currentUser = null;
            this.messages = [];
            this.sessions = [];
            this.documents = [];
            this.selectedFiles = [];
            this.pendingContextFiles = [];
            this.activeView = "chat";
            this.showHistoryPanel = false;
            this.streamBuffer = "";
            this.uploadProgress = "";
            this.currentUploadPercent = 0;
            this.completedUploads = 0;
            this.currentUploadName = "";
            this.isPageDragActive = false;
            this.pageDragDepth = 0;
            this.clearStreamFlushTimer();
            localStorage.removeItem("accessToken");
        },

        triggerUploadPicker() {
            if (!this.isAdmin) {
                alert("仅管理员可上传文件");
                return;
            }
            const input = this.$refs.globalFileInput || this.$refs.fileInput;
            input?.click?.();
        },

        bindPageUploadEvents() {
            this._boundHandleWindowDragEnter = (event) => this.handleWindowDragEnter(event);
            this._boundHandleWindowDragOver = (event) => this.handleWindowDragOver(event);
            this._boundHandleWindowDragLeave = (event) => this.handleWindowDragLeave(event);
            this._boundHandleWindowDrop = (event) => this.handleWindowDrop(event);
            window.addEventListener("dragenter", this._boundHandleWindowDragEnter);
            window.addEventListener("dragover", this._boundHandleWindowDragOver);
            window.addEventListener("dragleave", this._boundHandleWindowDragLeave);
            window.addEventListener("drop", this._boundHandleWindowDrop);
        },

        unbindPageUploadEvents() {
            if (!this._boundHandleWindowDragEnter) {
                return;
            }
            window.removeEventListener("dragenter", this._boundHandleWindowDragEnter);
            window.removeEventListener("dragover", this._boundHandleWindowDragOver);
            window.removeEventListener("dragleave", this._boundHandleWindowDragLeave);
            window.removeEventListener("drop", this._boundHandleWindowDrop);
        },

        isFileDragEvent(event) {
            const types = Array.from(event?.dataTransfer?.types || []);
            return types.includes("Files") || !!event?.dataTransfer?.files?.length;
        },

        isAcceptedUploadFile(file) {
            const name = file?.name?.toLowerCase?.() || "";
            return (
                name.endsWith(".pdf") ||
                name.endsWith(".doc") ||
                name.endsWith(".docx") ||
                name.endsWith(".xls") ||
                name.endsWith(".xlsx")
            );
        },

        isSameQueuedFile(left, right) {
            return (
                left?.name === right?.name &&
                left?.size === right?.size &&
                left?.lastModified === right?.lastModified
            );
        },

        queueSelectedFiles(files) {
            const incoming = Array.from(files || []).filter(Boolean);
            if (!incoming.length) {
                return 0;
            }

            const validFiles = incoming.filter((file) => this.isAcceptedUploadFile(file));
            const invalidCount = incoming.length - validFiles.length;
            const dedupedFiles = validFiles.filter(
                (file) => !this.selectedFiles.some((existing) => this.isSameQueuedFile(existing, file))
            );
            const remainingSlots = Math.max(this.maxUploadFiles - this.selectedFiles.length, 0);
            const acceptedFiles = dedupedFiles.slice(0, remainingSlots);

            if (acceptedFiles.length) {
                this.selectedFiles = this.selectedFiles.concat(acceptedFiles);
            }

            if (!acceptedFiles.length) {
                this.uploadProgress = `最多同时上传 ${this.maxUploadFiles} 个文件`;
                return 0;
            }

            const skippedForLimit = dedupedFiles.length - acceptedFiles.length;
            if (invalidCount > 0 || skippedForLimit > 0) {
                this.uploadProgress = `已加入 ${acceptedFiles.length} 个文件；仅支持 PDF、Word、Excel，且最多 ${this.maxUploadFiles} 个`;
            } else {
                this.uploadProgress = `已选择 ${this.selectedFiles.length} 个文件`;
            }

            return acceptedFiles.length;
        },

        removeSelectedFile(index) {
            this.selectedFiles.splice(index, 1);
            if (!this.selectedFiles.length && !this.isUploading) {
                this.uploadProgress = "";
            }
        },

        addPendingContextFile(file) {
            const filename = file?.filename || file?.name;
            if (!filename) return;
            if (this.pendingContextFiles.some((existing) => existing.filename === filename)) {
                return;
            }
            this.pendingContextFiles.push({
                filename,
                addedAt: Date.now(),
            });
        },

        removePendingContextFile(index) {
            this.pendingContextFiles.splice(index, 1);
        },

        consumePendingContextFiles() {
            const filenames = this.pendingContextFiles.map((file) => file.filename).filter(Boolean);
            this.pendingContextFiles = [];
            return filenames;
        },

        waitForActiveUpload() {
            if (!this.isUploading) {
                return Promise.resolve();
            }
            return new Promise((resolve) => {
                this.activeUploadResolvers.push(resolve);
            });
        },

        resolveActiveUploadWaiters() {
            const resolvers = this.activeUploadResolvers.splice(0);
            resolvers.forEach((resolve) => resolve());
        },

        handleWindowDragEnter(event) {
            if (!this.canAcceptKnowledgeUpload || !this.isFileDragEvent(event)) {
                return;
            }
            event.preventDefault();
            this.pageDragDepth += 1;
            this.isPageDragActive = true;
        },

        handleWindowDragOver(event) {
            if (!this.canAcceptKnowledgeUpload || !this.isFileDragEvent(event)) {
                return;
            }
            event.preventDefault();
            if (event.dataTransfer) {
                event.dataTransfer.dropEffect = "copy";
            }
            this.isPageDragActive = true;
        },

        handleWindowDragLeave(event) {
            if (!this.canAcceptKnowledgeUpload || !this.isFileDragEvent(event)) {
                return;
            }
            event.preventDefault();
            this.pageDragDepth = Math.max(this.pageDragDepth - 1, 0);
            if (this.pageDragDepth === 0) {
                this.isPageDragActive = false;
            }
        },

        async handleWindowDrop(event) {
            if (!this.canAcceptKnowledgeUpload || !this.isFileDragEvent(event)) {
                return;
            }
            event.preventDefault();
            this.pageDragDepth = 0;
            this.isPageDragActive = false;
            const addedCount = this.queueSelectedFiles(Array.from(event.dataTransfer?.files || []));
            if (addedCount > 0 && !this.isUploading) {
                await this.uploadDocument();
            }
        },

        handleCompositionStart() {
            this.isComposing = true;
        },

        handleCompositionEnd() {
            this.isComposing = false;
        },

        handleKeyDown(event) {
            if (event.key === "Enter" && !event.shiftKey && !this.isComposing) {
                event.preventDefault();
                this.handleSend();
            }
        },

        handleStop() {
            if (this.abortController) {
                this.abortController.abort();
            }
        },

        usePrompt(prompt) {
            this.userInput = prompt;
            this.$nextTick(() => {
                if (this.$refs.textarea) {
                    this.$refs.textarea.focus();
                    this.resetTextareaHeight();
                    this.autoResize({ target: this.$refs.textarea });
                }
            });
        },

        async handleSend() {
            if (!this.isAuthenticated) {
                alert("请先登录");
                return;
            }

            const text = this.userInput.trim();
            if (!text || this.isLoading || this.isComposing) return;

            const hadUploadWork = this.isUploading || this.selectedFiles.length > 0;
            if (this.isUploading) {
                await this.waitForActiveUpload();
            } else if (this.selectedFiles.length) {
                await this.uploadDocument();
            }
            if (hadUploadWork && !this.pendingContextFiles.length) {
                return;
            }

            const contextFiles = this.pendingContextFiles.map((file) => file.filename).filter(Boolean);
            this.messages.push(this.createUserMessage(text, contextFiles));
            this.userInput = "";
            this.$nextTick(() => {
                this.resetTextareaHeight();
                this.scheduleScrollToBottom();
            });

            this.messages.push(this.createBotMessage());
            const botMsgIdx = this.messages.length - 1;
            const streamOk = await this.streamChatToBotSlot(text, botMsgIdx, { regenerate: false, contextFiles });
            if (streamOk) {
                this.consumePendingContextFiles();
            }
        },

        canRegenerateAssistant(index) {
            if (this.messages.length === 0 || index !== this.messages.length - 1) {
                return false;
            }
            const msg = this.messages[index];
            if (msg.isUser || msg.isThinking) {
                return false;
            }
            const prev = this.messages[index - 1];
            return !!(prev && prev.isUser && prev.text);
        },

        async copyBotMessage(msg) {
            const t = (msg && msg.text) || "";
            if (!t) return;
            try {
                await navigator.clipboard.writeText(t);
            } catch (_) {
                try {
                    const ta = document.createElement("textarea");
                    ta.value = t;
                    ta.style.position = "fixed";
                    ta.style.left = "-9999px";
                    document.body.appendChild(ta);
                    ta.select();
                    document.execCommand("copy");
                    document.body.removeChild(ta);
                } catch {
                    alert("复制失败，请手动选择文本复制");
                }
            }
        },

        async regenerateAssistantAt(index) {
            if (!this.canRegenerateAssistant(index) || this.isLoading) {
                return;
            }
            const userMsg = this.messages[index - 1];
            if (!userMsg || !userMsg.isUser) {
                return;
            }
            const userText = userMsg.text;
            const bot = this.messages[index];
            bot.text = "";
            bot.html = this.renderMarkdown("");
            bot.isThinking = true;
            bot.ragTrace = null;
            bot.ragSteps = [];
            this.$nextTick(() => this.scheduleScrollToBottom());
            await this.streamChatToBotSlot(userText, index, { regenerate: true });
        },

        async streamChatToBotSlot(userText, botMsgIdx, options = {}) {
            const regenerate = !!(options && options.regenerate);
            const contextFiles = Array.isArray(options.contextFiles) ? options.contextFiles : [];
            let streamOk = false;
            let streamHadError = false;
            this.isLoading = true;
            this.streamBuffer = "";
            this.streamFlushScheduled = false;
            this.clearStreamFlushTimer();

            this.abortController = new AbortController();

            try {
                const response = await this.authFetch("/chat/stream", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        message: userText,
                        session_id: this.sessionId,
                        regenerate,
                        context_files: contextFiles,
                    }),
                    signal: this.abortController.signal,
                });

                if (!response.ok) throw new Error(`HTTP ${response.status}`);

                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                let buffer = "";
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });

                    let eventEndIndex;
                    while ((eventEndIndex = buffer.indexOf("\n\n")) !== -1) {
                        const eventStr = buffer.slice(0, eventEndIndex);
                        buffer = buffer.slice(eventEndIndex + 2);
                        const eventType = this.consumeSseEvent(eventStr, botMsgIdx);
                        if (eventType === "error") {
                            streamHadError = true;
                        }
                    }
                }
                streamOk = !streamHadError;
            } catch (error) {
                const botMsg = this.messages[botMsgIdx];
                if (botMsg) {
                    botMsg.isThinking = false;
                }
                if (error.name === "AbortError") {
                    this.streamBuffer += botMsg && botMsg.text ? "\n\n_(回答已被终止)_" : "(已终止回答)";
                } else {
                    this.streamBuffer += `\n\n出现了一点问题：${error.message}`;
                }
            } finally {
                this.flushStreamBuffer(botMsgIdx, true);
                this.isLoading = false;
                this.abortController = null;
                this.scheduleScrollToBottom();
            }
            return streamOk;
        },

        consumeSseEvent(eventStr, botMsgIdx) {
            if (!eventStr.startsWith("data: ")) return;
            const dataStr = eventStr.slice(6);
            if (dataStr === "[DONE]") return "done";

            try {
                const data = JSON.parse(dataStr);
                const botMsg = this.messages[botMsgIdx];
                if (!botMsg) return;

                if (data.type === "content") {
                    if (botMsg.isThinking) {
                        botMsg.isThinking = false;
                    }
                    this.streamBuffer += data.content || "";
                    this.scheduleStreamFlush(botMsgIdx);
                } else if (data.type === "trace") {
                    botMsg.ragTrace = data.rag_trace;
                } else if (data.type === "rag_step") {
                    if (!botMsg.ragSteps) {
                        botMsg.ragSteps = [];
                    }
                    botMsg.ragSteps.push(data.step);
                    this.scheduleScrollToBottom();
                } else if (data.type === "error") {
                    botMsg.isThinking = false;
                    this.streamBuffer += `\n\n[Error: ${data.content}]`;
                    this.scheduleStreamFlush(botMsgIdx);
                }
                return data.type;
            } catch (error) {
                console.warn("SSE parse error:", error);
            }
            return null;
        },

        scheduleStreamFlush(botMsgIdx) {
            if (this.streamFlushScheduled) return;
            this.streamFlushScheduled = true;
            this.streamFlushTimer = setTimeout(() => {
                this.streamFlushTimer = null;
                this.flushStreamBuffer(botMsgIdx);
            }, this.streamFlushIntervalMs);
        },

        clearStreamFlushTimer() {
            if (this.streamFlushTimer !== null) {
                clearTimeout(this.streamFlushTimer);
                this.streamFlushTimer = null;
            }
            this.streamFlushScheduled = false;
        },

        flushStreamBuffer(botMsgIdx, force = false) {
            if (force) {
                this.clearStreamFlushTimer();
            }

            if (!force && !this.streamBuffer) {
                this.streamFlushScheduled = false;
                return;
            }

            const botMsg = this.messages[botMsgIdx];
            if (!botMsg) {
                this.streamBuffer = "";
                this.streamFlushScheduled = false;
                return;
            }

            if (this.streamBuffer) {
                botMsg.text += this.streamBuffer;
                this.streamBuffer = "";
            }
            botMsg.html = this.renderMarkdown(botMsg.text);
            this.streamFlushScheduled = false;
            this.scheduleScrollToBottom();
        },

        autoResize(event) {
            const textarea = event.target;
            textarea.style.height = "auto";
            textarea.style.height = Math.min(textarea.scrollHeight, 160) + "px";
        },

        resetTextareaHeight() {
            if (this.$refs.textarea) {
                this.$refs.textarea.style.height = "auto";
            }
        },

        scheduleScrollToBottom() {
            if (this.scrollScheduled) return;
            this.scrollScheduled = true;
            requestAnimationFrame(() => {
                this.scrollScheduled = false;
                this.scrollToBottom();
            });
        },

        scrollToBottom() {
            if (this.$refs.chatContainer) {
                this.$refs.chatContainer.scrollTop = this.$refs.chatContainer.scrollHeight;
            }
        },

        handleNewChat() {
            if (!this.isAuthenticated) return;
            this.messages = [];
            this.sessionId = "session_" + Date.now();
            this.activeView = "chat";
            this.showHistoryPanel = false;
            this.$nextTick(() => this.scheduleScrollToBottom());
        },

        handleClearChat() {
            if (confirm("确定要清空当前对话吗？")) {
                this.messages = [];
            }
        },

        async handleHistory() {
            if (!this.isAuthenticated) return;
            this.showHistoryPanel = true;
            try {
                const response = await this.authFetch("/sessions");
                if (!response.ok) {
                    throw new Error("Failed to load sessions");
                }
                const data = await response.json();
                this.sessions = data.sessions;
            } catch (error) {
                alert("加载历史记录失败：" + error.message);
            }
        },

        async loadSession(sessionId) {
            this.sessionId = sessionId;
            this.showHistoryPanel = false;
            this.activeView = "chat";

            try {
                const response = await this.authFetch(`/sessions/${encodeURIComponent(sessionId)}`);
                if (!response.ok) {
                    throw new Error("Failed to load session messages");
                }
                const data = await response.json();
                this.messages = data.messages.map((msg) => {
                    if (msg.type === "human") {
                        return this.createUserMessage(msg.content);
                    }
                    return this.createBotMessage(msg.content, msg.rag_trace || null);
                });
                this.$nextTick(() => this.scheduleScrollToBottom());
            } catch (error) {
                alert("加载会话失败：" + error.message);
                this.messages = [];
            }
        },

        clearHistoryEditBlurTimer() {
            if (this.historyEditBlurTimer) {
                clearTimeout(this.historyEditBlurTimer);
                this.historyEditBlurTimer = null;
            }
        },

        cancelHistoryRename() {
            this.clearHistoryEditBlurTimer();
            this.historyEditSessionId = null;
            this.historyEditDraft = "";
        },

        startHistoryRename(session) {
            if (!session || !this.isAuthenticated) return;
            if (this.historyEditSessionId === session.session_id) return;
            this.clearHistoryEditBlurTimer();
            this.historyEditSessionId = session.session_id;
            this.historyEditDraft = session.title || session.session_id;
            this.$nextTick(() => {
                const panel = this.$el && this.$el.querySelector(".history-panel");
                const inp = panel && panel.querySelector(".history-title-input");
                if (inp) {
                    inp.focus();
                    inp.select();
                }
            });
        },

        onHistoryTitleKeydown(event, session) {
            if (event.key === "Enter") {
                event.preventDefault();
                this.clearHistoryEditBlurTimer();
                this.commitHistoryRename(session);
            } else if (event.key === "Escape") {
                event.preventDefault();
                this.cancelHistoryRename();
            }
        },

        scheduleCommitHistoryRename(session) {
            this.clearHistoryEditBlurTimer();
            this.historyEditBlurTimer = setTimeout(() => {
                this.historyEditBlurTimer = null;
                this.commitHistoryRename(session);
            }, 150);
        },

        async commitHistoryRename(session) {
            if (!session || !this.isAuthenticated) return;
            if (this.historyEditSessionId !== session.session_id) return;
            const sid = session.session_id;
            const prevDisplay = `${session.title || sid}`.trim();
            const draft = `${this.historyEditDraft || ""}`.trim();
            if (draft === prevDisplay) {
                this.cancelHistoryRename();
                return;
            }
            const titlePayload = !draft || draft === sid ? "" : draft;
            try {
                const response = await this.authFetch(`/sessions/${encodeURIComponent(sid)}`, {
                    method: "PATCH",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ title: titlePayload }),
                });
                const payload = await response.json().catch(() => ({}));
                if (!response.ok) {
                    throw new Error(payload.detail || "重命名失败");
                }
                const idx = this.sessions.findIndex((s) => s.session_id === sid);
                if (idx !== -1) {
                    this.sessions[idx] = {
                        ...this.sessions[idx],
                        title: payload.title ?? null,
                        updated_at: new Date().toISOString(),
                    };
                    this.sessions.sort(
                        (a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime(),
                    );
                }
                this.cancelHistoryRename();
            } catch (error) {
                alert("重命名失败：" + error.message);
            }
        },

        async deleteSession(sessionId) {
            const label =
                this.sessions.find((s) => s.session_id === sessionId)?.title || sessionId;
            if (!confirm(`确定要删除会话「${label}」吗？`)) {
                return;
            }

            try {
                const response = await this.authFetch(`/sessions/${encodeURIComponent(sessionId)}`, {
                    method: "DELETE",
                });

                const payload = await response.json().catch(() => ({}));
                if (!response.ok) {
                    throw new Error(payload.detail || "Delete failed");
                }

                this.sessions = this.sessions.filter((s) => s.session_id !== sessionId);

                if (this.sessionId === sessionId) {
                    this.messages = [];
                    this.sessionId = "session_" + Date.now();
                    this.activeView = "chat";
                }
            } catch (error) {
                alert("删除会话失败：" + error.message);
            }
        },

        handleKnowledge() {
            if (!this.isAdmin) {
                alert("仅管理员可访问知识库");
                return;
            }
            this.activeView = "knowledge";
            this.showHistoryPanel = false;
            this.loadDocuments();
        },

        async loadDocuments() {
            this.documentsLoading = true;
            try {
                const response = await this.authFetch("/documents");
                if (!response.ok) {
                    const data = await response.json().catch(() => ({}));
                    throw new Error(data.detail || "Failed to load documents");
                }
                const data = await response.json();
                this.documents = data.documents;
            } catch (error) {
                alert("加载文档列表失败：" + error.message);
            } finally {
                this.documentsLoading = false;
            }
        },

        async handleFileSelect(event) {
            const addedCount = this.queueSelectedFiles(event.target.files);
            if (event.target) {
                event.target.value = "";
            }
            if (addedCount > 0 && this.activeView === "chat" && !this.isUploading) {
                await this.uploadDocument();
            }
        },

        uploadSingleFile(file, onProgress) {
            return new Promise((resolve, reject) => {
                const xhr = new XMLHttpRequest();
                xhr.open("POST", "/documents/upload");

                const headers = this.authHeaders();
                Object.entries(headers).forEach(([key, value]) => xhr.setRequestHeader(key, value));

                xhr.upload.addEventListener("progress", (event) => {
                    if (event.lengthComputable) {
                        onProgress?.(Math.round((event.loaded / event.total) * 100));
                    }
                });

                xhr.addEventListener("load", () => {
                    let payload = {};
                    try {
                        payload = xhr.responseText ? JSON.parse(xhr.responseText) : {};
                    } catch (_) {
                        payload = {};
                    }

                    if (xhr.status === 401) {
                        this.handleLogout();
                        reject(new Error("登录已过期，请重新登录"));
                        return;
                    }

                    if (xhr.status < 200 || xhr.status >= 300) {
                        reject(new Error(payload.detail || "Upload failed"));
                        return;
                    }

                    onProgress?.(100);
                    resolve(payload);
                });

                xhr.addEventListener("error", () => {
                    reject(new Error("网络异常，上传未完成"));
                });

                const formData = new FormData();
                formData.append("file", file);
                xhr.send(formData);
            });
        },
        async uploadDocument() {
            if (!this.selectedFiles.length) {
                alert("\u8bf7\u5148\u9009\u62e9\u6587\u4ef6");
                return;
            }
            if (this.isUploading) return;

            this.isUploading = true;
            this.completedUploads = 0;
            this.currentUploadPercent = 0;

            try {
                const filesToUpload = [...this.selectedFiles];
                for (let index = 0; index < filesToUpload.length; index += 1) {
                    const file = filesToUpload[index];
                    this.currentUploadName = file.name;
                    this.currentUploadPercent = 0;
                    this.uploadProgress = `\u6b63\u5728\u4e0a\u4f20 ${index + 1}/${filesToUpload.length}\uff1a${file.name}`;
                    const data = await this.uploadSingleFile(file, (percent) => {
                        this.currentUploadPercent = percent;
                    });
                    this.completedUploads = index + 1;
                    this.currentUploadPercent = 100;
                    this.uploadProgress = data.message || `${file.name} \u4e0a\u4f20\u6210\u529f`;
                    this.addPendingContextFile(data.filename ? data : file);
                }

                if (this.$refs.fileInput) {
                    this.$refs.fileInput.value = "";
                }
                if (this.$refs.globalFileInput) {
                    this.$refs.globalFileInput.value = "";
                }
                await this.loadDocuments();
                this.selectedFiles = [];
                this.currentUploadName = "";
                setTimeout(() => {
                    this.uploadProgress = "";
                    this.currentUploadPercent = 0;
                    this.completedUploads = 0;
                }, 3000);
            } catch (error) {
                this.uploadProgress = `\u4e0a\u4f20\u5931\u8d25\uff1a${error.message}`;
            } finally {
                this.isUploading = false;
                this.resolveActiveUploadWaiters();
            }
        },

        async deleteDocument(filename) {
            if (!confirm(`确定要删除文档 "${filename}" 吗？这将同时删除 Milvus 中的所有相关向量。`)) {
                return;
            }

            try {
                const response = await this.authFetch(`/documents/${encodeURIComponent(filename)}`, {
                    method: "DELETE",
                });

                if (!response.ok) {
                    const error = await response.json().catch(() => ({}));
                    throw new Error(error.detail || "Delete failed");
                }

                await this.loadDocuments();
            } catch (error) {
                alert("删除文档失败：" + error.message);
            }
        },

        getFileIcon(fileType) {
            if (fileType === "PDF") {
                return "fas fa-file-pdf";
            }
            if (fileType === "Word") {
                return "fas fa-file-word";
            }
            if (fileType === "Excel") {
                return "fas fa-file-excel";
            }
            return "fas fa-file-lines";
        },

        currentThinkingLabel(msg) {
            if (msg.ragSteps && msg.ragSteps.length) {
                return msg.ragSteps[msg.ragSteps.length - 1].label;
            }
            return "正在思考中...";
        },

        traceSummary(trace) {
            if (!trace) return "";
            if (!trace.tool_used) return "未调用工具";
            const stage = trace.retrieval_stage || trace.retrieval_mode || "已检索";
            return `${trace.tool_name || "工具"} · ${stage}`;
        },

        traceChunks(trace) {
            if (!trace) return [];
            const chunks = [];
            const add = (items) => {
                if (Array.isArray(items)) {
                    items.forEach((item) => chunks.push(item));
                }
            };
            add(trace.initial_retrieved_chunks);
            add(trace.expanded_retrieved_chunks);
            if (!chunks.length) {
                add(trace.retrieved_chunks);
            }
            return chunks.slice(0, 8);
        },
    },
}).mount("#app");
