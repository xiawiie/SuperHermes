import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";
import vm from "node:vm";

const __dirname = dirname(fileURLToPath(import.meta.url));

const read = (file) => readFileSync(join(__dirname, file), "utf8");

const index = read("index.html");
const css = read("style.css");
const script = read("script.js");
const backendSchemas = read("../backend/contracts/schemas.py");
const backendApi = read("../backend/api.py");
const backendRouterAuth = read("../backend/routers/auth.py");
const backendRouterChat = read("../backend/routers/chat.py");
const backendRouterSessions = read("../backend/routers/sessions.py");
const backendRouterDocuments = read("../backend/routers/documents.py");
const backendAgent = read("../backend/chat/agent.py");
const backendTools = read("../backend/chat/tools.py");
const backendRagPipeline = read("../backend/rag/pipeline.py");
const backendRagUtils = read("../backend/rag/utils.py");
const backendRagRetrieval = read("../backend/rag/retrieval.py");

function loadAppOptions() {
  let capturedOptions = null;
  const windowListeners = new Map();

  const context = {
    console,
    setTimeout,
    clearTimeout,
    AbortController,
    FormData: class MockFormData {
      constructor() {
        this.entries = [];
      }

      append(key, value) {
        this.entries.push([key, value]);
      }
    },
    localStorage: {
      getItem() {
        return null;
      },
      setItem() {},
      removeItem() {},
    },
    document: {
      createElement() {
        let text = "";
        return {
          set textContent(value) {
            text = value ?? "";
          },
          get innerHTML() {
            return String(text)
              .replace(/&/g, "&amp;")
              .replace(/</g, "&lt;")
              .replace(/>/g, "&gt;");
          },
        };
      },
      body: {
        classList: {
          add() {},
          remove() {},
        },
      },
    },
    window: {
      marked: null,
      hljs: null,
      addEventListener(type, handler) {
        windowListeners.set(type, handler);
      },
      removeEventListener(type) {
        windowListeners.delete(type);
      },
    },
    alert() {},
    confirm() {
      return true;
    },
    Vue: {
      createApp(options) {
        capturedOptions = options;
        return {
          mount() {},
        };
      },
    },
  };

  vm.runInNewContext(script, context, { filename: "script.js" });
  return { options: capturedOptions, context, windowListeners };
}

function createVm(overrides = {}) {
  const { options, context } = loadAppOptions();
  const state = {
    ...options.data(),
    $refs: {
      fileInput: { value: "" },
    },
    $nextTick(callback) {
      callback?.();
    },
    ...overrides,
  };

  for (const [name, fn] of Object.entries(options.methods || {})) {
    state[name] = fn.bind(state);
  }

  for (const [name, getter] of Object.entries(options.computed || {})) {
    Object.defineProperty(state, name, {
      enumerable: true,
      get() {
        return getter.call(state);
      },
    });
  }

  return { vm: state, context };
}

const checks = [
  {
    name: "uses the new centered floating shell instead of the old sidebar wrapper",
    run() {
      assert.match(index, /class="app-shell"/);
      assert.match(index, /class="top-nav"/);
      assert.match(index, /class="workspace"/);
      assert.match(index, /class="composer-dock"/);
      assert.doesNotMatch(index, /class="sidebar"/);
      assert.doesNotMatch(index, /class="app-wrapper"/);
    },
  },
  {
    name: "uses logo-horse.svg branding and removes old emoji branding",
    run() {
      assert.match(index, /logo-horse\.svg/);
      assert.match(css, /\.brand-mark/);
      assert.doesNotMatch(index, /🐱/u);
    },
  },
  {
    name: "uses ink and paper design tokens instead of the old pink palette",
    run() {
      assert.match(css, /--ink-bg:/);
      assert.match(css, /--paper:/);
      assert.match(css, /--cinnabar:/);
      assert.doesNotMatch(css, /--primary-color:\s*#ff9e99/i);
      assert.doesNotMatch(css, /--bg-color:\s*#fff5f5/i);
    },
  },
  {
    name: "keeps backend-facing API routes intact",
    run() {
      assert.match(backendApi, /include_router/);
      assert.match(backendRouterAuth, /\/auth\/register/);
      assert.match(backendRouterAuth, /\/auth\/login/);
      assert.match(backendRouterAuth, /\/auth\/me/);
      assert.match(backendRouterChat, /\/chat/);
      assert.match(backendRouterChat, /request\.context_files/);
      assert.match(backendRouterChat, /run_chat_stream/);
      assert.match(backendRouterSessions, /\/sessions/);
      assert.match(backendRouterSessions, /storage\./);
      assert.match(backendRouterDocuments, /\/documents\/upload/);
      assert.match(backendRouterDocuments, /DocumentService/);
      assert.match(backendRouterDocuments, /get_document_service/);
      [
        "/auth/me",
        "/auth/login",
        "/auth/register",
        "/chat/stream",
        "/sessions",
        "/documents",
        "/documents/upload",
      ].forEach((route) => assert.ok(script.includes(route), `missing ${route}`));
    },
  },
  {
    name: "buffers streaming output and avoids deep message watchers",
    run() {
      assert.match(script, /streamBuffer/);
      assert.match(script, /scheduleStreamFlush/);
      assert.match(script, /flushStreamBuffer/);
      assert.match(script, /streamFlushTimer/);
      assert.match(script, /streamFlushIntervalMs/);
      assert.match(script, /scheduleScrollToBottom/);
      assert.doesNotMatch(script, /deep:\s*true/);
      assert.doesNotMatch(index, /parseMarkdown\(msg\.text\)/);
    },
  },
  {
    name: "exposes history and knowledge as floating panels",
    run() {
      assert.match(index, /history-panel/);
      assert.match(index, /knowledge-view/);
      assert.match(script, /handleKnowledge/);
    },
  },
  {
    name: "supports global file picking from chat, page-level drop zone, and a top upload progress bar",
    run() {
      assert.match(index, /type="file"[^>]*multiple/);
      assert.match(index, /ref="globalFileInput"/);
      assert.match(index, /composer-upload-tray/);
      assert.match(index, /class="upload-actions"/);
      assert.match(index, /upload-trigger/);
      assert.match(index, /fa-cloud-arrow-up/);
      assert.match(index, /@click="triggerUploadPicker"/);
      assert.match(index, /:disabled="!selectedFiles.length \|\| isUploading"/);
      assert.match(index, /upload-status-bar/);
      assert.match(index, /page-drop-overlay/);
      assert.match(script, /maxUploadFiles/);
      assert.match(script, /queueSelectedFiles/);
      assert.match(script, /handleWindowDrop/);
      assert.match(script, /triggerUploadPicker/);
      assert.match(script, /pendingContextFiles/);
      assert.match(script, /context_files/);
      assert.match(css, /\.upload-actions\s*\{[\s\S]*display:\s*inline-grid;[\s\S]*grid-template-columns:\s*repeat\(2,\s*minmax\(0,\s*1fr\)\)/);
      assert.match(css, /\.upload-actions \.ghost-action\s*\{[\s\S]*width:\s*142px;/);
      assert.match(css, /\.upload-actions \.ghost-action:hover:not\(:disabled\)/);
      assert.match(css, /\.selected-files > \.primary-action\s*\{\s*display:\s*none;/);
    },
  },
  {
    name: "limits queued files to the configured maximum",
    run() {
      const { vm } = createVm();
      const files = Array.from({ length: 7 }, (_, idx) => ({
        name: `doc-${idx + 1}.pdf`,
        size: idx + 1,
        lastModified: idx + 100,
      }));

      vm.queueSelectedFiles(files);

      assert.equal(vm.selectedFiles.length, vm.maxUploadFiles);
      assert.equal(vm.selectedFiles[0].name, "doc-1.pdf");
      assert.equal(vm.selectedFiles.at(-1).name, `doc-${vm.maxUploadFiles}.pdf`);
    },
  },
  {
    name: "dropping files on the page queues them and starts upload in chat view",
    async run() {
      const { vm } = createVm({
        token: "token",
        currentUser: { username: "admin", role: "admin" },
        activeView: "chat",
      });

      let uploadTriggered = false;
      vm.uploadDocument = async () => {
        uploadTriggered = true;
      };

      await vm.handleWindowDrop({
        preventDefault() {},
        dataTransfer: {
          files: [{ name: "dragged.pdf", size: 1, lastModified: 1 }],
        },
      });

      assert.equal(vm.selectedFiles.length, 1);
      assert.equal(vm.selectedFiles[0].name, "dragged.pdf");
      assert.equal(uploadTriggered, true);
    },
  },
  {
    name: "clicking the chat upload trigger opens the shared file picker",
    run() {
      const { vm } = createVm({
        token: "token",
        currentUser: { username: "admin", role: "admin" },
      });
      let clicked = false;
      vm.$refs.globalFileInput = {
        click() {
          clicked = true;
        },
      };

      vm.triggerUploadPicker();

      assert.equal(clicked, true);
    },
  },
  {
    name: "uploaded files remain pending until the current chat turn consumes them as context",
    async run() {
      const { vm } = createVm({
        token: "token",
        currentUser: { username: "admin", role: "admin" },
        selectedFiles: [{ name: "manual.pdf", size: 1, lastModified: 1 }],
      });
      vm.uploadSingleFile = async () => ({ filename: "manual.pdf", message: "ok" });
      vm.loadDocuments = async () => {};

      await vm.uploadDocument();

      assert.deepEqual(Array.from(vm.pendingContextFiles.map((file) => file.filename)), ["manual.pdf"]);

      let streamedOptions = null;
      vm.userInput = "总结这份文档";
      vm.resetTextareaHeight = () => {};
      vm.scheduleScrollToBottom = () => {};
      vm.streamChatToBotSlot = async (_text, _idx, options) => {
        streamedOptions = options;
        return true;
      };

      await vm.handleSend();

      assert.deepEqual(Array.from(streamedOptions.contextFiles), ["manual.pdf"]);
      assert.equal(vm.pendingContextFiles.length, 0);
    },
  },
  {
    name: "sending while upload is active waits and uses that file in the current turn",
    async run() {
      const { vm } = createVm({
        token: "token",
        currentUser: { username: "admin", role: "admin" },
        isUploading: true,
      });
      let releaseUpload = null;
      vm.waitForActiveUpload = () =>
        new Promise((resolve) => {
          releaseUpload = () => {
            vm.addPendingContextFile({ filename: "slow.pdf" });
            vm.isUploading = false;
            resolve();
          };
        });
      let streamedOptions = null;
      vm.userInput = "分析这个文档";
      vm.resetTextareaHeight = () => {};
      vm.scheduleScrollToBottom = () => {};
      vm.streamChatToBotSlot = async (_text, _idx, options) => {
        streamedOptions = options;
        return true;
      };

      const sendPromise = vm.handleSend();
      assert.equal(streamedOptions, null);
      releaseUpload();
      await sendPromise;

      assert.deepEqual(Array.from(streamedOptions.contextFiles), ["slow.pdf"]);
      assert.equal(vm.pendingContextFiles.length, 0);
    },
  },
  {
    name: "failed chat streaming keeps attached context files available for retry",
    async run() {
      const { vm } = createVm({
        token: "token",
        currentUser: { username: "admin", role: "admin" },
        pendingContextFiles: [{ filename: "retry.pdf", addedAt: 1 }],
      });
      let streamedOptions = null;
      vm.userInput = "分析这个文档";
      vm.resetTextareaHeight = () => {};
      vm.scheduleScrollToBottom = () => {};
      vm.streamChatToBotSlot = async (_text, _idx, options) => {
        streamedOptions = options;
        return false;
      };

      await vm.handleSend();

      assert.deepEqual(Array.from(streamedOptions.contextFiles), ["retry.pdf"]);
      assert.deepEqual(Array.from(vm.pendingContextFiles.map((file) => file.filename)), ["retry.pdf"]);
    },
  },
  {
    name: "computes a real upload progress percentage across the queued files",
    run() {
      const { vm } = createVm({
        selectedFiles: [
          { name: "a.pdf", size: 10, lastModified: 1 },
          { name: "b.pdf", size: 20, lastModified: 2 },
        ],
        completedUploads: 1,
        currentUploadPercent: 50,
      });

      assert.equal(vm.uploadProgressPercent, 75);
    },
  },
  {
    name: "threads attached context files through backend RAG filtering",
    run() {
      assert.match(backendSchemas, /context_files:\s*Optional\[List\[str\]\]/);
      assert.match(backendRouterChat, /request\.context_files/);
      assert.match(backendAgent, /context_files/);
      assert.match(backendTools, /set_rag_context_files/);
      assert.match(backendTools, /run_rag_graph\(query,\s*context_files=/);
      assert.match(backendRagPipeline, /context_files/);
      assert.match(backendRagRetrieval, /filename in \[/);
      assert.match(backendRagUtils, /retrieve_context_documents/);
      assert.match(backendRagPipeline, /attached_context_chunks/);
      assert.match(backendAgent, /_with_retrieved_context_instruction/);
      assert.match(backendAgent, /model_instance\.astream/);
      assert.match(backendRagPipeline, /JSON/i);
      assert.match(backendRagPipeline, /grade_error/);
    },
  },
];

const failures = [];

for (const check of checks) {
  try {
    await check.run();
    console.log(`PASS ${check.name}`);
  } catch (error) {
    failures.push({ check, error });
    console.error(`FAIL ${check.name}`);
    console.error(error.message);
  }
}

if (failures.length) {
  process.exitCode = 1;
}
