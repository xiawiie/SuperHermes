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
    name: "supports multi-file upload selection, page-level drop zone, and a top upload progress bar",
    run() {
      assert.match(index, /type="file"[^>]*multiple/);
      assert.match(index, /upload-status-bar/);
      assert.match(index, /page-drop-overlay/);
      assert.match(script, /maxUploadFiles/);
      assert.match(script, /queueSelectedFiles/);
      assert.match(script, /handleWindowDrop/);
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
    name: "dropping files on the page queues them and starts upload in knowledge view",
    async run() {
      const { vm } = createVm({
        token: "token",
        currentUser: { username: "admin", role: "admin" },
        activeView: "knowledge",
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
