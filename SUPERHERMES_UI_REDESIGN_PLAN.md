# SuperHermes UI Redesign Design And Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 保留 SuperHermes 后端功能和接口，重构前端为 Claude.ai 式居中悬浮布局，并形成“玄墨宣纸”的水墨视觉系统、像素印章 logo、流畅的聊天体验和更快的页面响应速度。

**Architecture:** 不迁移项目技术栈，不引入新的构建系统。继续使用当前 `frontend/index.html` + `frontend/style.css` + `frontend/script.js` 的 Vue 3 CDN 单页结构，在现有 FastAPI 静态挂载方式下完成视觉重构、组件分层和渲染性能优化。后端 API、认证、RAG、文档管理、会话存储全部作为稳定边界保留。

**Tech Stack:** FastAPI static serving, Vue 3 CDN, vanilla CSS, vanilla JavaScript, `marked`, `highlight.js`, Font Awesome CDN.

---

## 1. Design Source And Direction

### 1.1 User Preference

用户明确偏好：

- Claude.ai 类布局。
- B 方案：居中悬浮布局。
- 对话区域居中浮动。
- 顶部悬浮导航栏。
- 水墨风。
- Claude 风格，但不是直接复制。
- Logo 需要类似 Claude 的像素风格，并且更帅。
- 当前页面反应慢，需要优化性能。
- 后端功能必须保留。
- UI 设计、布局、动效、特效可以大胆重构。

### 1.2 Reference Page Learning

参考页面 `http://localhost:53042/` 的可学习部分：

- 顶部轻量状态栏提供稳定页面锚点。
- 主内容区居中，视觉重心明确。
- 大预览卡片把布局意图直接表达出来。
- 底部固定提示条形成三段式结构。
- 不使用传统左侧导航，页面更接近 Claude.ai 的中心对话体验。

不应照搬的部分：

- 当前配色偏演示稿，缺少最终产品质感。
- 卡片圆角、边框、阴影较普通，有默认组件感。
- 三卡片展示适合方案选择，不适合聊天产品主界面。
- 纯黑方案过硬，水墨感不足。
- 朱砂色方案方向对，但红色使用过重。
- 月白方案高级感较好，但缺少“帅”的品牌识别和深度。

### 1.3 Final Design Direction: 玄墨宣纸

最终方向定义为 **玄墨宣纸**：

- Claude.ai 的克制布局。
- 水墨留白与宣纸材质。
- 砚台深色背景，而不是普通黑。
- 朱砂只作为关键动作和印章点缀。
- Bot 消息轻边界，用户消息清晰但不喧宾夺主。
- 像素印章 logo 作为品牌记忆点。
- 视觉上要安静、锋利、专业，不再可爱粉色。

---

## 2. Non-Negotiable Backend Boundary

后端功能不可破坏。前端改造必须继续兼容以下接口和行为。

### 2.1 Auth

- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/me`
- Bearer token 存储在 `localStorage.accessToken`
- 当前用户结构继续使用 `{ username, role }`
- `role === "admin"` 时显示知识库管理入口

### 2.2 Chat

- `POST /chat/stream`
- 请求体继续使用：

```json
{
  "message": "用户输入",
  "session_id": "session_..."
}
```

- SSE 事件类型继续支持：
  - `content`
  - `rag_step`
  - `trace`
  - `error`
  - `[DONE]`

### 2.3 Session History

- `GET /sessions`
- `GET /sessions/{session_id}`
- `DELETE /sessions/{session_id}`

### 2.4 Admin Documents

- `GET /documents`
- `POST /documents/upload`
- `DELETE /documents/{filename}`
- 文件类型继续支持 `.pdf`, `.doc`, `.docx`, `.xls`, `.xlsx`

### 2.5 Stop Generation

- 前端必须继续使用 `AbortController.abort()` 中断当前流式请求。
- UI 必须在生成中把发送按钮切换为停止按钮。

---

## 3. Current Frontend Problems

### 3.1 Performance Problems

当前性能瓶颈集中在 `frontend/script.js` 和 `frontend/index.html`：

- 每个 SSE token 到达都会修改 `messages[botMsgIdx].text`。
- `index.html` 中使用 `v-html="msg.isUser ? escapeHtml(msg.text) : parseMarkdown(msg.text)"`，导致机器人消息每次渲染都重新执行 `marked.parse()`。
- `script.js` 使用 `watch: { messages: { deep: true } }`，任何 token 更新都会触发滚动。
- 每个流式 chunk 循环末尾都执行 `$nextTick(() => scrollToBottom())`。
- 完整 RAG trace 数据挂在响应式消息对象上，展开前也承担响应式追踪成本。
- 历史、知识库、对话都挤在同一个大模板中，DOM 层级复杂。

### 3.2 UX Problems

- 旧 UI 是粉色可爱风，与用户目标不符。
- 左侧 sidebar 让页面不像 Claude.ai。
- 移动端直接隐藏 sidebar，功能不可完整访问。
- 历史侧栏右侧抽屉较传统，不够轻。
- 文档管理像后台表单，没有产品气质。
- RAG trace 展示信息很多，但视觉层级不清。
- Logo 使用 emoji，不具备品牌识别。

### 3.3 Text And Encoding Problems

当前读取到的工作区文件存在中文乱码风险。实现阶段必须用 UTF-8 保存 `index.html`、`style.css`、`script.js`，并验证浏览器中中文正常显示。

---

## 4. Layered Architecture

### Layer 0: Backend Contract Layer

责任：

- 保留所有后端接口路径。
- 保留 token 鉴权方式。
- 保留 SSE 协议解析。
- 保留管理员权限逻辑。
- 保留文档上传、删除、列表刷新。

设计原则：

- 前端重构只能改变调用方式的组织和 UI 显示，不能改变请求语义。
- 后端不做视觉重构相关变更。
- 若需要开发时静态资源不缓存，可保留 `backend/app.py` 的 no-cache middleware。

涉及文件：

- `backend/app.py`：仅作为静态资源服务边界，不主动修改业务逻辑。
- `backend/api.py`：只读参考，不修改。

### Layer 1: App Shell Layout Layer

责任：

- 替代旧 `.sidebar + .main-content` 布局。
- 建立 Claude.ai 风格的三段式结构：
  - 顶部悬浮导航栏。
  - 中央对话画布。
  - 底部悬浮输入框。

目标结构：

```html
<div id="app" class="app-shell">
  <header class="top-nav">...</header>
  <main class="workspace">
    <section class="auth-view">...</section>
    <section class="chat-view">...</section>
    <section class="knowledge-view">...</section>
    <aside class="history-panel">...</aside>
  </main>
  <div class="composer-dock">...</div>
</div>
```

布局规则：

- `.app-shell` 占满视口。
- `.top-nav` 固定在顶部中央，`position: fixed`。
- `.workspace` 是主滚动容器，负责内容滚动。
- `.chat-view` 最大宽度 `920px`，水平居中。
- `.composer-dock` 固定在底部中央，最大宽度与 `.chat-view` 对齐。
- 旧 sidebar 彻底移除，不隐藏在移动端。

### Layer 2: Visual Token Layer

责任：

- 用 CSS variables 定义完整视觉系统。
- 把颜色、间距、半径、阴影、动效、排版集中管理。

推荐 tokens：

```css
:root {
  --ink-bg: #14110d;
  --ink-bg-soft: #1b1712;
  --ink-surface: rgba(31, 27, 22, 0.78);
  --ink-surface-solid: #211d18;
  --paper: #eee6d8;
  --paper-soft: #d8cec0;
  --paper-muted: #9c9285;
  --paper-faint: rgba(238, 230, 216, 0.08);
  --cinnabar: #b84a35;
  --cinnabar-strong: #d25a40;
  --bamboo: #7f8f73;
  --line: rgba(238, 230, 216, 0.1);
  --line-strong: rgba(238, 230, 216, 0.18);
  --shadow-soft: 0 18px 50px rgba(0, 0, 0, 0.34);
  --shadow-tight: 0 8px 24px rgba(0, 0, 0, 0.28);
  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 16px;
  --nav-height: 48px;
  --content-max: 920px;
  --composer-max: 920px;
}
```

Typography：

- 不新增字体依赖。
- 中文优先使用系统字体栈：

```css
font-family:
  "Noto Sans SC",
  "Microsoft YaHei",
  "PingFang SC",
  "Hiragino Sans GB",
  system-ui,
  sans-serif;
```

- 品牌字可使用更有书卷感的 serif fallback：

```css
font-family:
  "Noto Serif SC",
  "Songti SC",
  "SimSun",
  serif;
```

背景设计：

- 深砚台底色。
- 使用 CSS radial-gradient 做极轻微墨晕。
- 使用 repeating-linear-gradient 做低透明纸纹。
- 不使用装饰性 orb、紫蓝渐变或大面积 bokeh。

### Layer 3: Brand And Logo Layer

责任：

- 替换猫 emoji。
- 创建 SuperHermes 像素印章 logo。
- Logo 通过 HTML/CSS grid 实现，不引入图片。

Logo 结构：

```html
<div class="pixel-seal" aria-hidden="true">
  <span class="p p-..."></span>
  ...
</div>
```

设计规格：

- 采用 7x7 或 9x9 网格。
- 猫耳轮廓 + M 字中心结构。
- 主色使用 `--paper-muted` 或 `--paper`。
- 右下角 1-2 个像素使用 `--cinnabar` 作为印章点。
- 顶部 nav 尺寸 `28px`。
- 欢迎页尺寸 `72px`。
- 不循环动画，只在页面初始出现时执行一次像素点亮。

### Layer 4: Navigation Layer

责任：

- 用顶部悬浮导航替代旧 sidebar。
- 提供所有原侧栏功能入口。

导航项：

- Logo + `SuperHermes`
- `新对话`
- `历史`
- `知识库`，仅管理员可见
- 当前用户 badge
- `退出`

交互规则：

- `新对话` 调用 `handleNewChat()`。
- `历史` 打开 `history-panel`。
- `知识库` 设置 `activeView = "knowledge"` 并调用 `loadDocuments()`。
- `退出` 调用 `handleLogout()`。
- 当前 active 项使用朱砂细底色，不使用高饱和大面积背景。

移动端规则：

- 顶部导航变为可横向滚动的紧凑 pill bar。
- 用户名在窄屏只显示头像或首字母。
- 所有主要功能必须可访问，不能隐藏。

### Layer 5: Auth Layer

责任：

- 登录注册界面和未登录状态。
- 保留原认证逻辑。

视觉设计：

- 居中浮动 auth panel。
- 面板宽度 `min(440px, calc(100vw - 32px))`。
- 背景为 `rgba(31, 27, 22, 0.82)`。
- 顶部显示大尺寸 pixel seal。
- 标题使用 `SuperHermes`，副标题使用简洁中文。

文案：

- 登录标题：`进入 SuperHermes`
- 注册标题：`创建 SuperHermes 账号`
- 副标题：`登录后可以使用对话、历史记录与知识库检索。`
- 用户名 placeholder：`用户名`
- 密码 placeholder：`密码`
- 管理员邀请码 placeholder：`管理员邀请码`
- 登录按钮：`登录`
- 注册按钮：`注册`

错误处理：

- 继续使用现有 `alert()` 可以先保留。
- 后续优化可以改成顶部 toast，但第一轮不引入 toast 系统，避免范围膨胀。

### Layer 6: Chat Reading Layer

责任：

- 消息流展示。
- 欢迎态展示。
- Markdown 和代码块展示。

欢迎态：

- 居中显示大 pixel seal。
- 标题：`有什么想问 SuperHermes？`
- 副标题：`可以检索知识库、分析文档、解释代码，或继续一段已有对话。`
- 三个 prompt chip：
  - `解释这个知识库里的核心结论`
  - `帮我整理一份技术方案`
  - `分析 RAG 检索为什么不准`

消息设计：

- Bot 消息：
  - 左对齐。
  - 默认无大气泡，使用自然文本块。
  - 左侧可加一条淡墨竖线，区分 AI 回复。
  - 最大宽度 `82%`。
  - Markdown 段落行高 `1.72`。

- User 消息：
  - 右对齐。
  - 使用深墨棕或朱砂深色气泡。
  - 最大宽度 `68%`。
  - 边角半径 `14px`，尾角不夸张。

- Code block：
  - 深色代码块。
  - 边框使用 `--line`。
  - 横向滚动。
  - 不使用亮白背景。

### Layer 7: Composer Layer

责任：

- 输入框、附件按钮、发送按钮、停止按钮。
- 保留 `Enter` 发送、`Shift+Enter` 换行、中文输入法 composition 保护。

视觉设计：

- 底部居中悬浮。
- 容器最大宽度 `920px`。
- 背景 `rgba(31, 27, 22, 0.86)`。
- 边框 `1px solid var(--line-strong)`。
- 获得焦点后边框变为 `rgba(184, 74, 53, 0.55)`。
- 发送按钮使用朱砂印章样式。
- 停止按钮使用深红描边样式。

交互规则：

- 空输入时发送按钮 disabled。
- 生成中显示停止按钮。
- `handleStop()` 保持 `AbortController.abort()`。
- 输入框自动高度最大 `160px`。
- 输入完成后 reset 高度。

### Layer 8: History Layer

责任：

- 历史会话列表、加载、删除。

视觉设计：

- 改为居中浮层或顶部导航下方 popover。
- 不再使用右侧满高抽屉。
- 宽度 `min(680px, calc(100vw - 32px))`。
- 最大高度 `70vh`。
- 内部滚动。

会话项结构：

- 左侧：会话 ID 或后续可推导标题。
- 中间：消息数。
- 右侧：更新时间、删除按钮。

交互规则：

- 点击会话项调用 `loadSession(session.session_id)`。
- 点击删除按钮调用 `deleteSession(session.session_id)`。
- 当前会话使用淡朱砂左边框。
- 删除按钮默认低透明，hover 或 focus 显示。

### Layer 9: Knowledge Base Layer

责任：

- 管理员文档上传、文档列表、删除。

视觉设计：

- 不再叫“设置”，改成“知识库”。
- 主体是居中工作台。
- 顶部信息：
  - 标题：`知识库`
  - 描述：`上传文档后，SuperHermes 会切分、向量化并写入检索库。`

上传区域：

- 宣纸投递区视觉。
- 边框为淡墨虚线。
- 选择文件按钮为朱砂小按钮。
- 选中文件后显示文件名、类型、开始上传按钮。

文档列表：

- 使用紧凑行，不使用大卡片。
- 每行显示：
  - 文件图标。
  - 文件名。
  - 文件类型。
  - chunk 数量。
  - 删除按钮。

权限：

- 非管理员不显示知识库入口。
- 如果非管理员通过状态误入知识库视图，调用 `handleSettings()` 的权限检查或直接回到聊天视图。

### Layer 10: RAG Trace Layer

责任：

- 展示实时 RAG 步骤。
- 展示完整检索过程。
- 降低默认渲染成本。

实时思考态：

- Bot 消息未输出 content 前，显示“墨迹时间线”。
- 默认显示当前步骤 label。
- 已发生步骤以小点和细线纵向排列。

完成后 trace：

- 默认只显示一个 summary row：
  - `检索过程`
  - `工具：search_knowledge_base` 或 `未使用`
  - `展开`

展开后才渲染完整细节：

- 检索阶段。
- 相关性评分。
- 重写策略。
- rerank 状态。
- auto-merging 状态。
- 初次检索结果。
- 重写后检索结果。
- 向量检索结果。

性能规则：

- `ragTrace` 可以保留在消息对象中。
- 大段来源列表必须通过 `details[open]` 或 Vue 状态控制后渲染。
- 默认不渲染 chunk text。

### Layer 11: State And Rendering Performance Layer

责任：

- 解决页面慢的问题。
- 降低 token 流式输出期间 DOM 和 Markdown 成本。

必须实现的优化：

1. Token buffer

当前逻辑每个 token 直接写入消息文本。改为：

```js
this.streamBuffer += data.content;
this.scheduleStreamFlush(botMsgIdx);
```

2. requestAnimationFrame flush

```js
scheduleStreamFlush(botMsgIdx) {
  if (this.streamFlushScheduled) return;
  this.streamFlushScheduled = true;
  requestAnimationFrame(() => {
    this.flushStreamBuffer(botMsgIdx);
  });
}
```

3. Cached markdown HTML

每条 bot 消息增加：

```js
{
  text: "",
  html: "",
  isUser: false,
  isThinking: true,
  ragTrace: null,
  ragSteps: []
}
```

流式期间低频更新 `html`，结束时强制最终更新。

4. Remove deep watcher

删除：

```js
watch: {
  messages: {
    deep: true
  }
}
```

改为在明确节点调用：

- 新消息 push 后。
- stream flush 后。
- stream 完成后。
- 加载历史后。

5. Throttled scroll

```js
scheduleScrollToBottom() {
  if (this.scrollScheduled) return;
  this.scrollScheduled = true;
  requestAnimationFrame(() => {
    this.scrollScheduled = false;
    this.scrollToBottom();
  });
}
```

6. Lazy trace rendering

完整 trace 只在用户展开时渲染。

### Layer 12: Motion Layer

责任：

- 提供高级但低成本的动效。

动效规则：

- 只使用 `opacity` 和 `transform` 作为主要动画属性。
- 不对大面积 `filter: blur()` 做动画。
- 不做持续循环背景动画。
- 尊重 `prefers-reduced-motion`。

具体动效：

- `.top-nav` 初始从 `translateY(-8px)` 到 `0`。
- `.welcome-screen` 初始从 `translateY(8px)` 到 `0`。
- `.message` 初始从 `translateY(6px)` 到 `0`。
- `.pixel-seal .p` 初次加载分批点亮。
- `.thinking-trace-line` 新增时淡入。
- 发送按钮 hover 轻微下压，不旋转。

### Layer 13: Responsive And Accessibility Layer

责任：

- 移动端完整可用。
- 键盘和屏幕阅读器基础可用。

响应式规则：

- `>= 1024px`：完整居中布局，顶部 nav 宽度自适应。
- `768px - 1023px`：内容最大宽度 `calc(100vw - 40px)`。
- `< 768px`：
  - 顶部 nav 左右留白 `12px`。
  - nav 横向滚动。
  - chat padding 缩小。
  - user message 最大宽度 `86%`。
  - composer 距底 `10px`。
  - history panel 宽度 `calc(100vw - 24px)`。

Accessibility：

- 所有 icon button 必须有 `title` 或 `aria-label`。
- 可点击 div 应改为 button 或增加键盘可访问处理。
- focus state 使用清晰边框。
- 文本对比度保持可读。

---

## 5. File Responsibility Map

### `frontend/index.html`

职责：

- 页面 DOM 结构。
- Vue 模板。
- 顶部导航、认证、聊天、历史、知识库、输入框结构。

修改重点：

- 删除旧 sidebar。
- 重建 app shell。
- 把设置面板改为知识库视图。
- 消息 v-html 改为使用缓存字段 `msg.html`。
- trace 详细列表改为懒渲染。

### `frontend/style.css`

职责：

- 完整视觉系统。
- 水墨背景、top nav、chat view、composer、auth、history、knowledge、message、trace、responsive。

修改重点：

- 全量替换旧粉色可爱主题。
- 使用 tokens。
- 实现 pixel seal。
- 实现响应式布局。
- 实现 motion 和 reduced-motion。

### `frontend/script.js`

职责：

- Vue state。
- Auth flow。
- Chat streaming。
- Session flow。
- Document flow。
- Rendering performance helpers。

修改重点：

- 增加 `activeView` 或统一替代 `activeNav`。
- 增加 stream buffer 和 flush 调度。
- 增加 markdown cache。
- 删除 messages deep watcher。
- 增加 scroll throttle。
- 增加 trace 展开状态。

### `backend/app.py`

职责：

- FastAPI app。
- 静态资源挂载。
- 开发期 no-cache。

修改重点：

- 本计划不要求修改。
- 如果验证时静态资源缓存导致旧样式残留，可以保留当前 no-cache middleware。

---

## 6. Implementation Phases

### Phase 1: Safety Baseline And Encoding

目标：先保证能回退、能观察、中文不乱码。

- [ ] Step 1: 查看工作区状态。

```powershell
git -C "C:\Users\goahe\Desktop\Project\SuperHermes" status --short
```

Expected:

- 看到已有未提交改动。
- 不回滚用户已有改动。

- [ ] Step 2: 读取前端三文件，确认当前结构。

```powershell
Get-Content -Raw "C:\Users\goahe\Desktop\Project\SuperHermes\frontend\index.html"
Get-Content -Raw "C:\Users\goahe\Desktop\Project\SuperHermes\frontend\style.css"
Get-Content -Raw "C:\Users\goahe\Desktop\Project\SuperHermes\frontend\script.js"
```

Expected:

- 能看到 Vue template。
- 能看到 `handleSend()`、`parseMarkdown()`、`watch.messages`。

- [ ] Step 3: 修改前确保保存为 UTF-8。

Implementation rule:

- 使用 editor 或 patch 保持 UTF-8。
- 修改后浏览器中必须显示正常中文。

### Phase 2: App Shell Layout

目标：把旧侧栏布局替换为居中悬浮骨架。

- [ ] Step 1: 在 `index.html` 中把根容器改为 `.app-shell`。
- [ ] Step 2: 新增 `.top-nav`。
- [ ] Step 3: 新增 `.workspace`。
- [ ] Step 4: 聊天区域改为 `.chat-view`。
- [ ] Step 5: 输入区域改为 `.composer-dock`。
- [ ] Step 6: 确认 `isAuthenticated` 为 false 时只显示 auth view。
- [ ] Step 7: 确认 `activeView === "chat"` 时显示聊天。
- [ ] Step 8: 确认 `activeView === "knowledge"` 且 `isAdmin` 时显示知识库。

Acceptance:

- 未登录时没有旧 sidebar。
- 登录后顶部 nav 出现。
- 对话区域水平居中。
- 输入框底部居中悬浮。

### Phase 3: Visual Token System

目标：建立玄墨宣纸视觉基础。

- [ ] Step 1: 在 `style.css` 顶部替换 `:root` tokens。
- [ ] Step 2: 改写 `body` 背景为深砚台 + 纸纹。
- [ ] Step 3: 设置全局字体、文本颜色、滚动条。
- [ ] Step 4: 删除旧粉色变量和相关依赖。

Acceptance:

- 页面不再出现粉色可爱主题。
- 背景是深水墨风。
- 文字对比度清晰。

### Phase 4: Pixel Seal Logo

目标：替换 emoji logo，建立品牌识别。

- [ ] Step 1: 在 `index.html` 中添加 `.pixel-seal` 结构。
- [ ] Step 2: 在 `style.css` 中实现 7x7 或 9x9 pixel grid。
- [ ] Step 3: 顶部 nav 使用小尺寸 logo。
- [ ] Step 4: 欢迎页使用大尺寸 logo。
- [ ] Step 5: 添加一次性点亮动画。

Acceptance:

- 页面不再出现 `🐱` 作为主 logo，改为小马形象。
- Logo 有马耳/印章的像素特征。
- Logo 在深色背景中清晰。

### Phase 5: Navigation

目标：所有原侧栏功能进入顶部导航。

- [ ] Step 1: `新对话` button 绑定 `handleNewChat()`。
- [ ] Step 2: `历史` button 绑定 `handleHistory()`。
- [ ] Step 3: `知识库` button 只在 `isAdmin` 时显示，绑定 `handleKnowledge()`。
- [ ] Step 4: 用户 badge 显示 username 和 role。
- [ ] Step 5: 退出 button 绑定 `handleLogout()`。
- [ ] Step 6: 更新 `script.js`，用 `activeView` 管理 `chat` 和 `knowledge`。

Acceptance:

- 新建会话正常。
- 历史面板正常打开。
- 管理员看到知识库入口。
- 普通用户看不到知识库入口。
- 退出后回到登录视图。

### Phase 6: Auth View

目标：登录注册变成居中浮动水墨面板。

- [ ] Step 1: 保留现有 `authForm` 和 `handleAuthSubmit()`。
- [ ] Step 2: 重写 auth panel HTML 结构。
- [ ] Step 3: 文案改为正常中文。
- [ ] Step 4: 样式改为深墨浮层。
- [ ] Step 5: 登录成功后设置 `activeView = "chat"`。

Acceptance:

- 登录流程正常。
- 注册流程正常。
- 管理员邀请码输入仍只在选择 admin 时显示。
- 中文无乱码。

### Phase 7: Chat Message UI

目标：实现 Claude 风阅读流。

- [ ] Step 1: 欢迎态改为 pixel seal + 标题 + prompt chips。
- [ ] Step 2: Bot 消息改为轻边界文本块。
- [ ] Step 3: User 消息改为右侧深色气泡。
- [ ] Step 4: Markdown 样式适配深色主题。
- [ ] Step 5: Code block 样式适配深色主题。

Acceptance:

- 消息不再像旧式圆润聊天气泡。
- Bot 回复可读性强。
- 用户消息清晰但不刺眼。
- 代码块不使用亮白背景。

### Phase 8: Composer UI

目标：底部输入框成为主操作区。

- [ ] Step 1: 保留 textarea 绑定。
- [ ] Step 2: 保留 composition 事件。
- [ ] Step 3: 保留 `Enter` 发送和 `Shift+Enter` 换行。
- [ ] Step 4: 发送按钮改为朱砂印章样式。
- [ ] Step 5: 生成中按钮改为停止样式。
- [ ] Step 6: 输入框 focus 状态使用朱砂边线。

Acceptance:

- 空输入不能发送。
- 生成中可以停止。
- 中文输入法不误发送。
- 输入框高度自动扩展。

### Phase 9: History Panel

目标：历史记录变成居中浮层。

- [ ] Step 1: `showHistorySidebar` 改名或继续复用为 `showHistoryPanel`。
- [ ] Step 2: HTML 类名改为 `.history-panel`。
- [ ] Step 3: 使用居中 overlay 或 popover。
- [ ] Step 4: 会话行显示 session_id、message_count、updated_at。
- [ ] Step 5: 删除按钮保留 stop propagation。

Acceptance:

- 点击历史按钮加载会话列表。
- 点击会话加载消息。
- 删除会话正常。
- 当前会话有视觉标识。

### Phase 10: Knowledge Base Workbench

目标：管理员文档管理改为知识库工作台。

- [ ] Step 1: `handleSettings()` 改名为 `handleKnowledge()` 或保留方法名但改 UI 文案。
- [ ] Step 2: `activeView = "knowledge"`。
- [ ] Step 3: 上传区重构为宣纸投递区。
- [ ] Step 4: 文档列表改为紧凑数据行。
- [ ] Step 5: 删除文档保留 confirm。

Acceptance:

- 管理员可加载文档列表。
- 选择文件后能上传。
- 上传后自动刷新列表。
- 删除文档后自动刷新列表。

### Phase 11: RAG Trace And Thinking UI

目标：保留 RAG 可观测能力，同时降低默认噪音和渲染成本。

- [ ] Step 1: 思考中状态显示当前步骤。
- [ ] Step 2: RAG 步骤用墨迹时间线展示。
- [ ] Step 3: 完成后 trace 默认折叠。
- [ ] Step 4: 展开后显示完整 trace。
- [ ] Step 5: 来源 chunk 文本只在展开状态渲染。

Acceptance:

- `rag_step` 到达时 UI 实时变化。
- `trace` 到达后可展开查看。
- 未使用工具时显示 `未使用`。
- 大段来源不会默认撑满页面。

### Phase 12: Streaming Performance

目标：解决页面反应慢。

- [ ] Step 1: 在 `data()` 中新增：

```js
streamBuffer: "",
streamFlushScheduled: false,
scrollScheduled: false
```

- [ ] Step 2: 新增 `scheduleStreamFlush(botMsgIdx)`。
- [ ] Step 3: 新增 `flushStreamBuffer(botMsgIdx, force = false)`。
- [ ] Step 4: `content` 事件只写 buffer，不直接频繁写 message text。
- [ ] Step 5: 每次 flush 后更新 `msg.html`。
- [ ] Step 6: 流结束和异常结束时强制 flush。
- [ ] Step 7: 删除 messages deep watcher。
- [ ] Step 8: 新增 `scheduleScrollToBottom()`。

Acceptance:

- 长回复输出时页面不卡顿明显下降。
- Markdown 仍能在最终结果正确渲染。
- 滚动仍跟随底部。
- 停止生成后显示已停止提示。

### Phase 13: Responsive Pass

目标：移动端完整可用。

- [ ] Step 1: 添加 `<768px` media query。
- [ ] Step 2: top nav 横向滚动。
- [ ] Step 3: chat content padding 缩小。
- [ ] Step 4: composer 宽度适配。
- [ ] Step 5: history 和 knowledge panel 宽度适配。

Acceptance:

- 手机宽度下可以登录、发送、打开历史。
- 管理员手机宽度下可以打开知识库。
- 按钮文字不溢出。
- 输入框不遮挡内容。

### Phase 14: Verification

目标：用实际运行验证功能和视觉。

- [ ] Step 1: 启动后端。

```powershell
uv run uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

Expected:

- 服务启动成功。
- 访问 `http://127.0.0.1:8000/` 能打开前端。

- [ ] Step 2: 验证未登录页面。

Expected:

- 显示水墨登录面板。
- 中文正常。
- 无旧粉色主题。

- [ ] Step 3: 验证登录。

Expected:

- 登录成功进入聊天页。
- 顶部导航出现。

- [ ] Step 4: 验证发送普通消息。

Expected:

- 用户消息右侧出现。
- Bot 流式输出。
- 页面不卡顿。

- [ ] Step 5: 验证停止生成。

Expected:

- 生成中按钮切换为停止。
- 点击停止后请求中断。
- Bot 消息显示已停止状态。

- [ ] Step 6: 验证历史。

Expected:

- 打开历史面板。
- 会话列表可加载。
- 点击可恢复会话。
- 删除可用。

- [ ] Step 7: 验证管理员知识库。

Expected:

- 管理员看到知识库入口。
- 文档列表可加载。
- 文件可选择并上传。
- 文档可删除。

- [ ] Step 8: 验证移动端。

Expected:

- 375px 宽度下导航、聊天、输入框可用。
- 没有文本重叠。

---

## 7. Acceptance Criteria

### Visual Acceptance

- 页面符合“玄墨宣纸”方向。
- 顶部悬浮导航明确。
- 对话区域居中。
- 输入框底部居中悬浮。
- Logo 是像素印章风，不再是 emoji。
- UI 不再是粉色可爱风。
- 朱砂色只作为关键点缀。
- 不出现紫色渐变、装饰 orb、过度玻璃拟态。

### Functional Acceptance

- 登录、注册正常。
- 当前用户获取正常。
- 退出正常。
- 新对话正常。
- 流式聊天正常。
- 停止生成正常。
- 历史列表正常。
- 会话加载正常。
- 会话删除正常。
- 管理员文档列表正常。
- 管理员文档上传正常。
- 管理员文档删除正常。
- 普通用户不能看到知识库入口。

### Performance Acceptance

- 长文本流式输出时页面明显比原来更流畅。
- 不再对 `messages` 使用 deep watcher。
- 不再每 token 执行完整 Markdown parse。
- 滚动被 requestAnimationFrame 节流。
- RAG trace 详情默认懒渲染。

### Code Acceptance

- 不新增依赖。
- 不引入构建系统。
- 不改后端业务接口。
- 文件仍为静态前端三件套。
- 中文文案 UTF-8 正常显示。
- CSS tokens 清晰。
- 旧 sidebar 相关 CSS 被删除或不再使用。

---

## 8. Risks And Mitigations

### Risk 1: Streaming Markdown Cache Might Lag During Generation

Mitigation:

- 流式期间可以每帧或每 50ms 更新一次 HTML。
- 流结束时强制最终 `marked.parse()`。
- 如果代码块在流式中未闭合，最终渲染后会恢复正确。

### Risk 2: RAG Trace Is Large

Mitigation:

- 默认只显示 summary。
- 来源 chunk 文本只在展开时渲染。
- 保留原始 `ragTrace` 数据，但减少默认 DOM。

### Risk 3: Existing Uncommitted Changes

Mitigation:

- 实施前先 `git status --short`。
- 不回滚用户改动。
- 对已改动文件做增量改造。
- 如果同一文件已有用户编辑，先读完整文件再改。

### Risk 4: Local Backend Dependencies May Not Be Running

Mitigation:

- 静态视觉可以通过 FastAPI 或直接静态打开检查。
- 完整功能验证需要 PostgreSQL、Redis、Milvus、模型环境变量。
- 若依赖未启动，最终报告必须说明未验证的后端功能。

### Risk 5: Dark Theme Readability

Mitigation:

- 主文字使用暖宣纸色。
- 次级文字不低于可读对比。
- 表单输入和代码块增加边框层级。
- 移动端增大触控目标。

---

## 9. Recommended Implementation Order

最推荐的实现顺序：

1. Phase 1: Safety Baseline And Encoding
2. Phase 2: App Shell Layout
3. Phase 3: Visual Token System
4. Phase 4: Pixel Seal Logo
5. Phase 5: Navigation
6. Phase 6: Auth View
7. Phase 7: Chat Message UI
8. Phase 8: Composer UI
9. Phase 12: Streaming Performance
10. Phase 11: RAG Trace And Thinking UI
11. Phase 9: History Panel
12. Phase 10: Knowledge Base Workbench
13. Phase 13: Responsive Pass
14. Phase 14: Verification

原因：

- 先搭骨架，再做视觉。
- 先保证聊天主路径，再做管理面板。
- 性能优化在主 UI 成型后进行，方便验证真实体验。
- RAG trace、历史、知识库属于二级功能，但不能遗漏。

---

## 10. Final Product Shape

完成后 SuperHermes 应该呈现为：

- 一个深色水墨质感的 AI 助手。
- 第一眼看到的是居中悬浮的 Claude 风界面。
- 顶部导航轻薄、克制。
- 中央对话区留白充足，阅读体验优先。
- 底部输入框稳定、强识别、响应快。
- Pixel seal logo 形成独立品牌感。
- RAG 检索过程可展开查看，但默认不打扰阅读。
- 管理员知识库像一个专业工作台，而不是临时设置页。
- 页面流式输出明显比原来更顺。

