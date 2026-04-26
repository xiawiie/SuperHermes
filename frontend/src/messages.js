(function () {
    function nextMessageId() {
        return "msg_" + Date.now() + "_" + Math.random().toString(16).slice(2);
    }

    function createUserMessage(text, contextFiles) {
        return {
            id: nextMessageId(),
            text,
            html: "",
            isUser: true,
            contextFiles: contextFiles || [],
        };
    }

    function createBotMessage(text, renderMarkdown, ragTrace) {
        const content = text || "";
        return {
            id: nextMessageId(),
            text: content,
            html: typeof renderMarkdown === "function" ? renderMarkdown(content) : content,
            isUser: false,
            isThinking: !content,
            ragTrace: ragTrace || null,
            ragSteps: [],
        };
    }

    window.SuperHermesMessages = {
        createUserMessage,
        createBotMessage,
    };
})();
