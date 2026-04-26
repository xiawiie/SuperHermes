(function () {
    function createAuthHeaders(token, extra) {
        const headers = { ...(extra || {}) };
        if (token) {
            headers.Authorization = `Bearer ${token}`;
        }
        return headers;
    }

    async function authFetch(token, onUnauthorized, url, options) {
        const opts = { ...(options || {}) };
        opts.headers = createAuthHeaders(token, opts.headers || {});
        const response = await fetch(url, opts);
        if (response.status === 401) {
            if (typeof onUnauthorized === "function") {
                onUnauthorized();
            }
            throw new Error("登录已过期，请重新登录");
        }
        return response;
    }

    window.SuperHermesApi = {
        createAuthHeaders,
        authFetch,
    };
})();
