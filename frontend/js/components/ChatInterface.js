/* --- Global Component: <chat-interface> ---
 *
 * index.html renders <chat-interface> once, inside the v-if="isAuthenticated"
 * wrapper, so this widget is available on every page after login. It is a
 * floating launcher plus a slide-up panel; the full-page AgriBot tab is
 * unchanged and keeps its own history.
 *
 * Talks to POST /api/ai/agribot on the Node backend, which proxies Groq.
 * That route is farmer-only, so other roles get told why up front instead of
 * discovering it through a 403.
 *
 * Loaded as a plain script after vue.min.js — no build step, ES5-friendly.
 */
(function () {
    'use strict';

    if (typeof Vue === 'undefined') {
        console.error('[ChatInterface] Vue is not loaded — check script order in index.html.');
        return;
    }

    // marked exposes parse() in v4+ and a bare callable in v3. Support both,
    // and degrade to escaped plain text if the CDN failed to load.
    function renderMarkdown(text) {
        try {
            // SANITISE before this reaches v-html. The text is LLM output, and an
            // LLM will reproduce whatever a user typed at it -- so a farmer who
            // pastes an <img onerror> into the chatbot gets it stored and then
            // rendered for whoever reads the thread, with the JWT sitting in
            // localStorage. marked does not strip HTML; it passes raw tags through
            // by design.
            const clean = (html) => (typeof DOMPurify !== 'undefined'
                ? DOMPurify.sanitize(html, { USE_PROFILES: { html: true } })
                // No DOMPurify means no safe render. Escape and show as plain
                // text rather than trusting it.
                : String(html).replace(/[&<>"']/g, (c) => ({
                    '&': '&amp;', '<': '&lt;', '>': '&gt;',
                    '"': '&quot;', "'": '&#39;',
                }[c])));

            if (typeof marked !== 'undefined') {
                if (typeof marked.parse === 'function') return clean(marked.parse(text));
                if (typeof marked === 'function') return clean(marked(text));
            }
        } catch (e) {
            console.warn('[ChatInterface] Markdown render failed:', e);
        }
        return escapeHtml(text).replace(/\n/g, '<br>');
    }

    function escapeHtml(str) {
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    Vue.component('chat-interface', {
        data: function () {
            return {
                open: false,
                input: '',
                messages: [],
                loading: false,
                error: null
            };
        },

        computed: {
            // The root Vue instance owns auth state and the API base URL.
            root: function () { return this.$root; },
            apiUrl: function () { return this.$root.apiUrl; },
            token: function () { return this.$root.token; },
            role: function () { return this.$root.role; },
            isFarmer: function () { return this.$root.role === 'farmer'; },

            storageKey: function () {
                var id = this.$root.currentUserIdentifier || 'anon';
                return 'pestivid_chatwidget_' + id;
            },

            // Unanswered farmer questions are the only thing worth badging.
            unreadHint: function () {
                return !this.open && this.messages.length === 0;
            }
        },

        watch: {
            messages: {
                deep: true,
                handler: function () {
                    this.persist();
                    this.scrollToBottom();
                }
            },
            open: function (isOpen) {
                if (isOpen) this.scrollToBottom();
            }
        },

        created: function () {
            this.restore();
        },

        methods: {
            toggle: function () {
                this.open = !this.open;
                if (this.open && this.messages.length === 0) this.greet();
            },

            greet: function () {
                if (!this.isFarmer) {
                    this.messages.push({
                        id: 'sys_role',
                        sender: 'bot',
                        text: 'AgriBot is available to farmer accounts. You are signed in as **' +
                              (this.role || 'unknown') + '**, so chat is read-only here.',
                        ts: new Date().toISOString()
                    });
                    return;
                }
                this.messages.push({
                    id: 'sys_hello',
                    sender: 'bot',
                    text: "Hi — I'm AgriBot. Ask me about crop disease, pests, soil, irrigation, or anything on the PestiVid platform.",
                    ts: new Date().toISOString()
                });
            },

            send: function () {
                var question = this.input.trim();
                if (!question || this.loading) return;
                if (!this.isFarmer) {
                    this.error = 'Only farmer accounts can use AgriBot.';
                    return;
                }

                this.error = null;
                this.messages.push({
                    id: 'u_' + Date.now(),
                    sender: 'user',
                    text: question,
                    ts: new Date().toISOString()
                });
                this.input = '';
                this.loading = true;

                var self = this;
                axios.post(
                    this.apiUrl + '/ai/agribot',
                    { question: question },
                    { headers: { Authorization: 'Bearer ' + this.token } }
                ).then(function (res) {
                    self.messages.push({
                        id: 'b_' + Date.now(),
                        sender: 'bot',
                        text: res.data.answer,
                        ts: new Date().toISOString()
                    });
                }).catch(function (err) {
                    // Surface the server's own reason — usually a missing
                    // GROQ_API_KEY or an expired token — rather than a generic
                    // failure the user cannot act on.
                    var msg = (err.response && err.response.data && err.response.data.message) ||
                              err.message || 'Request failed.';
                    self.error = msg;
                    self.messages.push({
                        id: 'e_' + Date.now(),
                        sender: 'bot',
                        isError: true,
                        text: 'Could not reach AgriBot: ' + msg,
                        ts: new Date().toISOString()
                    });
                }).then(function () {
                    self.loading = false;
                });
            },

            clear: function () {
                this.messages = [];
                this.error = null;
                try { localStorage.removeItem(this.storageKey); } catch (e) {}
                this.greet();
            },

            persist: function () {
                try {
                    localStorage.setItem(this.storageKey, JSON.stringify(this.messages.slice(-60)));
                } catch (e) {
                    console.warn('[ChatInterface] Could not persist history:', e);
                }
            },

            restore: function () {
                try {
                    var raw = localStorage.getItem(this.storageKey);
                    if (raw) this.messages = JSON.parse(raw);
                } catch (e) {
                    this.messages = [];
                }
            },

            scrollToBottom: function () {
                var self = this;
                this.$nextTick(function () {
                    var box = self.$refs.log;
                    if (box) box.scrollTop = box.scrollHeight;
                });
            },

            bodyHtml: function (m) {
                // Bot replies are markdown from the model; user text is shown
                // verbatim and escaped so a pasted "<script>" stays inert.
                return m.sender === 'bot'
                    ? renderMarkdown(m.text)
                    : escapeHtml(m.text).replace(/\n/g, '<br>');
            },

            time: function (m) {
                try {
                    return new Date(m.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                } catch (e) { return ''; }
            }
        },

        template: [
            '<div class="ci-root">',
            '  <button class="ci-launcher" @click="toggle" :aria-expanded="String(open)" aria-label="Toggle AgriBot chat">',
            '    <i :class="open ? \'fas fa-times\' : \'fas fa-robot\'"></i>',
            '    <span v-if="unreadHint" class="ci-dot"></span>',
            '  </button>',
            '  <div class="ci-panel" v-show="open">',
            '    <div class="ci-header">',
            '      <span><i class="fas fa-robot"></i> AgriBot</span>',
            '      <div class="ci-header-actions">',
            '        <button @click="clear" title="Clear conversation"><i class="fas fa-trash"></i></button>',
            '        <button @click="open = false" title="Close"><i class="fas fa-chevron-down"></i></button>',
            '      </div>',
            '    </div>',
            '    <div class="ci-log" ref="log">',
            '      <div v-for="m in messages" :key="m.id" class="ci-msg" :class="[\'ci-\' + m.sender, m.isError ? \'ci-err\' : \'\']">',
            '        <div class="ci-bubble" v-html="bodyHtml(m)"></div>',
            '        <div class="ci-time">{{ time(m) }}</div>',
            '      </div>',
            '      <div v-if="loading" class="ci-msg ci-bot">',
            '        <div class="ci-bubble ci-typing"><span></span><span></span><span></span></div>',
            '      </div>',
            '    </div>',
            '    <div v-if="error" class="ci-error">{{ error }}</div>',
            '    <div class="ci-input">',
            '      <input type="text" v-model="input" @keyup.enter="send"',
            '             :disabled="loading || !isFarmer"',
            '             :placeholder="isFarmer ? \'Ask AgriBot...\' : \'Farmer accounts only\'">',
            '      <button @click="send" :disabled="loading || !input.trim() || !isFarmer">',
            '        <i class="fas fa-paper-plane"></i>',
            '      </button>',
            '    </div>',
            '  </div>',
            '</div>'
        ].join('\n')
    });
})();
