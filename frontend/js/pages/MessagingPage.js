/* --- Messaging live-refresh: MessagingPage.js ---
 *
 * The messaging UI itself lives inline in index.html (the
 * v-if="current==='messaging'" section), so this file deliberately does not
 * render a second copy of it.
 *
 * What it adds is the piece that was missing: the inline code only fetches
 * conversations when you switch pages, so an incoming message never shows up
 * until you navigate away and back. This polls loadUserConversations() while
 * the messaging page is open, and stops as soon as you leave it or the tab
 * goes to the background.
 *
 * Attaches via a Vue mixin applied globally, so it hooks the existing root
 * instance without touching index.html.
 */
(function () {
    'use strict';

    if (typeof Vue === 'undefined') {
        console.error('[MessagingPage] Vue is not loaded — check script order in index.html.');
        return;
    }

    var POLL_MS = 8000;

    // _msgPollId and _msgVisibilityHandler are set directly on the instance
    // rather than declared in data(): Vue 2 refuses to proxy keys starting with
    // an underscore, and neither value needs to be reactive.
    Vue.mixin({
        mounted: function () {
            if (this !== this.$root) return;

            var self = this;

            // Re-check whenever the page or auth state changes.
            this.$watch(
                function () { return [self.current, self.isAuthenticated].join('|'); },
                function () { self._syncMessagePolling(); },
                { immediate: true }
            );

            // Browsers throttle timers in hidden tabs; stop cleanly instead of
            // letting requests pile up, and refresh once on return.
            this._msgVisibilityHandler = function () {
                if (document.hidden) {
                    self._stopMessagePolling();
                } else {
                    self._syncMessagePolling();
                    if (self.current === 'messaging') self._pollMessagesOnce();
                }
            };
            document.addEventListener('visibilitychange', this._msgVisibilityHandler);
        },

        beforeDestroy: function () {
            if (this !== this.$root) return;
            this._stopMessagePolling();
            if (this._msgVisibilityHandler) {
                document.removeEventListener('visibilitychange', this._msgVisibilityHandler);
            }
        },

        methods: {
            _syncMessagePolling: function () {
                var shouldPoll = this.current === 'messaging' &&
                                 this.isAuthenticated &&
                                 !document.hidden;
                if (shouldPoll) this._startMessagePolling();
                else this._stopMessagePolling();
            },

            _startMessagePolling: function () {
                if (this._msgPollId) return;
                var self = this;
                this._msgPollId = setInterval(function () {
                    self._pollMessagesOnce();
                }, POLL_MS);
                console.log('[MessagingPage] Live refresh on (' + POLL_MS + 'ms).');
            },

            _stopMessagePolling: function () {
                if (!this._msgPollId) return;
                clearInterval(this._msgPollId);
                this._msgPollId = null;
                console.log('[MessagingPage] Live refresh off.');
            },

            _pollMessagesOnce: function () {
                if (typeof this.loadUserConversations !== 'function') {
                    // Root changed shape — stop rather than erroring every tick.
                    this._stopMessagePolling();
                    return;
                }
                var r = this.loadUserConversations();
                if (r && typeof r.catch === 'function') {
                    r.catch(function (e) {
                        console.warn('[MessagingPage] Refresh failed:', e.message);
                    });
                }
            }
        }
    });
})();
