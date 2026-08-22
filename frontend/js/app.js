/* --- App bootstrap: app.js ---
 *
 * The main Vue instance is inline at the bottom of index.html, so this file
 * stays out of rendering. It handles the cross-cutting things that instance
 * has no hook for:
 *
 *   1. A visible banner when the Node backend is unreachable. Without it the
 *      app just renders an empty marketplace and logs errors to a console the
 *      user is not looking at.
 *   2. Readable axios errors — the backend answers with { message }, which the
 *      default "Request failed with status code 500" hides.
 *   3. A one-line console summary of which services are actually up.
 *
 * Runs before the Vue instance is constructed, so keep it independent of it.
 */
(function () {
    'use strict';

    // Same runtime config as index.html; no hardcoded origin.
    var CFG = (window.__PESTVID || {});
    var API_BASE = CFG.apiBase || '/api';
    var API_ORIGIN = API_BASE.replace(/\/api$/, '');
    var FLASK_ORIGIN = CFG.mlBase || '/ml';

    // ---------------------------------------------------------------------
    // 1. Readable axios errors
    // ---------------------------------------------------------------------

    if (typeof axios !== 'undefined') {
        axios.interceptors.response.use(
            function (res) { return res; },
            function (err) {
                var body = err.response && err.response.data;
                var serverMsg = body && (body.message || body.error);
                if (serverMsg) {
                    // Keep the original on .originalMessage in case a caller
                    // was matching on the status-code string.
                    err.originalMessage = err.message;
                    err.message = serverMsg;
                }
                if (err.response && err.response.status === 401) {
                    console.warn('[app] 401 from ' + (err.config && err.config.url) +
                                 ' — token missing or expired.');
                }
                return Promise.reject(err);
            }
        );
    }

    // ---------------------------------------------------------------------
    // 2. Offline banner
    // ---------------------------------------------------------------------

    function showBanner(html, tone) {
        var existing = document.getElementById('app-status-banner');
        if (existing) existing.remove();

        var colors = tone === 'warn'
            ? { bg: '#fffbeb', border: '#f59e0b', fg: '#92400e' }
            : { bg: '#fef2f2', border: '#dc2626', fg: '#991b1b' };

        var bar = document.createElement('div');
        bar.id = 'app-status-banner';
        bar.setAttribute('role', 'alert');
        bar.style.cssText = [
            'position:fixed', 'top:0', 'left:0', 'right:0', 'z-index:100',
            'padding:0.6rem 1rem', 'font:600 0.8rem Inter,system-ui,sans-serif',
            'text-align:center',
            'background:' + colors.bg,
            'color:' + colors.fg,
            'border-bottom:2px solid ' + colors.border
        ].join(';');
        bar.innerHTML = html +
            ' <button style="margin-left:0.75rem;background:transparent;border:none;' +
            'cursor:pointer;font-weight:700;color:inherit" aria-label="Dismiss">&times;</button>';
        bar.querySelector('button').onclick = function () { bar.remove(); };
        document.body.appendChild(bar);
    }

    function checkBackend() {
        return fetch(API_ORIGIN + '/api/listings', { method: 'GET' })
            .then(function (r) { return r.ok || r.status === 401; })
            .catch(function () { return false; });
    }

    function checkFlask() {
        return fetch(FLASK_ORIGIN + '/health', { method: 'GET' })
            .then(function (r) { return r.ok; })
            .catch(function () { return false; });
    }

    // ---------------------------------------------------------------------
    // 3. Startup report
    // ---------------------------------------------------------------------

    function boot() {
        Promise.all([checkBackend(), checkFlask()]).then(function (results) {
            var backendUp = results[0];
            var flaskUp = results[1];

            console.log('%c PestiVid ', 'background:#10b981;color:#fff;border-radius:3px',
                'backend ' + (backendUp ? 'up' : 'DOWN') + ' at ' + API_BASE + ' | ' +
                'flask ML ' + (flaskUp ? 'up' : 'not running') + ' at ' + FLASK_ORIGIN);

            if (!backendUp) {
                showBanner(
                    'Backend not reachable at <code>' + API_ORIGIN + '</code>. ' +
                    'Login, marketplace and messaging will not work. ' +
                    'Start it with <code>node dev-server.js</code> in <code>backend/</code>.',
                    'error'
                );
                return;
            }

            if (!flaskUp) {
                // Not an error: /api/ai/predict-proxy falls back to Groq.
                console.info('[app] Flask ML server is not running — plant analysis ' +
                             'will fall back to the Groq LLM path.');
            }
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', boot);
    } else {
        boot();
    }
})();
