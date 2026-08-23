// Minimal stand-in for the canvas runtime, so an artboard can be rendered in a
// plain browser for measurement. Expands <sc-for> and {{...}} using the values
// the artboard's own DCLogic.renderVals() returns. Not a renderer — just enough
// that the audit measures real text instead of template placeholders.
(function () {
  var script = document.querySelector('script[data-dc-script]');
  var vals = {};
  if (script) {
    try {
      var src = script.textContent.replace(/^\s*class\s+Component\s+extends\s+DCLogic\s*\{/, 'return (function(){ var o = {');
      // simpler: pull the body of renderVals() and eval it
      var m = script.textContent.match(/renderVals\s*\(\s*\)\s*\{([\s\S]*?)\n\s*\}\s*\n?\s*\}/);
      if (m) vals = new Function(m[1])() || {};
    } catch (e) { vals = {}; }
  }

  function lookup(path, scope) {
    var parts = String(path).trim().split('.');
    var cur = scope && Object.prototype.hasOwnProperty.call(scope, parts[0]) ? scope : vals;
    for (var i = 0; i < parts.length; i++) {
      if (cur == null) return '';
      cur = cur[parts[i]];
    }
    return cur == null ? '' : cur;
  }

  function interpolate(str, scope) {
    return str.replace(/\{\{([^}]+)\}\}/g, function (_, p) { return lookup(p, scope); });
  }

  // <sc-for list="{{items}}" as="x"> ... </sc-for>
  document.querySelectorAll('sc-for').forEach(function (node) {
    var listExpr = (node.getAttribute('list') || '').replace(/[{}]/g, '').trim();
    var as = node.getAttribute('as') || 'item';
    var items = lookup(listExpr, null);
    if (!Array.isArray(items)) items = [];
    var tpl = node.innerHTML;
    var out = '';
    items.forEach(function (it) {
      var scope = {}; scope[as] = it;
      out += tpl.replace(/\{\{([^}]+)\}\}/g, function (_, p) { return lookup(p, scope); });
    });
    var frag = document.createElement('div');
    frag.style.display = 'contents';
    frag.innerHTML = out;
    node.parentNode.replaceChild(frag, node);
  });

  // remaining {{...}} in text nodes and style attributes
  var walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null);
  var texts = [];
  while (walker.nextNode()) texts.push(walker.currentNode);
  texts.forEach(function (t) {
    if (t.nodeValue.indexOf('{{') !== -1) t.nodeValue = interpolate(t.nodeValue, null);
  });
  document.querySelectorAll('[style]').forEach(function (el) {
    var s = el.getAttribute('style');
    if (s && s.indexOf('{{') !== -1) el.setAttribute('style', interpolate(s, null));
  });

  // <helmet> is a data container in this format, never rendered
  document.querySelectorAll('helmet').forEach(function (h) {
    h.style.display = 'none';
    // but its <style>/<link> must stay live, so move them to <head>
    Array.prototype.slice.call(h.querySelectorAll('style, link')).forEach(function (n) {
      document.head.appendChild(n);
    });
  });
  document.querySelectorAll('x-dc').forEach(function (x) { x.style.display = 'block'; });
  document.documentElement.setAttribute('data-shim-ready', '1');
})();
