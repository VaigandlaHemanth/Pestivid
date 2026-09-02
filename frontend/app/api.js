// The API client. One place that knows the routes, the token and what a failure
// means, so no page has to.

const BASE = location.port === '3000' || location.port === '3001'
  ? '/api' : (localStorage.getItem('pv.api') || '/api');
const TIMEOUT = 15000;

export const session = {
  get token() { return localStorage.getItem('pv.token'); },
  get user() { try { return JSON.parse(localStorage.getItem('pv.user')); } catch { return null; } },
  set(token, user) {
    localStorage.setItem('pv.token', token);
    localStorage.setItem('pv.user', JSON.stringify(user));
  },
  clear() { localStorage.removeItem('pv.token'); localStorage.removeItem('pv.user'); },
};

export class ApiError extends Error {
  constructor(status, body, url) {
    super(body?.message || `${status} on ${url}`);
    this.status = status; this.body = body; this.url = url;
    // A free inference tier rate-limits per minute and per day. A screen that
    // asks questions has to be able to say so rather than showing nothing.
    this.rateLimited = status === 429;
    this.offline = status === 0;
  }
}

async function request(method, pathname, body) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), TIMEOUT);
  let res;
  try {
    res = await fetch(BASE + pathname, {
      method,
      signal: ctl.signal,
      headers: {
        ...(body ? { 'content-type': 'application/json' } : {}),
        ...(session.token ? { authorization: `Bearer ${session.token}` } : {}),
      },
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (e) {
    clearTimeout(timer);
    throw new ApiError(0, { message: e.name === 'AbortError' ? 'That took too long.' : 'No connection.' }, pathname);
  }
  clearTimeout(timer);

  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { message: text.slice(0, 200) }; }

  if (res.status === 401) {
    // The server revokes a token by bumping tokenVersion, so a 401 here means
    // the session is genuinely dead, not that this one call failed.
    session.clear();
    if (!/\/auth\/(login|register)$/.test(pathname)) location.href = './signin.html';
  }
  if (!res.ok) throw new ApiError(res.status, data, pathname);
  return data;
}

const get = p => request('GET', p);
const post = (p, b) => request('POST', p, b);
const put = (p, b) => request('PUT', p, b);
const del = p => request('DELETE', p);

/**
 * Send a clip the only way a free host allows.
 *
 * The API cannot carry it -- a function request body caps at 4.5 MB and a
 * forty-second clip is about 10 MB -- so the handset asks for a one-use URL,
 * posts the file straight to storage, and then tells us the identifier. The
 * server fetches the object back and hashes it there, which is why the page can
 * still say the hash came from the bytes we received.
 *
 * onProgress gets 0..1 so the screen can drive a scaleX bar.
 */
export async function sendVideo(file, meta, onProgress) {
  const ticket = await api.videos.requestUpload({ crop: meta.crop });
  if (file.size > ticket.maxBytes) {
    throw new ApiError(413, { message: `That clip is ${Math.round(file.size / 1e6)} MB and the limit is ${Math.round(ticket.maxBytes / 1e6)} MB.` }, 'upload');
  }
  const form = new FormData();
  form.append(ticket.field || 'file', file, file.name || 'clip.mp4');
  form.append('network', 'public');

  const cid = await new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();          // fetch cannot report upload progress
    xhr.open('POST', ticket.url);
    xhr.upload.onprogress = e => { if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total); };
    xhr.onerror = () => reject(new ApiError(0, { message: 'The upload stopped. It is still on your phone.' }, 'upload'));
    xhr.ontimeout = () => reject(new ApiError(0, { message: 'The upload took too long.' }, 'upload'));
    xhr.onload = () => {
      if (xhr.status < 200 || xhr.status >= 300) {
        // Storage is rate limited on the free tier. That is a wait, not a
        // refusal, and the clip is still on the phone either way.
        return reject(new ApiError(xhr.status, {
          message: xhr.status === 429
            ? 'Storage is busy just now. Your video is still on your phone, it will go on the next try.'
            : 'Storage would not take the file.',
        }, 'upload'));
      }
      let d = null;
      try { d = JSON.parse(xhr.responseText); } catch { /* handled below */ }
      const id = d?.data?.cid || d?.cid || d?.IpfsHash;
      id ? resolve(id) : reject(new ApiError(502, { message: 'Storage did not say where it put the file.' }, 'upload'));
    };
    xhr.send(form);
  });

  // Nothing is recorded until the server has read the object back itself.
  return api.videos.confirmUpload({ ...meta, cid });
}

export const api = {
  auth: {
    login: (email, password) => post('/auth/login', { email, password }),
    register: (b) => post('/auth/register', b),
    me: () => get('/auth/me'),
    changePassword: (currentPassword, newPassword) => post('/auth/change-password', { currentPassword, newPassword }),
    signOutEverywhere: () => post('/auth/sign-out-everywhere', {}),
  },
  videos: {
    mine: (farmerId) => get(`/videos/farmer/${farmerId}`),
    all: () => get('/videos'),
    provenance: (cid) => get(`/videos/${cid}/provenance`),
    anchor: (cid) => get(`/videos/${cid}/anchor`),
    // Uploading through the API is not possible on the free tier: a function
    // request body caps at 4.5 MB and a 40-second clip is about 10 MB. The
    // handset asks for a one-use URL, sends the file straight to storage, and
    // the server then pulls the object back and hashes it itself -- which is
    // what keeps "the server hashes the bytes it received" true.
    requestUpload: (meta) => post('/videos/upload-url', meta),
    confirmUpload: (b) => post('/videos/confirm-upload', b),
  },
  projects: {
    open: () => get('/funding-requests'),
    one: (id) => get(`/funding-requests/${id}`),
    mine: (farmerId) => get(`/funding-requests/farmer/${farmerId}`),
    create: (b) => post('/funding-requests', b),
    reportHarvest: (id, b) => post(`/funding-requests/${id}/harvest`, b),
  },
  investments: {
    mine: (investorId) => get(`/investments/investor/${investorId}`),
    // The route answers {project, investments}, and both callers treated the
    // answer as an array. On the payout screen that threw
    // "investments.reduce is not a function" ON SCREEN -- and because it threw
    // before any bind ran, the page kept the artboard's numbers: a payout of
    // 4,04,160 split between four named investors, none of it real.
    onProject: (projectId) => get(`/investments/project/${projectId}`)
      .then(r => (Array.isArray(r) ? r : (r?.investments || []))),
    create: (b) => post('/investments', b),
  },
  listings: {
    all: () => get('/listings'),
    mine: (farmerId) => get(`/listings/farmer/${farmerId}`),
    media: (id) => get(`/listings/${id}/media`),
    create: (b) => post('/listings', b),
  },
  purchases: {
    asBuyer: (buyerId) => get(`/purchases/buyer/${buyerId}`),
    asFarmer: (farmerId) => get(`/purchases/farmer/${farmerId}`),
    create: (b) => post('/purchases', b),
  },
  messages: {
    threads: (userId) => get(`/messaging/conversations/${userId}`),
    inThread: (conversationId) => get(`/messaging/conversations/${conversationId}/messages`),
    send: (conversationId, b) => post(`/messaging/conversations/${conversationId}/messages`, b),
    markRead: (conversationId) => put(`/messaging/conversations/${conversationId}/messages/read`),
    open: (b) => post('/messaging/conversations', b),
  },
  notifications: {
    mine: (userId) => get(`/notifications/user/${userId}`),
    read: (id) => put(`/notifications/${id}/read`),
    dismiss: (id) => del(`/notifications/${id}`),
  },
  money: { transactions: (userId) => get(`/transactions/user/${userId}`) },
  ai: {
    // POST /ai/chatbot takes { messages: [{role, content}] } and answers
    // { text }. This sent { question, history } and read r.answer, so the
    // chatbot answered "Invalid chat data format" to every question ever asked
    // -- and would have printed the not-covered fallback even on success.
    ask: (question, history = []) => post('/ai/chatbot', {
      messages: [
        ...history.slice(-8).map(m => ({ role: m.role, content: String(m.content || '') })),
        { role: 'user', content: String(question || '') },
      ],
    }).then(r => ({ ...r, answer: r?.text ?? r?.answer ?? '' })),
    leaf: (b) => post('/ai/analyze-plant', b),
  },
  admin: { flagged: () => get('/videos/review-queue') },
};

// ---- formatting, in one place so two screens cannot disagree ----------------

/** Indian grouping, no decimals, and a real rupee sign. */
export const rupees = (n) => n == null ? 'not yet'
  : '₹' + Math.round(n).toLocaleString('en-IN', { maximumFractionDigits: 0 });

/* A lot's price is a RANGE the farmer will accept, and four screens printed it
 * as "₹26,000, ₹34,000" -- two figures joined by a comma, which reads as two
 * prices, or as one price with a stray thousand. One dash, one figure. */
export const rupeeRange = (lo, hi) => {
  if (lo == null && hi == null) return 'not yet';
  if (lo == null || hi == null || Number(lo) === Number(hi)) return rupees(lo ?? hi);
  return `${rupees(lo)} – ${rupees(hi)}`;
};

/* Which glyph a notice wears, and where pressing it goes.
 *
 * Shared by the notices page and the banner that shows a notice the moment it
 * arrives, so the two cannot disagree about what a notice IS.
 *
 * The server names the event in `type`, and that is what decides. The words
 * were being matched by regex instead, so "Bob sent you a message" wore the
 * paperwork glyph and "Charlie put ₹50,000 into your season" did too: neither
 * sentence contained the verb the pattern was waiting for. The words are only a
 * fallback now, for rows written before the types were consistent. */
export function noticeKind(n) {
  switch (n?.type) {
    case 'message': return 'person';
    case 'investment': case 'funding': case 'purchase': case 'sale': case 'payout': return 'money';
    case 'listing': return 'listing';
    case 'warning': case 'error': return 'wrong';
    case 'success': return 'proved';
    default: break;
  }
  const txt = `${n?.title || ''} ${n?.message || ''}`.toLowerCase();
  if (/\bwrote\b|asked you|sent you a message/.test(txt)) return 'person';
  if (/\bblock\b/.test(txt) || /date .*(landed|written)/.test(txt)) return 'proved';
  if (/bought|paid|funded|put .* into|asking for money|investor/.test(txt)) return 'money';
  if (/listed|listing/.test(txt)) return 'listing';
  if (/failed|queried|problem|cannot|refus/.test(txt)) return 'wrong';
  return 'listing';
}

/* Where a notice leads, when it leads anywhere. Only pairs that really exist: a
 * screen that shows the thing the notice is about, to the role reading it.
 * Anything else is null, and a caller shows no chevron and goes nowhere. */
export function noticeDestination(n, role) {
  // An admin's notices are the queue they exist to act on.
  if (role === 'admin') return n?.itemType ? 'admin' : null;
  switch (n?.itemType) {
    // Straight into the conversation. Older notices carry a message id here
    // and fall back to the first thread, which is what messages.js did anyway.
    case 'Message': return n.itemId ? `messages?c=${n.itemId}` : 'messages';
    case 'FundingRequest':
      return role === 'investor' ? 'invest' : role === 'farmer' ? 'money' : null;
    case 'Investment':
      return role === 'investor' ? 'portfolio' : role === 'farmer' ? 'money' : null;
    case 'Listing':
      if (role === 'buyer') return n.type === 'purchase' ? 'orders' : 'market';
      if (role === 'farmer') return n.type === 'purchase' ? 'money' : 'home';
      return null;
    default: return null;
  }
}

/* Whose notice is this?
 *
 * A global notification is one document every user can see, and the notices
 * page promises in its own rail that "only your own plots and your own money
 * appear here". It was breaking that promise three times over on the farmer's
 * screen: "a farmer is asking for money to grow Potato, 2 acres" is addressed
 * to somebody with money to put in, and it was showing up on the screen of the
 * farmer who is asking for it.
 *
 * So a broadcast reaches the role it was written for. Anything addressed to one
 * person is that person's, whatever its type. This lives here because BOTH the
 * notices page and the envelope badge in the app bar have to count the same
 * set -- the page saying "4 you have not read" under a badge reading 7 is worse
 * than either number being wrong on its own.
 */
const BROADCAST_FOR = {
  funding: ['investor', 'admin'],   // somebody is asking for money
  listing: ['buyer', 'admin'],      // a lot has come up for sale
};
export function noticeForRole(n, role) {
  if (!n || !n.global) return true;
  const who = BROADCAST_FOR[n.type];
  return !who || who.includes(role);
}

export const dayMonth = (iso) => {
  if (!iso) return 'not yet';
  const d = new Date(iso);
  return d.toLocaleDateString('en-IN', { day: 'numeric', month: 'long' });
};

/** "Today, 8:40 am" reads faster in a field than a date does. */
export const whenShort = (iso) => {
  // No date is not an em dash. A caller that gets '' removes the element rather
  // than printing a placeholder on a line of its own where a time would be.
  if (!iso) return '';
  const d = new Date(iso), now = new Date();
  const t = d.toLocaleTimeString('en-IN', { hour: 'numeric', minute: '2-digit' }).toLowerCase();
  const days = Math.round((new Date(now.toDateString()) - new Date(d.toDateString())) / 86400000);
  if (days === 0) return `Today, ${t}`;
  if (days === 1) return `Yesterday, ${t}`;
  return `${dayMonth(iso)}, ${t}`;
};

/* Does this server write dates at all? Development runs with anchoring off,
 * and every screen went on saying "usually by tomorrow" about dates that
 * would never land. Asked once per page; until the answer arrives the screens
 * assume yes, which is what production is. */
export const anchoring = { enabled: true, known: false };
export const anchoringReady = get('/videos/anchoring')
  .then((r) => { anchoring.enabled = Boolean(r?.enabled); anchoring.known = true; })
  .catch(() => {});

/** The three states a video's date can be in, in the words the screens use. */
export function dateState(v) {
  // `text` is the full line for a detail screen. `short` is for a row in a
  // list, where the same sentence on every row says it three times too often --
  // the group note above the list carries the explanation once.
  if (v?.anchored && v?.blockHeight) return {
    kind: 'proved',
    text: `Date stamped · block ${Number(v.blockHeight).toLocaleString('en-IN')}`,
    short: `Block ${Number(v.blockHeight).toLocaleString('en-IN')}`,
  };
  if (v?.cid) return anchoring.enabled ? {
    kind: 'waiting',
    text: 'On our server · date being written, usually by tomorrow',
    short: 'Date being written',
  } : {
    kind: 'waiting',
    text: 'On our server · this server is not writing dates yet',
    short: 'Date not written here',
  };
  return { kind: 'phone', text: 'Not sent yet', short: 'Not sent yet' };
}
