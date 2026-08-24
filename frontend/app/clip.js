// Handing a filmed clip from the record screen to the send screen.
//
// This was `window.__pvClip = {...}` followed by `location.href = './sent.html'`
// -- and a navigation destroys `window`. So the object was gone before the next
// page ran, and every clip a farmer filmed arrived at a screen that said "there
// is no clip to send". The core action of the whole product could not complete.
//
// sessionStorage cannot carry it either: it stores strings, and a forty-second
// clip is about ten megabytes. IndexedDB stores a Blob as a Blob, survives the
// navigation, and is the only option that does both.
//
// The clip still never leaves the handset here, which is what lets the send
// screen offer "throw this one away" as a real choice rather than a request to
// delete something already uploaded.

const DB = 'pv.clip';
const STORE = 'clip';
const KEY = 'pending';

function open() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function run(mode, fn) {
  return open().then(db => new Promise((resolve, reject) => {
    const tx = db.transaction(STORE, mode);
    const out = fn(tx.objectStore(STORE));
    tx.oncomplete = () => {
      db.close();
      // `out?.result ?? out` was wrong: an EMPTY store gives a request whose
      // result is undefined, so ?? fell through and handed back the IDBRequest
      // itself -- which is truthy. The send screen then believed there was a
      // clip and rendered "0 seconds · NaN MB · on your phone".
      resolve(out && typeof out === 'object' && 'result' in out ? out.result : out);
    };
    tx.onerror = () => { db.close(); reject(tx.error); };
  }));
}

/** Store the clip for the next screen. Resolves before the caller navigates. */
export function putClip(clip) {
  if (!('indexedDB' in window)) return Promise.reject(new Error('no indexedDB'));
  return run('readwrite', store => store.put(clip, KEY));
}

/**
 * Take the clip, removing it as it goes.
 *
 * Removing is deliberate: a reload of the send screen after a successful upload
 * must not offer to send the same clip twice, and a clip left in the database is
 * a video sitting on somebody's phone that they think they threw away.
 */
export async function takeClip() {
  if (!('indexedDB' in window)) return null;
  try {
    const clip = await run('readonly', store => store.get(KEY));
    if (clip) await run('readwrite', store => store.delete(KEY));
    return clip || null;
  } catch {
    return null;
  }
}

/** Drop it without reading it. Used by "throw this one away". */
export function dropClip() {
  if (!('indexedDB' in window)) return Promise.resolve();
  return run('readwrite', store => store.delete(KEY)).catch(() => {});
}
