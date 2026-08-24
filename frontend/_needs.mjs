// Query strings for the pages that are ABOUT something.
//
// Five pages take an id and are blank without it. Every harness loaded them
// bare, so all any of them ever checked was the empty state -- which is how
// payout's "Send it", the button that reports a harvest irreversibly, sat
// unwired while click-everything reported nothing dead on that page.
//
// The ids come from real records so the harnesses see the real screens.

const API = 'http://127.0.0.1:3001/api';

const login = async (role) => (await fetch(`${API}/auth/login`, {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify({ email: `demo.${role}@pestivid.sim`, password: 'password123' }),
})).json();

const get = async (path, token) => {
  const r = await fetch(`${API}${path}`, { headers: { authorization: `Bearer ${token}` } });
  if (!r.ok) return null;
  return r.json();
};

let cache = null;

/**
 * Resolve every parameterised page's query in one pass.
 *
 * Returns a plain map of slug -> query string (with the leading '?'), missing
 * entries for anything the database has no record for. A harness should treat a
 * missing entry as "check it bare" rather than skipping the page, so a page
 * whose data has gone away still gets looked at.
 */
export async function needs() {
  if (cache) return cache;
  const out = {};
  try {
    const farmer = await login('farmer');
    const fid = farmer.user?._id || farmer.user?.id;
    const projects = (await get(`/funding-requests/farmer/${fid}`, farmer.token)) || [];
    const list = Array.isArray(projects) ? projects : (projects.projects || []);
    // report-harvest wants one that has NOT been reported; payout is happy
    // either way and is more useful showing the reported record.
    const open = list.find(p => !p.harvestReportedAt) || list[0];
    const reported = list.find(p => p.harvestReportedAt) || list[0];
    if (reported) out.payout = `?project=${reported._id}`;
    if (open) out['report-harvest'] = `?project=${open._id}`;

    const videos = (await get(`/videos/farmer/${fid}`, farmer.token)) || [];
    const plotName = videos[0]?.crop || videos[0]?.location;
    if (plotName) out.plot = `?name=${encodeURIComponent(plotName)}`;

    const threads = (await get(`/messaging/conversations/${fid}`, farmer.token)) || [];
    const tl = Array.isArray(threads) ? threads : (threads.conversations || []);
    if (tl[0]?._id) out.thread = `?c=${tl[0]._id}`;

    const investor = await login('investor');
    const openToFund = (await get('/funding-requests', investor.token)) || [];
    const fl = Array.isArray(openToFund) ? openToFund : (openToFund.projects || []);
    if (fl[0]?._id) out['confirm-investment'] = `?project=${fl[0]._id}`;
  } catch {
    // A harness must still run against an empty or unreachable database.
  }
  cache = out;
  return out;
}
