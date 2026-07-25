/* Shared plumbing for both annotation tasks.
 *
 * Persistence is belt-and-braces: every answer is written to localStorage
 * immediately (instant, survives a closed tab) and POSTed to serve.py (survives
 * a different browser or machine). On load we take whichever copy has more
 * answers, so neither path can silently lose work.
 *
 * Item order is a deterministic function of the annotator's name, so resuming
 * always reproduces the same sequence — and two annotators see different
 * orders, which keeps order effects from correlating across people.
 */

const NAME_KEY = 'humanEval.annotator';

function getName() {
  return localStorage.getItem(NAME_KEY) || '';
}

function setName(name) {
  localStorage.setItem(NAME_KEY, name.trim());
}

function requireName() {
  const name = getName();
  if (!name) {
    window.location.href = 'index.html';
    throw new Error('no annotator name');
  }
  return name;
}

/* Deterministic per-annotator shuffle ------------------------------------- */

function cyrb53(str, seed = 0) {
  let h1 = 0xdeadbeef ^ seed, h2 = 0x41c6ce57 ^ seed;
  for (let i = 0; i < str.length; i++) {
    const ch = str.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return 4294967296 * (2097151 & h2) + (h1 >>> 0);
}

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Fisher-Yates seeded by (annotator name + task). */
function shuffleFor(items, name, task) {
  const rand = mulberry32(cyrb53(name + '::' + task) >>> 0);
  const out = items.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

/* Storage ----------------------------------------------------------------- */

class Store {
  /**
   * @param version bump when a task's answer *options* change. It is part of
   *   the localStorage key, so answers recorded under the old options are
   *   ignored instead of being re-POSTed and resurrecting values the task no
   *   longer offers. Server-side files must be deleted separately.
   * @param instructionVersion bump when the *wording* of the task changes
   *   without changing the options. Unlike `version` this invalidates nothing
   *   — it is recorded on the answers so a run can be told apart from one
   *   collected under earlier guidance.
   */
  constructor(task, name, total, version = 1, instructionVersion = 1) {
    this.task = task;
    this.name = name;
    this.total = total;
    this.instructionVersion = instructionVersion;
    this.answers = {};
    this.lsKey = `humanEval.${task}.v${version}.${name}`;
    this.pending = null;
    this.serverOk = true;
  }

  async load() {
    let local = {};
    try {
      local = JSON.parse(localStorage.getItem(this.lsKey) || '{}').answers || {};
    } catch (e) { local = {}; }

    let remote = {};
    try {
      const res = await fetch(this.url(), { cache: 'no-store' });
      if (res.ok) remote = (await res.json()).answers || {};
      else this.serverOk = false;
    } catch (e) {
      this.serverOk = false;   // static server (python -m http.server) — fine
    }

    // Prefer whichever side knows more; merge so nothing is dropped.
    this.answers = Object.assign({}, local, remote);
    if (Object.keys(local).length > Object.keys(remote).length) {
      this.answers = Object.assign({}, remote, local);
    }
    return this.answers;
  }

  url() {
    return `api/progress/${this.task}/${encodeURIComponent(this.name)}`;
  }

  payload() {
    return {
      task: this.task,
      annotator: this.name,
      total: this.total,
      instruction_version: this.instructionVersion,
      answered: Object.keys(this.answers).length,
      updated: new Date().toISOString(),
      answers: this.answers,
    };
  }

  set(id, value) {
    this.answers[id] = Object.assign({ at: new Date().toISOString() }, value);
    this.flushLocal();
    this.queueRemote();
  }

  flushLocal() {
    try {
      localStorage.setItem(this.lsKey, JSON.stringify(this.payload()));
    } catch (e) { /* quota — the server copy still has it */ }
  }

  /* Coalesce rapid answers into one POST. */
  queueRemote() {
    if (this.pending) clearTimeout(this.pending);
    this.pending = setTimeout(() => this.flushRemote(), 400);
  }

  async flushRemote() {
    this.pending = null;
    try {
      const res = await fetch(this.url(), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(this.payload()),
      });
      this.serverOk = res.ok;
    } catch (e) {
      this.serverOk = false;
    }
    if (this.onsave) this.onsave(this.serverOk);
  }

  count() {
    return Object.keys(this.answers).length;
  }

  /**
   * Flush any pending write, then re-read the server copy and confirm it holds
   * at least as many answers as we do. Used to tell the annotator their work is
   * safe — we verify it rather than assuming it, so the reassuring message can
   * never be shown when the server never actually received the data.
   */
  async verifySaved() {
    if (this.pending) { clearTimeout(this.pending); this.pending = null; }
    await this.flushRemote();
    try {
      const res = await fetch(this.url(), { cache: 'no-store' });
      if (!res.ok) return { ok: false, count: 0 };
      const remote = Object.keys((await res.json()).answers || {}).length;
      return { ok: remote >= this.count(), count: remote };
    } catch (e) {
      return { ok: false, count: 0 };
    }
  }

  download() {
    const blob = new Blob([JSON.stringify(this.payload(), null, 2)],
                          { type: 'application/json' });
    const a = document.createElement('a');
    const stamp = new Date().toISOString().slice(0, 10);
    a.href = URL.createObjectURL(blob);
    a.download = `${this.task}__${this.name.replace(/[^\w.-]+/g, '_')}__${stamp}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  }
}

/* Progress bar ------------------------------------------------------------ */

/**
 * @param label optional prefix. Passing one also switches off the percentage,
 *   which is meaningless when the bar is showing a position rather than
 *   completion (e.g. while reviewing a finished task).
 */
function renderProgress(el, done, total, label) {
  const pct = total ? (100 * done / total) : 0;
  el.querySelector('.fill').style.width = pct.toFixed(2) + '%';
  el.querySelector('.count').textContent = label
    ? `${label} ${done} / ${total}`
    : `${done} / ${total}  (${pct.toFixed(0)}%)`;
  el.classList.toggle('reviewing', !!label);
}

async function loadJSON(path) {
  const res = await fetch(path, { cache: 'no-store' });
  if (!res.ok) throw new Error(`could not load ${path} (${res.status})`);
  return res.json();
}
