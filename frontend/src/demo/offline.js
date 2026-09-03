/**
 * The published demo: the real interface, answering from a recording.
 *
 * This is the whole application — the same components, the same state, the
 * same api.js — served from a static host with no backend behind it. What it
 * cannot have is a backend, so `fetch` is intercepted here and the `/api`
 * routes are answered from responses the running system actually produced.
 *
 * Recorded, not written. Every answer, citation, page number, section and
 * timing in demo-data.json came back from qwen2.5:3b-instruct over three
 * indexed papers on an M1 with 8 GB. Nothing was composed to look good,
 * including the questions it declines to answer.
 *
 * Two things genuinely cannot work without a machine to run on, and both say
 * so rather than failing quietly:
 *
 *   - Adding a paper. Parsing, embedding and answering need the backend and a
 *     local model; a static page has neither. The upload control stays where
 *     it is and explains that, because a missing control looks like a missing
 *     feature.
 *   - Asking something that was not recorded. The demo says which questions it
 *     holds instead of pretending the papers do not support an answer — those
 *     are different statements, and conflating them would misrepresent the
 *     refusal behaviour this is partly here to show.
 *
 * Loaded only when the build sets VITE_DEMO, so a normal build does not carry
 * it and a local instance is never talking to a fixture.
 */

let data = null;

const json = (body, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });

const CANNOT_UPLOAD =
  'This is a published recording, so there is no machine behind it to read a ' +
  'new paper on. Adding your own papers is the thing LocalScholar is for — it ' +
  'runs on your hardware, with your library, and nothing leaves it. Install it ' +
  'from github.com/asifuddin01/LocalScholar to do that.';

function normalise(q) {
  return String(q || '').trim().toLowerCase().replace(/\s+/g, ' ').replace(/[?.!]+$/, '');
}

/** An answer shaped exactly like the backend's, saying what the demo holds. */
function notRecorded(question) {
  const held = Object.values(data.ask).map((a) => a.question);
  return {
    question,
    answer:
      'This is a recorded demo, so it can only replay questions that were ' +
      'asked while it was running — it has not been asked this one. That is ' +
      'not the same as the papers failing to support an answer.\n\n' +
      `Recorded questions:\n${held.map((q) => `• ${q}`).join('\n')}`,
    found: false,
    sources: [],
    cited_indexes: [],
    model: data.model,
    took_ms: 0,
  };
}

/** Route table. Order matters: the longer paths are tested first. */
async function handle(url, options) {
  const path = url.replace(/^https?:\/\/[^/]+/, '').split('?')[0];
  const method = (options?.method || 'GET').toUpperCase();

  if (method === 'POST' && path === '/api/documents') return json({ detail: CANNOT_UPLOAD }, 503);
  if (method === 'DELETE' && path.startsWith('/api/documents/')) {
    return json(
      { detail: 'The recorded library cannot be changed. Run LocalScholar to manage your own.' },
      503,
    );
  }

  if (method === 'POST' && path === '/api/ask') {
    const body = JSON.parse(options.body);
    const hit = data.ask[normalise(body.question)];
    return json(hit ?? notRecorded(body.question));
  }

  if (method === 'POST' && path === '/api/search') {
    const body = JSON.parse(options.body);
    const hit = data.search[normalise(body.query)];
    return json(hit ?? { query: body.query, method: body.method, results: [], took_ms: 0 });
  }

  if (method === 'POST' && path === '/api/compare') {
    const body = JSON.parse(options.body);
    const key = [...(body.document_ids || [])].sort().join('|');
    const hit = data.compare[key];
    if (hit) return json(hit);
    return json(
      {
        detail:
          'This recording holds comparisons between pairs of its three papers. ' +
          'Choose two of them, or run LocalScholar to compare your own.',
      },
      503,
    );
  }

  const route = data.routes[path];
  if (route !== undefined) return json(route);

  return json({ detail: `The recorded demo has no response for ${path}.` }, 404);
}

/**
 * A banner saying what this is.
 *
 * Injected here rather than written into index.html, because index.html is
 * also the local build's, and an instance running on somebody's own machine
 * must not carry a notice telling them they are looking at a recording.
 */
function installBanner() {
  const el = document.createElement('aside');
  el.setAttribute('role', 'note');
  el.style.cssText =
    'background:#2F566D;color:#fff;padding:.7rem 1rem;font:14px/1.5 -apple-system,' +
    'BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;text-align:center';
  el.innerHTML =
    '<strong>A recording of a real session.</strong> This is the whole interface, ' +
    'answering from responses the running system produced — three papers, a local ' +
    'model, nothing written by hand. Adding your own papers needs your own machine: ' +
    '<a href="https://github.com/asifuddin01/LocalScholar" style="color:#fff">' +
    'install it here</a>.';
  document.body.prepend(el);
}

export function installDemoBackend() {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', installBanner, { once: true });
  } else {
    installBanner();
  }

  const real = window.fetch.bind(window);

  window.fetch = async (input, options) => {
    const url = typeof input === 'string' ? input : input.url;
    if (!/^\/api\//.test(url.replace(/^https?:\/\/[^/]+/, ''))) return real(input, options);

    if (!data) {
      /* Relative to the document, so the same build works at a user site and
         at /LocalScholar/ on project pages. */
      data = await real(new URL('demo-data.json', document.baseURI).href).then((r) => r.json());
    }

    /* A visible pause. Every one of these took real seconds on real hardware —
       the recorded `took_ms` is in the response — and returning instantly
       would quietly misrepresent what running a 3B model locally feels like.
       Capped, because nobody should sit through eighty seconds to see a demo. */
    const started = performance.now();
    const response = await handle(url, options);
    const body = await response.clone().json().catch(() => null);
    const real_ms = Number(body?.took_ms) || 0;
    const wait = Math.min(real_ms, 2500) - (performance.now() - started);
    if (wait > 0) await new Promise((r) => setTimeout(r, wait));
    return response;
  };
}
