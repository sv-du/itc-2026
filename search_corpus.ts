/**
 * search_corpus.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * TF-IDF + cosine-similarity search over the ITC 2026 AI corpus.
 *
 * Public API
 * ──────────
 *   search(query: string, topK?: number): SearchResult[]
 *
 * The index is built once from the `corpus/` directory next to this file,
 * then serialised to `corpus_index.cache.json` so every subsequent run
 * (and every subsequent call within the same process) is near-instant.
 *
 * Algorithm
 * ─────────
 *   1. Load & clean each .md file (strip CPP nav boilerplate, keep H1+ body).
 *   2. Tokenise into unigrams + bigrams.
 *   3. Compute TF-IDF weights (sublinear TF · smooth IDF, min_df=2, max_df=85%).
 *   4. Store as an inverted index: term → [(docIdx, weight), …].
 *   5. At query time: vectorise query, sparse dot-product against inverted index,
 *      normalise by precomputed doc L2 norms → cosine similarity.
 *
 * Usage (CLI)
 * ───────────
 *   npx ts-node search_corpus.ts "transfer admission requirements"
 *   npx ts-node search_corpus.ts --top 10 "financial aid"
 *   npx ts-node search_corpus.ts --rebuild "parking permit"   # force index rebuild
 */

import * as fs   from "fs";
import * as path from "path";

// ── Types ─────────────────────────────────────────────────────────────────────

export interface SearchResult {
  /** 1-based rank */
  rank:     number;
  /** Filename within the corpus directory */
  filename: string;
  /** Cosine similarity score (0 – 1) */
  score:    number;
  /** Contextual snippet from the document */
  snippet:  string;
}

/** Runtime in-memory index. */
interface Index {
  N:         number;
  filenames: string[];
  texts:     string[];
  vocab:     Map<string, number>;                  // term  → column index
  inverted:  Map<number, Array<[number, number]>>; // col   → [(docIdx, weight)]
  norms:     number[];                             // docIdx → L2 norm
}

/** Serialisable shape written to / read from disk. */
interface CachePayload {
  N:         number;
  filenames: string[];
  texts:     string[];
  vocab:     Record<string, number>;
  inverted:  Record<string, Array<[number, number]>>;
  norms:     number[];
}

// ── Paths ─────────────────────────────────────────────────────────────────────

const CORPUS_DIR = path.join(process.cwd(), "corpus");
const CACHE_FILE = path.join(process.cwd(), "corpus_index.cache.json");

// ── Text cleaning ─────────────────────────────────────────────────────────────

/**
 * Strip CPP page nav/header boilerplate; keep only the real page content.
 *
 * Every CPP .md file starts with a large repeated navigation block before the
 * first `#` heading that marks the start of page-specific content. We drop
 * everything before that heading, then scrub remaining markdown noise.
 */
function cleanText(raw: string): string {
  const lines = raw.split(/\r?\n/);

  // Find the first H1 or H2 heading
  let start = 0;
  for (let i = 0; i < lines.length; i++) {
    if (/^#{1,2}\s+\S/.test(lines[i].trim())) { start = i; break; }
  }

  const out: string[] = [];
  for (let i = start; i < lines.length; i++) {
    let ln = lines[i].trim();
    if (!ln)                                              continue; // blank
    if (/^!\[.*?\]\(.*?\)$/.test(ln))                    continue; // image-only
    if (/^\[.*?\]\(https?:\/\/[^)]*\)$/.test(ln))        continue; // bare nav link
    if (/^[*\-|+]{1,3}$/.test(ln))                       continue; // stray bullets
    ln = ln.replace(/!\[.*?\]\(.*?\)/g, "");                        // inline images
    ln = ln.replace(/\[([^\]]+)\]\([^)]*\)/g, "$1");                // [text](url) → text
    ln = ln.trim();
    if (ln) out.push(ln);
  }

  return out.join(" ");
}

// ── Tokenisation ──────────────────────────────────────────────────────────────

const TOKEN_RE = /\b[a-zA-Z][a-zA-Z0-9\-_]{1,}\b/g;

function tokenize(text: string): string[] {
  return (text.match(TOKEN_RE) ?? []).map(t => t.toLowerCase());
}

/** Emit unigrams + bigrams (mirrors sklearn ngram_range=(1,2)). */
function ngrams(tokens: string[]): string[] {
  const out = [...tokens];
  for (let i = 0; i < tokens.length - 1; i++) {
    out.push(`${tokens[i]} ${tokens[i + 1]}`);
  }
  return out;
}

// ── Index building ─────────────────────────────────────────────────────────────

function buildIndex(): Index {
  if (!fs.existsSync(CORPUS_DIR)) {
    throw new Error(`Corpus directory not found: ${CORPUS_DIR}`);
  }

  const files = fs
    .readdirSync(CORPUS_DIR)
    .filter(f => f.endsWith(".md"))
    .sort();

  // Pass 1 — load, clean, tokenise
  const filenames:  string[]   = [];
  const texts:      string[]   = [];
  const tokenLists: string[][] = [];

  for (const file of files) {
    const raw     = fs.readFileSync(path.join(CORPUS_DIR, file), "utf8");
    const cleaned = cleanText(raw);
    if (cleaned.trim().length < 50) continue; // skip near-empty stubs
    filenames.push(file);
    texts.push(cleaned);
    tokenLists.push(ngrams(tokenize(cleaned)));
  }

  const N = filenames.length;

  // Pass 2 — document frequency
  const df = new Map<string, number>();
  for (const tokens of tokenLists) {
    for (const term of new Set(tokens)) {
      df.set(term, (df.get(term) ?? 0) + 1);
    }
  }

  // Pass 3 — vocabulary (min_df = 2, max_df = 85 %)
  const vocab = new Map<string, number>();
  for (const [term, count] of df) {
    if (count >= 2 && count / N <= 0.85) vocab.set(term, vocab.size);
  }

  // Precompute IDF per column index
  const reverseVocab: string[] = new Array(vocab.size);
  for (const [term, col] of vocab) reverseVocab[col] = term;

  const idfByCol: number[] = new Array(vocab.size);
  for (let col = 0; col < vocab.size; col++) {
    idfByCol[col] = Math.log(N / (df.get(reverseVocab[col]) ?? 1)) + 1;
  }

  // Pass 4 — TF-IDF weights into inverted index
  const inverted = new Map<number, Array<[number, number]>>();
  const norms: number[] = new Array(N).fill(0);

  for (let d = 0; d < N; d++) {
    const tf = new Map<number, number>();
    for (const term of tokenLists[d]) {
      const col = vocab.get(term);
      if (col !== undefined) tf.set(col, (tf.get(col) ?? 0) + 1);
    }

    for (const [col, count] of tf) {
      const weight = (1 + Math.log(count)) * idfByCol[col]; // sublinear TF · IDF
      norms[d] += weight * weight;
      if (!inverted.has(col)) inverted.set(col, []);
      inverted.get(col)!.push([d, weight]);
    }
  }

  for (let d = 0; d < N; d++) norms[d] = Math.sqrt(norms[d]);

  return { N, filenames, texts, vocab, inverted, norms };
}

// ── Disk cache ────────────────────────────────────────────────────────────────

function saveCache(index: Index): void {
  const payload: CachePayload = {
    N:         index.N,
    filenames: index.filenames,
    texts:     index.texts,
    vocab:     Object.fromEntries(index.vocab),
    inverted:  Object.fromEntries(
      [...index.inverted.entries()].map(([col, list]) => [String(col), list])
    ),
    norms:     index.norms,
  };
  fs.writeFileSync(CACHE_FILE, JSON.stringify(payload));
}

function loadCache(): Index {
  const raw = JSON.parse(fs.readFileSync(CACHE_FILE, "utf8")) as CachePayload;
  return {
    N:         raw.N,
    filenames: raw.filenames,
    texts:     raw.texts,
    vocab:     new Map(Object.entries(raw.vocab)),
    inverted:  new Map(
      Object.entries(raw.inverted).map(([col, list]) => [Number(col), list])
    ),
    norms:     raw.norms,
  };
}

// ── Singleton ─────────────────────────────────────────────────────────────────

let _index: Index | null = null;

function getIndex(forceRebuild = false): Index {
  if (_index && !forceRebuild) return _index;

  if (!forceRebuild && fs.existsSync(CACHE_FILE)) {
    process.stderr.write("Loading cached index…\n");
    const t0 = Date.now();
    _index = loadCache();
    process.stderr.write(
      `Cache loaded in ${((Date.now() - t0) / 1000).toFixed(1)}s` +
      `  (${_index.N} documents)\n`
    );
  } else {
    process.stderr.write("Building index from corpus…\n");
    const t0 = Date.now();
    _index = buildIndex();
    process.stderr.write(
      `Index built in ${((Date.now() - t0) / 1000).toFixed(1)}s` +
      `  (${_index.N} documents) — saving cache…\n`
    );
    saveCache(_index);
    process.stderr.write("Cache saved.\n");
  }

  return _index;
}

// ── Snippet extraction ────────────────────────────────────────────────────────

function getSnippet(text: string, queryTerms: string[], maxLen = 240): string {
  const lower = text.toLowerCase();
  let anchor = -1;
  for (const term of queryTerms) {
    const pos = lower.indexOf(term);
    if (pos >= 0 && (anchor < 0 || pos < anchor)) anchor = pos;
  }
  if (anchor < 0) anchor = 0;

  const start = Math.max(0, anchor - 60);
  const end   = Math.min(text.length, start + maxLen);
  let   snip  = text.slice(start, end).trim();
  if (start > 0)         snip = "…" + snip;
  if (end < text.length) snip = snip + "…";
  return snip;
}

// ── Public API ─────────────────────────────────────────────────────────────────

/**
 * Search the ITC 2026 AI corpus for files relevant to `query`.
 *
 * On the first call (or when `forceRebuild` is true) the index is built from
 * the `corpus/` directory and written to `corpus_index.cache.json`.
 * All subsequent calls within the same process — and future process starts —
 * deserialise the cache and complete in under a second.
 *
 * @param query        Natural-language search string
 * @param topK         Maximum results to return (default: 5)
 * @param forceRebuild Ignore existing cache and rebuild from scratch
 * @returns            Results ranked by cosine similarity, highest first
 *
 * @example
 *   import { search } from "./search_corpus";
 *
 *   const results = search("transfer admission requirements");
 *   for (const r of results) {
 *     console.log(r.rank, r.filename, r.score);
 *     console.log("  ", r.snippet);
 *   }
 */
export function search(
  query:        string,
  topK          = 5,
  forceRebuild  = false,
): SearchResult[] {
  const { vocab, inverted, norms, filenames, texts } = getIndex(forceRebuild);

  // Vectorise query (raw TF — sublinear dampening is less important for short inputs)
  const qTokens = ngrams(tokenize(query));
  const qTf = new Map<number, number>();
  for (const term of qTokens) {
    const col = vocab.get(term);
    if (col !== undefined) qTf.set(col, (qTf.get(col) ?? 0) + 1);
  }

  if (qTf.size === 0) return []; // none of the query terms are in the vocabulary

  let qNorm = 0;
  for (const count of qTf.values()) qNorm += count * count;
  qNorm = Math.sqrt(qNorm);

  // Sparse dot product via inverted index — only touches docs that share a term
  const scores = new Float64Array(norms.length); // zero-initialised
  for (const [col, qw] of qTf) {
    const postings = inverted.get(col);
    if (!postings) continue;
    for (const [d, dw] of postings) scores[d] += qw * dw;
  }

  // Normalise to cosine similarity
  for (let d = 0; d < scores.length; d++) {
    if (norms[d] > 0) scores[d] /= qNorm * norms[d];
  }

  // Collect non-zero, sort descending, slice top-K
  const ranked: Array<{ i: number; score: number }> = [];
  for (let i = 0; i < scores.length; i++) {
    if (scores[i] > 1e-9) ranked.push({ i, score: scores[i] });
  }
  ranked.sort((a, b) => b.score - a.score);

  const queryTerms = query.toLowerCase().split(/\s+/).filter(t => t.length > 2);

  return ranked.slice(0, topK).map(({ i, score }, rank) => ({
    rank:     rank + 1,
    filename: filenames[i],
    score:    Math.round(score * 10_000) / 10_000,
    snippet:  getSnippet(texts[i], queryTerms),
  }));
}

// ── CLI ───────────────────────────────────────────────────────────────────────

if (require.main === module) {
  const argv = process.argv.slice(2);

  const forceRebuild = argv.includes("--rebuild");
  const topIdx       = argv.indexOf("--top");
  const topK         = topIdx >= 0 ? parseInt(argv[topIdx + 1] ?? "5", 10) : 5;
  const queryParts   = argv.filter(
    (a, i) => a !== "--rebuild" && a !== "--top" && argv[i - 1] !== "--top"
  );
  const query = queryParts.join(" ").trim();

  if (!query) {
    console.log('Usage: ts-node search_corpus.ts [--rebuild] [--top N] "<query>"');
    process.exit(1);
  }

  const results = search(query, topK, forceRebuild);

  if (results.length === 0) {
    console.log("No results found.");
  } else {
    for (const r of results) {
      console.log(`#${r.rank}  [${r.score.toFixed(4)}]  ${r.filename}`);
      console.log(`     ${r.snippet}\n`);
    }
  }
}