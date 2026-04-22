/**
 * agent.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Campus Knowledge Agent — Anthropic tool-calling loop.
 *
 * Tools exposed to the model
 * ──────────────────────────
 *   search_corpus    TF-IDF keyword search → ranked filenames + snippets
 *   read_document    Fetch full cleaned text + source URL for one file
 *
 * The agent loop runs until the model emits a final text response
 * (stop_reason === "end_turn") or a hard iteration cap is hit.
 */

import Anthropic from "@anthropic-ai/sdk";
import * as fs   from "fs";
import * as path from "path";
import { search, SearchResult } from "./search_corpus";

// ── Config ────────────────────────────────────────────────────────────────────

const CORPUS_DIR  = path.join(process.cwd(), "corpus");
const INDEX_FILE  = path.join(CORPUS_DIR, "index.json");
const MAX_ITERS   = 10;   // max tool-call rounds per user turn
const MAX_TOKENS  = 4000;
const MODEL       = "claude-sonnet-4-20250514";

// ── URL index ─────────────────────────────────────────────────────────────────

/** filename → canonical URL (prefer https) */
const urlMap: Record<string, string> = (() => {
  if (!fs.existsSync(INDEX_FILE)) return {};
  const raw = JSON.parse(fs.readFileSync(INDEX_FILE, "utf8")) as Record<string, string>;
  const map: Record<string, string> = {};
  for (const [url, fname] of Object.entries(raw)) {
    if (!(fname in map) || (url.startsWith("https") && !map[fname].startsWith("https"))) {
      map[fname] = url;
    }
  }
  return map;
})();

function filenameToUrl(filename: string): string {
  return urlMap[filename] ?? `https://www.cpp.edu/ (${filename})`;
}

// ── Tool: read_document ───────────────────────────────────────────────────────

/** Read and clean a corpus file, returning content + source URL. */
function readDocument(filename: string): { content: string; url: string; found: boolean } {
  const filepath = path.join(CORPUS_DIR, filename);
  if (!fs.existsSync(filepath)) {
    return { content: "", url: "", found: false };
  }
  const raw     = fs.readFileSync(filepath, "utf8");
  const content = cleanForAgent(raw);
  return { content: content.slice(0, 6000), url: filenameToUrl(filename), found: true };
}

/** Light cleaning: strip boilerplate nav, keep real content. */
function cleanForAgent(raw: string): string {
  const lines = raw.split(/\r?\n/);
  let start   = 0;
  for (let i = 0; i < lines.length; i++) {
    if (/^#{1,2}\s+\S/.test(lines[i].trim())) { start = i; break; }
  }
  return lines
    .slice(start)
    .map(l => l.trim())
    .filter(l => l && !/^!\[/.test(l) && !/^\[.*?\]\(https?:/.test(l))
    .map(l => l.replace(/!\[.*?\]\(.*?\)/g, "").replace(/\[([^\]]+)\]\([^)]*\)/g, "$1").trim())
    .filter(Boolean)
    .join("\n");
}

// ── Tool definitions (Anthropic schema) ──────────────────────────────────────

const TOOLS: Anthropic.Tool[] = [
  {
    name: "search_corpus",
    description:
      "Search the Cal Poly Pomona knowledge base for pages relevant to a query. " +
      "Returns ranked results with filenames, relevance scores, and text snippets. " +
      "Use this first to discover which documents contain relevant information.",
    input_schema: {
      type: "object" as const,
      properties: {
        query: {
          type: "string",
          description: "Natural language search query, e.g. 'transfer admission requirements' or 'financial aid deadlines'",
        },
        top_k: {
          type: "number",
          description: "Number of results to return (default 5, max 10)",
        },
      },
      required: ["query"],
    },
  },
  {
    name: "read_document",
    description:
      "Fetch the full text content of a specific corpus document by filename. " +
      "Use this after search_corpus to read the complete content of a promising result " +
      "and get its source URL for attribution.",
    input_schema: {
      type: "object" as const,
      properties: {
        filename: {
          type: "string",
          description: "The exact filename returned by search_corpus, e.g. 'admissions__transfer.md'",
        },
      },
      required: ["filename"],
    },
  },
];

// ── Tool dispatcher ───────────────────────────────────────────────────────────

interface ToolInput {
  query?:    string;
  top_k?:   number;
  filename?: string;
}

function dispatchTool(name: string, input: ToolInput): string {
  if (name === "search_corpus") {
    const query  = input.query ?? "";
    const topK   = Math.min(input.top_k ?? 5, 10);
    const results: SearchResult[] = search(query, topK);

    if (results.length === 0) {
      return JSON.stringify({ results: [], message: "No matching documents found for this query." });
    }

    return JSON.stringify({
      results: results.map(r => ({
        rank:     r.rank,
        filename: r.filename,
        score:    r.score,
        snippet:  r.snippet,
        url:      filenameToUrl(r.filename),
      })),
    });
  }

  if (name === "read_document") {
    const filename = input.filename ?? "";
    const result   = readDocument(filename);

    if (!result.found) {
      return JSON.stringify({ error: `File not found: ${filename}` });
    }
    return JSON.stringify({ filename, url: result.url, content: result.content });
  }

  return JSON.stringify({ error: `Unknown tool: ${name}` });
}

// ── Types ─────────────────────────────────────────────────────────────────────

export type ConversationMessage =
  | { role: "user";      content: string }
  | { role: "assistant"; content: string };

export interface AgentResponse {
  answer:    string;
  sources:   Array<{ filename: string; url: string; title: string }>;
  toolCalls: Array<{ tool: string; input: unknown; resultSummary: string }>;
}

// ── System prompt ─────────────────────────────────────────────────────────────

const SYSTEM_PROMPT = `\
You are BroncoBot, a friendly and knowledgeable campus assistant for Cal Poly Pomona (CPP). \
You help students, prospective students, and visitors find accurate information about the university.

## How to answer questions

1. Use the **search_corpus** tool first to find relevant documents.
2. Use **read_document** on the 1-3 most promising results to get full details. After reading 2-3 documents, stop searching and write your answer. Do not call more than ${MAX_ITERS - 3} tools total.
3. Base your answer ONLY on what you find in the corpus. Do not invent facts.
4. If the corpus doesn't contain an answer, say so clearly and suggest the student contact the relevant office.
5. Always cite your sources: mention which page(s) the info came from and include the URL.

## Tone & style
- Friendly, helpful, and concise
- Use plain language — avoid jargon
- Format answers with short paragraphs or bullet points as appropriate
- For complex multi-part answers, use headers

## Source citation format
End every answer with a "📚 Sources" section listing the page titles and URLs you used.
If you couldn't find an answer, end with a "📞 Contact" section suggesting who to call or email.`;

// ── Agent loop ────────────────────────────────────────────────────────────────

export async function runAgent(
  history: ConversationMessage[],
  userMessage: string,
  apiKey: string,
): Promise<AgentResponse> {
  const client = new Anthropic({ apiKey });

  // Build the messages array for the API
  const messages: Anthropic.MessageParam[] = [
    ...history.map(m => ({ role: m.role as "user" | "assistant", content: m.content })),
    { role: "user", content: userMessage },
  ];

  const sourcesUsed   = new Map<string, { filename: string; url: string; title: string }>();
  const toolCallLog: AgentResponse["toolCalls"] = [];

  let iterCount = 0;

  // Agentic loop
  while (iterCount < MAX_ITERS) {
    iterCount++;
console.log(`[agent] iter ${iterCount}, messages so far:`, messages.length);
    const response = await client.messages.create({
      model:      MODEL,
      max_tokens: MAX_TOKENS,
      system:     SYSTEM_PROMPT,
      tools:      TOOLS,
      messages,
    });
 console.log(`[agent] stop_reason: ${response.stop_reason}`);
  console.log(`[agent] content:`, JSON.stringify(response.content, null, 2));
    // If the model is done, extract and return the final text
    if (response.stop_reason === "end_turn") {
      const textBlock = response.content.find(b => b.type === "text");
      const answer    = textBlock?.type === "text" ? textBlock.text : "(no response)";
      return { answer, sources: [...sourcesUsed.values()], toolCalls: toolCallLog };
    }

    // Process tool use blocks
    if (response.stop_reason === "tool_use") {
      // Append the assistant turn with all its content blocks
      messages.push({ role: "assistant", content: response.content });

      // Execute every tool call and collect results
      const toolResults: Anthropic.ToolResultBlockParam[] = [];

      for (const block of response.content) {
        if (block.type !== "tool_use") continue;

        const input  = block.input as ToolInput;
        const result = dispatchTool(block.name, input);
console.log(`[agent] tool: ${block.name}, input:`, input);
console.log(`[agent] result (first 200):`, result.slice(0, 200));
        // Track sources for read_document calls
        if (block.name === "read_document" && input.filename) {
          const parsed = JSON.parse(result) as { url?: string; content?: string };
          if (parsed.url) {
            const title = extractTitle(parsed.content ?? "", input.filename);
            sourcesUsed.set(input.filename, { filename: input.filename, url: parsed.url, title });
          }
        }

        // Track sources surfaced by search_corpus
        if (block.name === "search_corpus") {
          const parsed = JSON.parse(result) as { results?: Array<{ filename: string; url: string; snippet: string }> };
          // We'll record sources properly when read_document is called; just log the call here
        }

        toolCallLog.push({
          tool:          block.name,
          input,
          resultSummary: summariseResult(block.name, result),
        });

        toolResults.push({
          type:        "tool_result",
          tool_use_id: block.id,
          content:     result,
        });
      }

      // Feed results back to the model
      messages.push({ role: "user", content: toolResults });
      continue;
    }

    // Any other stop reason — just return what we have
    const textBlock = response.content.find(b => b.type === "text");
    const answer    = textBlock?.type === "text" ? textBlock.text : "(unexpected stop reason)";
    return { answer, sources: [...sourcesUsed.values()], toolCalls: toolCallLog };
  }

  return {
    answer:    "I ran into a problem processing your request. Please try again.",
    sources:   [],
    toolCalls: toolCallLog,
  };
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function extractTitle(content: string, fallback: string): string {
  const match = content.match(/^#+ (.+)/m);
  if (match) return match[1].trim();
  return fallback.replace(/\.md$/, "").replace(/__/g, " / ").replace(/_/g, " ");
}

function summariseResult(toolName: string, result: string): string {
  try {
    const parsed = JSON.parse(result);
    if (toolName === "search_corpus") {
      const n = parsed.results?.length ?? 0;
      return n > 0
        ? `Found ${n} results: ${parsed.results.map((r: { filename: string }) => r.filename).join(", ")}`
        : "No results found";
    }
    if (toolName === "read_document") {
      if (parsed.error) return parsed.error;
      const chars = parsed.content?.length ?? 0;
      return `Read ${chars} chars from ${parsed.filename} (${parsed.url})`;
    }
  } catch { /* ignore */ }
  return result.slice(0, 120);
}