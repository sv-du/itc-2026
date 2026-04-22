/**
 * server.ts
 * ─────────────────────────────────────────────────────────────────────────────
 * Express HTTP server for the Campus Knowledge Agent.
 *
 * Routes
 * ──────
 *   POST /api/chat          Multi-turn conversation with the agent
 *   GET  /api/health        Health / readiness check
 */

import express, { Request, Response, NextFunction } from "express";
import path    from "path";
import fs      from "fs";
import dotenv  from "dotenv";
import cors    from "cors";

import { runAgent, ConversationMessage } from "./agent";

dotenv.config();

// ── App setup ─────────────────────────────────────────────────────────────────

const app  = express();
const PORT = 80;

app.use(cors());
app.use(express.json({ limit: "2mb" }));

// ── Request validation helpers ────────────────────────────────────────────────

function getApiKey(): string {
  const env = process.env.CLAUDE_API_KEY;
  if (env && env.startsWith("sk-ant-")) return env;
  throw new Error("No Anthropic API key provided. Set CLAUDE_API_KEY in .env.");
}

function validateHistory(history: unknown): ConversationMessage[] {
  if (!Array.isArray(history)) return [];
  return history.filter(
    (m): m is ConversationMessage =>
      typeof m === "object" &&
      m !== null &&
      (m as ConversationMessage).role    in { user: 1, assistant: 1 } &&
      typeof (m as ConversationMessage).content === "string",
  );
}

// ── POST /api/chat ────────────────────────────────────────────────────────────

interface ChatRequestBody {
  message: string;
  history?: unknown[];
}

app.post("/api/chat", async (req: Request, res: Response) => {
  try {
    const { message, history: rawHistory } = req.body as ChatRequestBody;

    if (!message || typeof message !== "string" || !message.trim()) {
      res.status(400).json({ error: "message is required" });
      return;
    }

    const apiKey = getApiKey();
    const history = validateHistory(rawHistory ?? []);

    // Soft cap on history length to stay within context window
    const trimmedHistory = history.slice(-20);

    const result = await runAgent(trimmedHistory, message.trim(), apiKey);

    res.json({
      answer:    result.answer,
      sources:   result.sources,
      toolCalls: result.toolCalls,
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    console.error("[/api/chat]", message);

    // Surface API auth errors clearly
    if (message.includes("API key") || message.includes("authentication") || message.includes("401")) {
      res.status(401).json({ error: "Invalid or missing Anthropic API key." });
    } else {
      res.status(500).json({ error: message });
    }
  }
});

// ── GET /api/health ───────────────────────────────────────────────────────────

app.get("/api/health", (_req: Request, res: Response) => {
  const corpusDir  = path.join(process.cwd(), "corpus");
  const cacheFile  = path.join(process.cwd(), "corpus_index.cache.json");

  res.json({
    status:       "ok",
    timestamp:    new Date().toISOString(),
    corpus:       fs.existsSync(corpusDir)  ? "present" : "missing",
    searchCache:  fs.existsSync(cacheFile)  ? "present" : "missing (will build on first query)",
    model:        "claude-sonnet-4-20250514",
    apiKeySet:    !!(process.env.CLAUDE_API_KEY),
  });
});

app.get("/", (req, res) => {
  res.sendFile(path.join(process.cwd(), "public", "index.html"));
});

// ── 404 & error handlers ──────────────────────────────────────────────────────

app.use((_req: Request, res: Response) => {
  res.status(404).json({ error: "Not found" });
});

// eslint-disable-next-line @typescript-eslint/no-unused-vars
app.use((err: Error, _req: Request, res: Response, _next: NextFunction) => {
  console.error("[unhandled]", err);
  res.status(500).json({ error: err.message });
});

// ── Start ─────────────────────────────────────────────────────────────────────

app.listen(PORT, () => {
  console.log(`🐴 Campus Knowledge Agent — backend running\n`);
});

export default app;