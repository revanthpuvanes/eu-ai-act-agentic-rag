"use client";

import { useState } from "react";
import LiveAgentGraph from "@/components/LiveAgentGraph";
import BenchmarkDashboard from "@/components/BenchmarkDashboard";
import EngineeringStory from "@/components/EngineeringStory";
import EvidenceInspector, { type EvidencePassage } from "@/components/EvidenceInspector";
import {
  Activity,
  ArrowUpRight,
  BookOpen,
  BrainCircuit,
  CheckCircle2,
  LoaderCircle,
  Scale,
  Send,
  Sparkles,
  Zap,
} from "lucide-react";

type Step = {
  node: string;
  elapsed_seconds?: number;
  assessment?: {
    action?: string;
    reason?: string;
  };
  action_taken?: string;
};

type Result = {
  evidence?: EvidencePassage[];
  answer: string;
  cited_articles: number[];
  retrieved_articles: number[];
  abstained: boolean;
  action_taken: string;
  total_seconds: number;
  total_tokens: number;
  model: string;
};

const API =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

const LAW_URL =
  "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689";

const EXAMPLES = [
  "Which AI practices are prohibited under Article 5?",
  "What obligations apply to high-risk AI providers?",
  "When does the EU AI Act become applicable?",
];

export default function Home() {
  const [question, setQuestion] = useState("");
  const [selectedArticle, setSelectedArticle] = useState<number | null>(null);
  const [steps, setSteps] = useState<Step[]>([]);
  const [result, setResult] = useState<Result | null>(null);
  const [loading, setLoading] = useState(false);
  const [waitingForCloudRun, setWaitingForCloudRun] = useState(false);
  const [error, setError] = useState("");

  async function investigate(value?: string) {
    const text = (value || question).trim();

    if (loading || text.length < 10) return;

    setQuestion(text);
    setSteps([]);
    setResult(null);
    setError("");
    setLoading(true);
    setWaitingForCloudRun(true);

    try {
      const response = await fetch(`${API}/ask/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text }),
      });

      if (!response.ok) {
        throw new Error(
          response.status === 429
            ? "The agent is busy. Try again shortly."
            : `API error: ${response.status}`
        );
      }

      if (!response.body) {
        throw new Error("Streaming is unavailable.");
      }

      setWaitingForCloudRun(false);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      let buffer = "";
      let finished = false;

      while (true) {
        const { value: chunk, done } = await reader.read();

        if (done) break;

        buffer += decoder.decode(chunk, { stream: true });
        buffer = buffer.replace(/\r\n/g, "\n");

        let boundary = buffer.indexOf("\n\n");

        while (boundary !== -1) {
          const block = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);

          const eventName = block
            .split("\n")
            .find((line) => line.startsWith("event:"))
            ?.slice(6)
            .trim();

          const dataText = block
            .split("\n")
            .filter((line) => line.startsWith("data:"))
            .map((line) => line.slice(5).trim())
            .join("\n");

          if (dataText) {
            const data = JSON.parse(dataText);

            if (eventName === "node_completed") {
              setSteps((previous) => [...previous, data as Step]);
            }

            if (eventName === "completed") {
              setResult(data as Result);
              finished = true;
            }

            if (eventName === "error") {
              throw new Error(data.message || "Agent execution failed.");
            }
          }

          boundary = buffer.indexOf("\n\n");
        }
      }

      if (!finished) {
        throw new Error("The agent stream ended before completion.");
      }
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Unexpected error."
      );
    } finally {
      setLoading(false);
      setWaitingForCloudRun(false);
    }
  }

  return (
    <main className="min-h-screen bg-[#080d19] text-slate-100">
      {waitingForCloudRun && (
        <div
          role="status"
          aria-live="polite"
          className="fixed bottom-6 left-1/2 z-50 w-[calc(100%-2rem)] max-w-lg -translate-x-1/2 rounded-2xl border border-blue-500/30 bg-[#111b30] p-5 text-left shadow-2xl shadow-black/50"
        >
          <div className="flex items-start gap-3">
            <LoaderCircle
              size={22}
              className="mt-0.5 shrink-0 animate-spin text-blue-400"
            />
            <div>
              <p className="font-semibold text-white">
                Initializing AI Research Engine
              </p>
              <p className="mt-2 text-sm leading-relaxed text-slate-300">
                Our LangGraph agent runs on Google Cloud Run with
                automatic scale-to-zero to optimize cloud costs.
                After inactivity, the first investigation may take
                15–25 seconds while the container starts and AI
                models load.
              </p>
              <p className="mt-2 text-xs text-blue-300">
                Subsequent investigations are typically faster.
              </p>
            </div>
          </div>
        </div>
      )}
      <div className="mx-auto max-w-6xl px-5 pb-20">
        <header className="flex items-center justify-between border-b border-white/10 py-6">
          <div className="flex items-center gap-3">
            <div className="rounded-xl bg-blue-600 p-2.5">
              <Scale size={22} />
            </div>
            <div>
              <h1 className="font-bold">
                EU AI Act <span className="text-blue-400">Intelligence Lab</span>
              </h1>
              <p className="text-xs text-slate-500">
                Live evidence-grounded AI research
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-end gap-2">
            <span className="whitespace-nowrap rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1.5 text-xs font-medium text-emerald-300">
              ● Live LangGraph
            </span>

            <span
              title="Langfuse provides backend tracing and observability for agent executions"
              className="whitespace-nowrap rounded-full border border-purple-500/30 bg-purple-500/10 px-3 py-1.5 text-xs font-medium text-purple-300"
            >
              ◈ Observability: Langfuse
            </span>
          </div>
        </header>

        <section className="py-16 text-center">
          <div className="mb-5 inline-flex items-center gap-2 rounded-full border border-blue-500/30 bg-blue-500/10 px-4 py-2 text-xs text-blue-300">
            <Sparkles size={14} />
            REAL AGENT · REAL EVIDENCE · REAL DECISIONS
          </div>

          <h2 className="text-4xl font-bold tracking-tight sm:text-6xl">
            Ask the regulation.
            <br />
            <span className="text-blue-400">
              Inspect the intelligence.
            </span>
          </h2>

          <p className="mx-auto mt-6 max-w-2xl text-slate-400">
            Investigate the EU AI Act using a live LangGraph agent
            powered by hybrid retrieval and Mistral.
            Watch each completed agent step as it happens.
          </p>
        </section>

        <div className="grid items-start gap-6 lg:grid-cols-2">
          <div className="space-y-6">
            <section className="rounded-2xl border border-white/10 bg-[#101827] p-6">
              <h3 className="mb-5 flex items-center gap-2 font-semibold">
                <BrainCircuit size={18} className="text-blue-400" />
                Research workspace
              </h3>

              <form
                onSubmit={(event) => {
                  event.preventDefault();
                  void investigate();
                }}
              >
                <textarea
                  value={question}
                  onChange={(event) => setQuestion(event.target.value)}
                  maxLength={1000}
                  rows={4}
                  placeholder="Ask a question about the EU AI Act..."
                  className="w-full resize-none rounded-xl border border-white/10 bg-[#0b1220] p-4 text-sm leading-7 outline-none focus:border-blue-500"
                />

                <div className="mt-2 flex items-center justify-between gap-3 text-xs">
                  <span
                    className={
                      question.trim().length >= 10
                        ? "text-emerald-400"
                        : "text-slate-500"
                    }
                  >
                    {question.trim().length >= 10
                      ? "Ready to investigate"
                      : "Enter at least 10 characters"}
                  </span>
                  <span className="text-slate-600">
                    {question.length}/1000
                  </span>
                </div>

                <div className="mt-4 flex justify-end">
                  <button
                    type="submit"
                    disabled={loading || question.trim().length < 10}
                    className={`flex items-center gap-2 rounded-lg px-5 py-3 text-sm font-semibold transition-all duration-200 ${
                      loading
                        ? "cursor-wait bg-blue-600/60 text-white"
                        : question.trim().length >= 10
                          ? "bg-blue-600 text-white shadow-lg shadow-blue-500/25 hover:bg-blue-500 hover:shadow-blue-500/40"
                          : "cursor-not-allowed border border-white/10 bg-[#1b2638] text-slate-500"
                    }`}
                  >
                    {loading ? (
                      <LoaderCircle size={16} className="animate-spin" />
                    ) : (
                      <Send size={16} />
                    )}
                    {loading ? "Investigating..." : "Investigate"}
                  </button>
                </div>
              </form>

              <div className="mt-6 border-t border-white/10 pt-5">
                <p className="mb-3 text-xs uppercase tracking-wider text-slate-500">
                  Try a live investigation
                </p>

                <div className="space-y-2">
                  {EXAMPLES.map((example) => (
                    <button
                      key={example}
                      disabled={loading}
                      onClick={() => void investigate(example)}
                      className="flex w-full items-center justify-between rounded-lg border border-white/10 bg-white/[0.03] p-3 text-left text-xs text-slate-300 hover:border-blue-500/40 disabled:opacity-40"
                    >
                      {example}
                      <ArrowUpRight size={14} />
                    </button>
                  ))}
                </div>
              </div>
            </section>

            <section className="min-h-64 rounded-2xl border border-white/10 bg-[#101827] p-6">
              {result && result.cited_articles.length > 0 && (
                <div className="mb-5 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-blue-500/25 bg-blue-500/[0.08] p-3">
                  <div className="flex items-center gap-2 text-xs text-blue-200">
                    <BookOpen size={16} />
                    Click a cited article to inspect the original law and selected evidence.
                  </div>
                  <button
                    type="button"
                    onClick={() => setSelectedArticle(result.cited_articles[0])}
                    className="rounded-lg bg-blue-600 px-3 py-2 text-xs font-semibold text-white transition-colors hover:bg-blue-500"
                  >
                    Inspect legal evidence ↗
                  </button>
                </div>
              )}
              <h3 className="mb-5 flex items-center gap-2 font-semibold">
                <BookOpen size={18} className="text-blue-400" />
                Research findings
              </h3>

              {!result && !loading && !error && (
                <p className="py-12 text-center text-sm text-slate-500">
                  Submit a question to see a live research answer.
                </p>
              )}

              {loading && (
                <div className="flex items-center gap-3 text-sm text-slate-400">
                  <LoaderCircle size={18} className="animate-spin text-blue-400" />
                  Research in progress...
                </div>
              )}

              {error && (
                <p role="alert" className="rounded-lg bg-red-500/10 p-4 text-sm text-red-300">
                  {error}
                </p>
              )}

              {result && (
                <div className="space-y-5">
                  <div className="whitespace-pre-wrap text-sm leading-8 text-slate-200">
                    {result.answer}
                  </div>

                  <div className="border-t border-white/10 pt-4">
                    <p className="mb-3 text-xs text-slate-500">
                      CITED ARTICLES
                    </p>
                    <div className="flex flex-wrap gap-2">
                      {result.cited_articles.map((article) => (
                        <button
                          key={article}
                          type="button"
                          onClick={() => setSelectedArticle(article)}
                          className="rounded-lg border border-blue-500/30 bg-blue-500/10 px-3 py-2 text-xs text-blue-300 hover:bg-blue-500/20"
                        >
                          Article {article} ↗
                        </button>
                      ))}
                    </div>
                  </div>

                  <p className="text-xs text-slate-500">
                    AI-generated research, not legal advice.
                    Citations do not prove legal correctness.
                  </p>
                </div>
              )}
            </section>
          </div>

          <div className="space-y-6">
            <LiveAgentGraph steps={steps} loading={loading} />

            <section className="rounded-2xl border border-white/10 bg-[#101827] p-6">
              <h3 className="mb-5 flex items-center gap-2 font-semibold">
                <Activity size={18} className="text-cyan-400" />
                Live agent workflow
              </h3>

              {steps.length === 0 && !loading && (
                <p className="py-10 text-center text-sm text-slate-500">
                  Waiting for an investigation.
                </p>
              )}

              <div className="space-y-3">
                {steps.map((step, index) => (
                  <div
                    key={`${step.node}-${index}`}
                    className="rounded-xl border border-white/10 bg-[#0b1220] p-4"
                  >
                    <div className="flex items-center justify-between">
                      <span className="flex items-center gap-2 text-sm font-medium capitalize">
                        <CheckCircle2 size={16} className="text-emerald-400" />
                        {step.node.replaceAll("_", " ")}
                      </span>
                      <span className="font-mono text-xs text-slate-500">
                        {step.elapsed_seconds?.toFixed(2)}s
                      </span>
                    </div>

                    {step.assessment?.action && (
                      <p className="mt-2 text-xs text-cyan-300">
                        Decision: {step.assessment.action}
                      </p>
                    )}

                    {step.assessment?.reason && (
                      <p className="mt-2 text-xs leading-5 text-slate-400">
                        {step.assessment.reason}
                      </p>
                    )}

                    {step.action_taken && (
                      <p className="mt-2 text-xs text-blue-300">
                        Action: {step.action_taken}
                      </p>
                    )}
                  </div>
                ))}

                {loading && (
                  <div className="flex items-center gap-2 text-xs text-slate-400">
                    <LoaderCircle size={14} className="animate-spin" />
                    Waiting for next graph event...
                  </div>
                )}
              </div>
            </section>

            {result && (
              <section className="rounded-2xl border border-white/10 bg-[#101827] p-6">
                <h3 className="mb-5 flex items-center gap-2 font-semibold">
                  <Zap size={18} className="text-blue-400" />
                  Execution telemetry
                </h3>

                <div className="grid grid-cols-2 gap-3">
                  {[
                    ["Latency", `${result.total_seconds.toFixed(2)}s`],
                    ["Tokens", result.total_tokens.toLocaleString()],
                    ["Decision", result.action_taken || "—"],
                    ["Evidence articles", String(result.retrieved_articles.length)],
                  ].map(([label, value]) => (
                    <div
                      key={label}
                      className="rounded-xl bg-[#0b1220] p-4"
                    >
                      <div className="break-words text-lg font-semibold">
                        {value}
                      </div>
                      <div className="mt-1 text-xs text-slate-500">
                        {label}
                      </div>
                    </div>
                  ))}
                </div>

                <p className="mt-4 text-xs text-slate-500">
                  Model: {result.model}
                </p>
              </section>
            )}


          </div>
        </div>

        <BenchmarkDashboard />

        <EngineeringStory />

        <footer className="mt-16 border-t border-white/10 pt-6 text-center text-xs text-slate-500">
          EU AI Act Intelligence Lab · LangGraph · Mistral · FastAPI · Next.js
        </footer>
      </div>

      <EvidenceInspector
        articleNumber={selectedArticle}
        evidence={result?.evidence ?? []}
        onClose={() => setSelectedArticle(null)}
      />
    </main>
  );
}
