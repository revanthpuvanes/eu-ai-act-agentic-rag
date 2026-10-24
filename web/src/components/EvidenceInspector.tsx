"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ArrowUpRight,
  BookOpen,
  CheckCircle2,
  FileText,
  LoaderCircle,
  MapPin,
  X,
} from "lucide-react";

type Article = {
  article_number: number;
  article_id: string | null;
  title: string;
  text: string;
  source_url: string | null;
  source_sha256: string | null;
};

export type EvidencePassage = {
  id: number;
  article_number: number;
  text: string;
  start_char?: number | null;
  end_char?: number | null;
  source_url?: string | null;
};

type Props = {
  articleNumber: number | null;
  onClose: () => void;
  evidence?: EvidencePassage[];
};

type Range = {
  start: number;
  end: number;
};

const API =
  process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

const OFFICIAL_REGULATION =
  "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689";

function findEvidenceRanges(
  articleText: string,
  passages: EvidencePassage[]
): Range[] {
  const ranges: Range[] = [];

  for (const passage of passages) {
    if (!passage.text.trim()) continue;

    // Prefer verified offsets when they refer to the same text.
    const start = passage.start_char;
    const end = passage.end_char;

    if (
      typeof start === "number" &&
      typeof end === "number" &&
      start >= 0 &&
      end <= articleText.length &&
      start < end &&
      articleText.slice(start, end) === passage.text
    ) {
      ranges.push({ start, end });
      continue;
    }

    // Otherwise locate the exact passage text in the article.
    const exactIndex = articleText.indexOf(passage.text);

    if (exactIndex !== -1) {
      ranges.push({
        start: exactIndex,
        end: exactIndex + passage.text.length,
      });
    }
  }

  // Merge overlapping ranges so React renders clean segments.
  ranges.sort((a, b) => a.start - b.start);

  const merged: Range[] = [];

  for (const range of ranges) {
    const previous = merged[merged.length - 1];

    if (previous && range.start <= previous.end) {
      previous.end = Math.max(previous.end, range.end);
    } else {
      merged.push({ ...range });
    }
  }

  return merged;
}

function HighlightedArticle({
  text,
  ranges,
}: {
  text: string;
  ranges: Range[];
}) {
  if (ranges.length === 0) {
    return <>{text}</>;
  }

  const segments: React.ReactNode[] = [];
  let cursor = 0;

  ranges.forEach((range, index) => {
    if (range.start > cursor) {
      segments.push(
        <span key={`plain-${index}`}>
          {text.slice(cursor, range.start)}
        </span>
      );
    }

    segments.push(
      <mark
        key={`highlight-${index}`}
        className="rounded-sm bg-blue-500/25 px-0.5 text-blue-100 ring-1 ring-blue-400/30"
      >
        {text.slice(range.start, range.end)}
      </mark>
    );

    cursor = range.end;
  });

  if (cursor < text.length) {
    segments.push(
      <span key="plain-final">{text.slice(cursor)}</span>
    );
  }

  return <>{segments}</>;
}

export default function EvidenceInspector({
  articleNumber,
  onClose,
  evidence = [],
}: Props) {
  const [article, setArticle] = useState<Article | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const selectedPassages = useMemo(
    () =>
      evidence.filter(
        (passage) => passage.article_number === articleNumber
      ),
    [evidence, articleNumber]
  );

  const ranges = useMemo(
    () =>
      article
        ? findEvidenceRanges(article.text, selectedPassages)
        : [],
    [article, selectedPassages]
  );

  useEffect(() => {
    if (articleNumber === null) return;

    const controller = new AbortController();

    async function loadArticle() {
      setArticle(null);
      setError("");
      setLoading(true);

      try {
        const response = await fetch(
          `${API}/evidence/articles/${articleNumber}`,
          { signal: controller.signal }
        );

        if (!response.ok) {
          throw new Error(`Article request failed: ${response.status}`);
        }

        const data: Article = await response.json();

        if (!controller.signal.aborted) {
          setArticle(data);
        }
      } catch (err) {
        if (!controller.signal.aborted) {
          setError(
            err instanceof Error
              ? err.message
              : "Unable to load article."
          );
        }
      } finally {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      }
    }

    void loadArticle();

    return () => controller.abort();
  }, [articleNumber]);

  if (articleNumber === null) return null;

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/70 backdrop-blur-sm">
      <button
        type="button"
        aria-label="Close evidence inspector"
        onClick={onClose}
        className="absolute inset-0 cursor-default"
      />

      <aside
        role="dialog"
        aria-modal="true"
        aria-label={`Article ${articleNumber} evidence`}
        className="relative z-10 flex h-full w-full max-w-2xl flex-col border-l border-white/10 bg-[#0d1524] shadow-2xl"
      >
        <div className="flex items-start justify-between border-b border-white/10 p-6">
          <div>
            <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-widest text-blue-400">
              <BookOpen size={15} />
              Evidence provenance
            </div>

            <h2 className="text-xl font-bold">
              Article {articleNumber}
            </h2>

            <p className="mt-1 text-xs text-slate-400">
              Selected agent evidence + original EU AI Act text
            </p>
          </div>

          <button
            type="button"
            onClick={onClose}
            aria-label="Close inspector"
            className="rounded-lg border border-white/10 p-2 hover:bg-white/10"
          >
            <X size={20} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-6">
          {loading && (
            <div className="flex items-center gap-3 text-sm text-slate-400">
              <LoaderCircle
                size={18}
                className="animate-spin text-blue-400"
              />
              Loading official legal text...
            </div>
          )}

          {error && (
            <div className="rounded-xl border border-red-500/20 bg-red-500/10 p-4 text-sm text-red-300">
              {error}
            </div>
          )}

          {article && (
            <>
              <div className="mb-6 rounded-xl border border-blue-500/20 bg-blue-500/10 p-5">
                <div className="mb-2 flex items-center gap-2 text-xs text-blue-300">
                  <FileText size={15} />
                  ARTICLE TITLE
                </div>

                <h3 className="text-lg font-semibold">
                  {article.title}
                </h3>

                <p className="mt-3 text-xs text-slate-400">
                  {article.text.length.toLocaleString()} characters
                  {" · "}
                  Source SHA-256 recorded
                </p>
              </div>

              <section className="mb-7">
                <div className="mb-3 flex items-center justify-between gap-3">
                  <h3 className="flex items-center gap-2 text-sm font-semibold">
                    <CheckCircle2
                      size={16}
                      className="text-cyan-400"
                    />
                    Selected agent evidence
                  </h3>

                  <span className="rounded-full border border-white/10 px-2 py-1 text-[10px] text-slate-400">
                    {selectedPassages.length} passages
                  </span>
                </div>

                {selectedPassages.length === 0 ? (
                  <div className="rounded-xl border border-white/10 bg-[#080d19] p-4 text-xs leading-6 text-slate-400">
                    No selected passage data was supplied for this
                    article. The original regulation text is still
                    available below.
                  </div>
                ) : (
                  <div className="space-y-3">
                    {selectedPassages.map((passage, index) => (
                      <div
                        key={`${passage.id}-${index}`}
                        className="rounded-xl border border-cyan-500/20 bg-cyan-500/[0.06] p-4"
                      >
                        <div className="mb-3 flex items-center justify-between gap-2">
                          <span className="text-xs font-semibold text-cyan-300">
                            Evidence passage {index + 1}
                          </span>

                          {typeof passage.start_char === "number" &&
                            typeof passage.end_char === "number" && (
                              <span className="flex items-center gap-1 font-mono text-[10px] text-slate-500">
                                <MapPin size={11} />
                                {passage.start_char}–
                                {passage.end_char}
                              </span>
                            )}
                        </div>

                        <p className="whitespace-pre-wrap text-xs leading-6 text-slate-300">
                          {passage.text}
                        </p>
                      </div>
                    ))}
                  </div>
                )}
              </section>

              <div className="mb-4 flex items-center justify-between gap-3">
                <h3 className="text-sm font-semibold">
                  Original regulation text
                </h3>

                <a
                  href={article.source_url || OFFICIAL_REGULATION}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1 text-xs text-blue-400 hover:text-blue-300"
                >
                  EUR-Lex <ArrowUpRight size={14} />
                </a>
              </div>

              {selectedPassages.length > 0 && (
                <p className="mb-3 text-xs leading-5 text-slate-500">
                  {ranges.length > 0
                    ? "Blue highlights identify exact matches between selected evidence and the stored article."
                    : "No exact passage match was verified in the stored article; the original text is shown without highlights."}
                </p>
              )}

              <div className="whitespace-pre-wrap rounded-xl border border-white/10 bg-[#080d19] p-5 text-sm leading-8 text-slate-300">
                <HighlightedArticle
                  text={article.text}
                  ranges={ranges}
                />
              </div>

              <p className="mt-5 text-xs leading-6 text-slate-500">
                Evidence passages come from the agent's final
                selected chunks. The article text comes from the
                ingested legal corpus. Highlighting requires an
                exact text match and does not use an additional
                language model.
              </p>
            </>
          )}
        </div>
      </aside>
    </div>
  );
}
