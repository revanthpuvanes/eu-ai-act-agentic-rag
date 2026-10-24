"use client";

import {
  ArrowUpRight,
  BookOpen,
  BrainCircuit,
  CheckCircle2,
  Clock3,
  Code2,
  Database,
  ExternalLink,
  GitBranch,
  Layers3,
  Scale,
  ShieldCheck,
  Sparkles,
  Target,
} from "lucide-react";

const GITHUB =
  "https://github.com/revanthpuvanes/eu-ai-act-agentic-rag";

const REGULATION =
  "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689";

const ARCHITECTURES = [
  {
    number: "01",
    name: "Naive RAG",
    subtitle: "The baseline",
    description:
      "Retrieve relevant chunks and generate a grounded answer. Simple, fast, and inexpensive, but limited when legal evidence spans multiple provisions.",
    tags: ["Hybrid retrieval", "Single generation"],
    icon: Database,
  },
  {
    number: "02",
    name: "Expanded RAG",
    subtitle: "More context",
    description:
      "Expand retrieved evidence into fuller article context before generation. Improves citation alignment, with a larger prompt and higher token usage.",
    tags: ["Article expansion", "Context budget"],
    icon: Layers3,
  },
  {
    number: "03",
    name: "Deterministic Agent",
    subtitle: "Rule-based control",
    description:
      "Use a LangGraph workflow with deterministic evidence assessment and conditional expansion. More structured behavior without a separate LLM routing decision.",
    tags: ["LangGraph", "Rule-based routing"],
    icon: GitBranch,
  },
  {
    number: "04",
    name: "LLM-Driven Agent",
    subtitle: "Adaptive reasoning",
    description:
      "Let a separate model assessment choose whether to answer, expand, search again, or abstain. Adds flexibility, but costs more tokens and time.",
    tags: ["LLM routing", "Conditional tools"],
    icon: BrainCircuit,
  },
];

const RESULTS = [
  {
    label: "Highest citation F1",
    value: "0.793",
    system: "LLM-driven agent",
    icon: Target,
  },
  {
    label: "Lowest mean latency",
    value: "1.53s",
    system: "Deterministic agent",
    icon: Clock3,
  },
  {
    label: "Highest execution success",
    value: "100%",
    system: "Deterministic agent",
    icon: CheckCircle2,
  },
];

export default function EngineeringStory() {
  return (
    <section
      id="engineering"
      className="mt-12 space-y-8"
    >
      <div className="border-t border-white/10 pt-10">
        <div className="mb-4 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.2em] text-blue-400">
          <Code2 size={16} />
          Behind the intelligence
        </div>

        <div className="flex flex-col justify-between gap-6 lg:flex-row lg:items-end">
          <div className="max-w-3xl">
            <h2 className="text-3xl font-bold tracking-tight sm:text-4xl">
              Not just an answer.
              <br />
              <span className="text-blue-400">
                An engineering experiment.
              </span>
            </h2>

            <p className="mt-5 max-w-2xl text-sm leading-7 text-slate-400">
              Legal research demands more than plausible text.
              An AI system must retrieve the right provisions,
              handle missing evidence, cite its sources, and
              expose the trade-offs behind its decisions.
              This lab compares four approaches to that problem
              using the EU AI Act as its source corpus.
            </p>
          </div>

          <a
            href={GITHUB}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex shrink-0 items-center justify-center gap-2 rounded-xl border border-blue-500/30 bg-blue-500/10 px-5 py-3 text-sm font-semibold text-blue-200 transition-colors hover:bg-blue-500/20"
          >
            <Code2 size={17} />
            Explore the source code
            <ArrowUpRight size={15} />
          </a>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <div className="rounded-2xl border border-white/10 bg-[#101827] p-6">
          <Scale size={21} className="mb-4 text-blue-400" />

          <h3 className="mb-2 font-semibold">
            The challenge
          </h3>

          <p className="text-sm leading-7 text-slate-400">
            Legal questions may depend on definitions,
            exceptions, deadlines, or multiple articles.
            A fluent answer is not necessarily a supported one.
          </p>
        </div>

        <div className="rounded-2xl border border-white/10 bg-[#101827] p-6">
          <BookOpen size={21} className="mb-4 text-cyan-400" />

          <h3 className="mb-2 font-semibold">
            The evidence
          </h3>

          <p className="text-sm leading-7 text-slate-400">
            The official regulation is parsed into
            113 articles and 324 overlapping retrieval
            chunks, with source references preserved
            throughout the pipeline.
          </p>
        </div>

        <div className="rounded-2xl border border-white/10 bg-[#101827] p-6">
          <ShieldCheck
            size={21}
            className="mb-4 text-emerald-400"
          />

          <h3 className="mb-2 font-semibold">
            The discipline
          </h3>

          <p className="text-sm leading-7 text-slate-400">
            Structured outputs, citation validation,
            abstention paths, held-out evaluation,
            and observable agent decisions make the
            system inspectable rather than opaque.
          </p>
        </div>
      </div>

      <div className="rounded-2xl border border-white/10 bg-[#101827] p-6 sm:p-8">
        <div className="mb-7 flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-widest text-blue-400">
              <GitBranch size={15} />
              Architecture comparison
            </div>

            <h3 className="text-xl font-bold">
              Four systems. One evaluation framework.
            </h3>

            <p className="mt-2 text-sm text-slate-400">
              From a retrieval baseline to an adaptive
              LangGraph research agent.
            </p>
          </div>

          <span className="rounded-full border border-white/10 px-3 py-1.5 text-xs text-slate-400">
            Same legal question dataset
          </span>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          {ARCHITECTURES.map((item) => {
            const Icon = item.icon;

            return (
              <div
                key={item.number}
                className="rounded-xl border border-white/10 bg-[#0b1220] p-5 transition-colors hover:border-blue-500/30"
              >
                <div className="mb-4 flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    <div className="rounded-lg border border-blue-500/20 bg-blue-500/10 p-2.5">
                      <Icon
                        size={19}
                        className="text-blue-300"
                      />
                    </div>

                    <div>
                      <h4 className="text-sm font-semibold">
                        {item.name}
                      </h4>
                      <p className="mt-1 text-xs text-slate-500">
                        {item.subtitle}
                      </p>
                    </div>
                  </div>

                  <span className="font-mono text-xs text-slate-600">
                    {item.number}
                  </span>
                </div>

                <p className="min-h-20 text-sm leading-6 text-slate-400">
                  {item.description}
                </p>

                <div className="mt-4 flex flex-wrap gap-2">
                  {item.tags.map((tag) => (
                    <span
                      key={tag}
                      className="rounded-md border border-white/10 px-2 py-1 text-[10px] text-slate-400"
                    >
                      {tag}
                    </span>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="rounded-2xl border border-white/10 bg-[#101827] p-6 sm:p-8">
        <div className="mb-6 flex items-center gap-2 text-xs font-semibold uppercase tracking-widest text-cyan-400">
          <Target size={16} />
          What the experiment revealed
        </div>

        <div className="grid gap-3 md:grid-cols-3">
          {RESULTS.map((item) => {
            const Icon = item.icon;

            return (
              <div
                key={item.label}
                className="rounded-xl border border-white/10 bg-[#0b1220] p-5"
              >
                <Icon
                  size={18}
                  className="mb-4 text-blue-400"
                />

                <div className="text-3xl font-bold tracking-tight text-white">
                  {item.value}
                </div>

                <div className="mt-2 text-xs font-semibold text-slate-200">
                  {item.label}
                </div>

                <div className="mt-1 text-xs text-slate-500">
                  {item.system}
                </div>
              </div>
            );
          })}
        </div>

        <div className="mt-6 rounded-xl border border-cyan-500/15 bg-cyan-500/[0.05] p-5">
          <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-cyan-200">
            <Sparkles size={17} />
            The key trade-off
          </div>

          <p className="text-sm leading-7 text-slate-300">
            On the 35 held-out questions where all four
            systems completed successfully, the LLM-driven
            agent achieved the highest citation F1
            (0.793), but its successful executions used
            approximately 4,549 tokens on average and
            took 2.92 seconds on average. The deterministic
            agent was faster and completed all 40 test
            executions successfully.
          </p>
        </div>

        <p className="mt-4 text-xs leading-6 text-slate-500">
          Citation F1 measures agreement with reference
          article citations, not full legal correctness.
          Citation scores use the 35-question paired subset;
          execution success uses all 40 held-out questions.
          Latency and token averages are measured on
          successful executions for each architecture.
        </p>
      </div>

      <div className="rounded-2xl border border-blue-500/20 bg-gradient-to-r from-blue-500/[0.10] to-cyan-500/[0.04] p-6 sm:p-8">
        <div className="flex flex-col justify-between gap-5 md:flex-row md:items-center">
          <div>
            <h3 className="text-xl font-bold">
              Explore the implementation.
            </h3>

            <p className="mt-2 max-w-xl text-sm leading-6 text-slate-400">
              Inspect the retrieval pipeline, LangGraph
              routing, evaluation scripts, and official
              legal source behind this live demonstration.
            </p>
          </div>

          <div className="flex flex-wrap gap-3">
            <a
              href={GITHUB}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-3 text-xs font-semibold text-white transition-colors hover:bg-blue-500"
            >
              <Code2 size={16} />
              GitHub repository
              <ArrowUpRight size={14} />
            </a>

            <a
              href={REGULATION}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 rounded-lg border border-white/15 bg-white/[0.04] px-4 py-3 text-xs font-semibold text-slate-200 transition-colors hover:bg-white/[0.08]"
            >
              <ExternalLink size={15} />
              Official EU AI Act
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}
