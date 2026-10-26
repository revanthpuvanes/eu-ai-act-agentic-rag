"use client";

import { useState } from "react";
import {
  Activity,
  BarChart3,
  Clock3,
  Cpu,
  FlaskConical,
  ShieldCheck,
} from "lucide-react";

type Strategy = {
  name: string;
  shortName: string;
  description: string;
  success: number;
  citationF1: number;
  latency: number;
  tokens: number;
  color: string;
};

const STRATEGIES: Strategy[] = [
  {
    name: "Naive RAG",
    shortName: "Naive",
    description: "Hybrid retrieval followed by direct generation.",
    success: 95.0,
    citationF1: 0.6134,
    latency: 1.602,
    tokens: 1838.2,
    color: "bg-slate-400",
  },
  {
    name: "Expanded RAG",
    shortName: "Expanded",
    description: "Hybrid retrieval with expanded article context.",
    success: 92.5,
    citationF1: 0.7629,
    latency: 1.720,
    tokens: 2325.9,
    color: "bg-cyan-400",
  },
  {
    name: "Deterministic LangGraph",
    shortName: "Deterministic",
    description: "Rule-based evidence assessment and graph routing.",
    success: 100.0,
    citationF1: 0.6320,
    latency: 1.532,
    tokens: 1897.3,
    color: "bg-violet-400",
  },
  {
    name: "LLM-Driven LangGraph",
    shortName: "LLM Agent",
    description: "Model-driven evidence assessment and tool decisions.",
    success: 97.5,
    citationF1: 0.7932,
    latency: 2.917,
    tokens: 4548.9,
    color: "bg-blue-400",
  },
];

type Metric = "citationF1" | "success" | "latency" | "tokens";

const METRICS: {
  key: Metric;
  label: string;
  unit: string;
  better: "higher" | "lower";
}[] = [
  {
    key: "citationF1",
    label: "Citation F1",
    unit: "",
    better: "higher",
  },
  {
    key: "success",
    label: "Execution success",
    unit: "%",
    better: "higher",
  },
  {
    key: "latency",
    label: "Mean latency",
    unit: "s",
    better: "lower",
  },
  {
    key: "tokens",
    label: "Mean tokens",
    unit: "",
    better: "lower",
  },
];

function formatMetric(metric: Metric, value: number) {
  if (metric === "citationF1") return value.toFixed(4);
  if (metric === "success") return `${value.toFixed(1)}%`;
  if (metric === "latency") return `${value.toFixed(2)}s`;
  return Math.round(value).toLocaleString();
}

export default function BenchmarkDashboard() {
  const [metric, setMetric] = useState<Metric>("citationF1");

  const currentMetric = METRICS.find(
    (item) => item.key === metric
  )!;

  const maximum = Math.max(
    ...STRATEGIES.map((strategy) => strategy[metric])
  );

  const bestValue =
    currentMetric.better === "higher"
      ? maximum
      : Math.min(
          ...STRATEGIES.map((strategy) => strategy[metric])
        );

  return (
    <section className="mt-8 overflow-hidden rounded-2xl border border-white/10 bg-[#101827]">
      <div className="flex w-full items-center justify-between gap-4 p-6">
        <div>
          <h2 className="text-xl font-bold">
            Four-strategy benchmark
          </h2>
          <p className="mt-2 text-sm text-slate-400">
            Real held-out results · 40 questions · 160 executions
          </p>
        </div>

      </div>

      <div className="space-y-8 border-t border-white/10 p-6">
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            {[
              {
                icon: FlaskConical,
                label: "Held-out questions",
                value: "40",
              },
              {
                icon: Activity,
                label: "Executions",
                value: "160",
              },
              {
                icon: ShieldCheck,
                label: "Complete pairs",
                value: "35",
              },
              {
                icon: Cpu,
                label: "Architectures",
                value: "4",
              },
            ].map((item) => {
              const Icon = item.icon;

              return (
                <div
                  key={item.label}
                  className="rounded-xl border border-white/5 bg-[#0b1220] p-4"
                >
                  <Icon size={17} className="mb-3 text-blue-400" />
                  <div className="text-2xl font-bold">
                    {item.value}
                  </div>
                  <div className="mt-1 text-xs text-slate-500">
                    {item.label}
                  </div>
                </div>
              );
            })}
          </div>

          <div>
            <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
              <h3 className="flex items-center gap-2 font-semibold">
                <BarChart3 size={18} className="text-blue-400" />
                Architecture comparison
              </h3>

              <div className="flex flex-wrap gap-2">
                {METRICS.map((item) => (
                  <button
                    key={item.key}
                    type="button"
                    onClick={() => setMetric(item.key)}
                    className={`rounded-lg px-3 py-2 text-xs transition ${
                      metric === item.key
                        ? "bg-blue-600 text-white"
                        : "border border-white/10 text-slate-400 hover:text-white"
                    }`}
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="space-y-5 rounded-xl border border-white/5 bg-[#0b1220] p-5">
              {STRATEGIES.map((strategy) => {
                const value = strategy[metric];
                const isBest = value === bestValue;

                return (
                  <div key={strategy.name}>
                    <div className="mb-2 flex items-center justify-between gap-3">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="text-sm font-medium">
                          {strategy.name}
                        </span>

                        {isBest && (
                          <span className="rounded bg-emerald-500/10 px-2 py-0.5 text-[10px] text-emerald-300">
                            Best on metric
                          </span>
                        )}
                      </div>

                      <span className="font-mono text-sm font-semibold tabular-nums">
                        {formatMetric(metric, value)}
                      </span>
                    </div>

                    <div className="h-3 overflow-hidden rounded-full bg-white/5">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${strategy.color}`}
                        style={{
                          width: `${Math.max(
                            2,
                            (value / maximum) * 100
                          )}%`,
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            <p className="mt-3 text-xs leading-5 text-slate-500">
              {currentMetric.better === "higher"
                ? "Higher values are better for this metric."
                : "Lower values are better for this metric."}
              {" "}
              Bar lengths show measured values, not an overall
              quality ranking.
            </p>
          </div>

          <div className="grid gap-3 md:grid-cols-2">
            {STRATEGIES.map((strategy) => (
              <div
                key={strategy.name}
                className="rounded-xl border border-white/10 bg-[#0b1220] p-5"
              >
                <div className="mb-3 flex items-center gap-2">
                  <span
                    className={`h-2.5 w-2.5 rounded-full ${strategy.color}`}
                  />
                  <h4 className="font-semibold">
                    {strategy.name}
                  </h4>
                </div>

                <p className="mb-4 min-h-10 text-xs leading-5 text-slate-400">
                  {strategy.description}
                </p>

                <div className="grid grid-cols-2 gap-3 text-xs">
                  <div>
                    <div className="text-slate-500">
                      Citation F1
                    </div>
                    <div className="mt-1 font-mono text-sm font-semibold">
                      {strategy.citationF1.toFixed(4)}
                    </div>
                  </div>

                  <div>
                    <div className="text-slate-500">
                      Execution success
                    </div>
                    <div className="mt-1 font-mono text-sm font-semibold">
                      {strategy.success.toFixed(1)}%
                    </div>
                  </div>

                  <div>
                    <div className="flex items-center gap-1 text-slate-500">
                      <Clock3 size={12} />
                      Mean latency
                    </div>
                    <div className="mt-1 font-mono text-sm font-semibold">
                      {strategy.latency.toFixed(2)}s
                    </div>
                  </div>

                  <div>
                    <div className="text-slate-500">
                      Mean tokens
                    </div>
                    <div className="mt-1 font-mono text-sm font-semibold">
                      {Math.round(
                        strategy.tokens
                      ).toLocaleString()}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-5">
            <h4 className="mb-2 text-sm font-semibold text-amber-200">
              How to interpret these results
            </h4>

            <p className="text-xs leading-6 text-slate-400">
              Citation F1 is calculated on 31 answerable questions within the 35 questions with
              complete paired executions across all four
              strategies. Execution success is calculated over
              40 attempted questions per strategy. Latency and
              token averages describe successful executions.
              Citation F1 measures citation matching against
              annotated references, not whether the generated
              legal explanation is correct.
            </p>

            <p className="mt-3 text-xs leading-6 text-slate-400">
              The LLM-driven agent achieved the highest citation
              F1 in this evaluation, but used substantially more
              tokens and had higher latency. The deterministic
              graph had the highest execution success rate.
              These are observed results on this dataset, not
              universal performance guarantees.
            </p>
          </div>
      </div>
    </section>
  );
}
