import { ArrowUpRight, CheckCircle2, GitCompareArrows, ShieldCheck, TrendingUp } from "lucide-react";

const comparisons = [
  {
    title: "Naive RAG → Expanded RAG",
    improvement: "+0.1495",
    interval: "[+0.0602, +0.2538]",
    verdict: "Positive improvement in this benchmark",
    color: "text-emerald-400",
  },
  {
    title: "Expanded RAG → LLM-driven LangGraph",
    improvement: "+0.0303",
    interval: "[-0.0385, +0.1172]",
    verdict: "Incremental benefit remains uncertain",
    color: "text-amber-400",
  },
];

const architectures = [
  { name: "Naive RAG", f1: "0.6134", success: "95.0%", latency: "1.602s" },
  { name: "Expanded RAG", f1: "0.7629", success: "92.5%", latency: "1.720s" },
  { name: "Deterministic LangGraph", f1: "0.6320", success: "100%", latency: "1.532s" },
  { name: "LLM-driven LangGraph", f1: "0.7932", success: "97.5%", latency: "2.917s" },
];

export default function ResearchFindings() {
  return (
    <section id="research-findings" className="mt-16 space-y-8">
      <div className="text-center">
        <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-blue-500/30 bg-blue-500/10 px-4 py-2 text-xs font-semibold uppercase tracking-widest text-blue-300">
          <TrendingUp size={14} />
          Research Findings
        </div>

        <h2 className="text-3xl font-bold tracking-tight sm:text-4xl">
          What the benchmarks <span className="text-blue-400">actually revealed</span>
        </h2>

        <p className="mx-auto mt-4 max-w-3xl text-sm leading-7 text-slate-400">
          Four RAG architectures. Forty held-out questions. One hundred sixty attempted
          executions. The results challenge the assumption that more agentic
          complexity automatically produces better evidence-grounded answers.
        </p>
      </div>

      <div className="rounded-3xl border border-blue-500/30 bg-gradient-to-br from-blue-500/15 via-[#101827] to-[#101827] p-7 sm:p-9">
        <p className="text-xs font-semibold uppercase tracking-widest text-blue-300">
          Primary discovery
        </p>

        <h3 className="mt-4 max-w-3xl text-2xl font-bold leading-tight text-white sm:text-3xl">
          Better retrieval context mattered more than adding agent routing.
        </h3>

        <p className="mt-5 max-w-3xl text-sm leading-8 text-slate-300">
          Expanding retrieved article context improved mean citation F1 by
          <strong className="text-emerald-300"> +0.1495 </strong>
          over naive RAG on 31 answerable questions with successful executions
          across all four architectures.
        </p>

        <div className="mt-7 grid gap-4 sm:grid-cols-3">
          {[
            ["+0.1495", "Citation F1 improvement"],
            ["[+0.0602, +0.2538]", "95% bootstrap confidence interval"],
            ["12 / 2 / 17", "Question-level wins / losses / ties"],
          ].map(([value, label]) => (
            <div key={label} className="rounded-xl border border-white/10 bg-[#080d19]/70 p-5">
              <div className="break-words text-xl font-bold text-white">{value}</div>
              <div className="mt-2 text-xs leading-5 text-slate-400">{label}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {comparisons.map((item) => (
          <div key={item.title} className="rounded-2xl border border-white/10 bg-[#101827] p-6">
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <GitCompareArrows size={16} className="text-blue-400" />
              Paired architecture comparison
            </div>

            <h3 className="mt-4 text-lg font-semibold text-white">{item.title}</h3>

            <div className={`mt-5 text-4xl font-bold ${item.color}`}>
              {item.improvement}
            </div>

            <p className="mt-2 text-xs text-slate-400">
              Mean citation F1 difference · 95% CI {item.interval}
            </p>

            <p className="mt-5 border-t border-white/10 pt-4 text-sm text-slate-300">
              {item.verdict}
            </p>
          </div>
        ))}
      </div>

      <div className="rounded-2xl border border-white/10 bg-[#101827] p-6 sm:p-8">
        <h3 className="text-xl font-semibold text-white">
          Four architectures, different trade-offs
        </h3>

        <p className="mt-2 text-sm text-slate-400">
          LLM-driven routing achieved the highest citation F1, but deterministic
          LangGraph completed every execution.
        </p>

        <div className="mt-6 overflow-x-auto">
          <table className="w-full min-w-[540px] text-left text-sm">
            <thead>
              <tr className="border-b border-white/10 text-xs text-slate-400">
                <th className="py-3 pr-4 font-medium">Architecture</th>
                <th className="px-3 py-3 font-medium">Citation F1</th>
                <th className="px-3 py-3 font-medium">Success</th>
                <th className="pl-3 py-3 font-medium">Latency</th>
              </tr>
            </thead>
            <tbody>
              {architectures.map((item) => (
                <tr key={item.name} className="border-b border-white/5 last:border-0">
                  <td className="py-4 pr-4 font-medium text-slate-200">{item.name}</td>
                  <td className="px-3 py-4 text-blue-300">{item.f1}</td>
                  <td className="px-3 py-4 text-slate-300">{item.success}</td>
                  <td className="pl-3 py-4 text-slate-300">{item.latency}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        <div className="rounded-2xl border border-white/10 bg-[#101827] p-6">
          <ShieldCheck className="text-emerald-400" size={22} />
          <div className="mt-5 text-3xl font-bold text-white">40/40</div>
          <h3 className="mt-2 font-semibold text-white">Most reliable execution</h3>
          <p className="mt-3 text-sm leading-7 text-slate-400">
            Deterministic LangGraph completed all 40 benchmark questions successfully.
          </p>
        </div>

        <div className="rounded-2xl border border-white/10 bg-[#101827] p-6">
          <TrendingUp className="text-blue-400" size={22} />
          <div className="mt-5 text-3xl font-bold text-white">26/39</div>
          <h3 className="mt-2 font-semibold text-white">Agent chose expansion</h3>
          <p className="mt-3 text-sm leading-7 text-slate-400">
            The LLM-driven agent selected context expansion in 66.7% of
            successful executions.
          </p>
        </div>

        <div className="rounded-2xl border border-white/10 bg-[#101827] p-6">
          <CheckCircle2 className="text-amber-400" size={22} />
          <div className="mt-5 text-3xl font-bold text-white">6/160</div>
          <h3 className="mt-2 font-semibold text-white">Validation failures</h3>
          <p className="mt-3 text-sm leading-7 text-slate-400">
            Five unsupported-citation errors and one invalid-abstention error.
            No recorded timeouts or rate-limit failures.
          </p>
        </div>
      </div>

      <div className="rounded-2xl border border-amber-500/20 bg-amber-500/[0.05] p-6">
        <h3 className="text-lg font-semibold text-white">
          The engineering takeaway
        </h3>

        <p className="mt-3 text-sm leading-8 text-slate-300">
          LLM-driven LangGraph improved citation F1 over expanded RAG by only
          <strong> +0.0303</strong>, with a 95% confidence interval that includes zero.
          It also used approximately <strong>96% more tokens</strong> and had
          approximately <strong>70% higher latency</strong>.
          In this benchmark, its extra complexity did not establish a statistically
          reliable citation-quality advantage over simpler context expansion.
        </p>

        <p className="mt-4 text-xs leading-6 text-slate-500">
          Methodology: 40 held-out questions, 160 attempted executions, 35 complete
          paired questions, and 31 answerable complete pairs for citation F1.
          Confidence intervals use 20,000 paired bootstrap resamples.
          Citation F1 measures article references, not semantic answer correctness.
        </p>

        <a
          href="https://github.com/revanthpuvanes/eu-ai-act-agentic-rag#-four-rag-architectures-one-benchmark-a-surprising-trade-off"
          target="_blank"
          rel="noopener noreferrer"
          className="mt-5 inline-flex items-center gap-2 text-sm font-semibold text-blue-300 hover:text-blue-200"
        >
          Read the full benchmark methodology
          <ArrowUpRight size={16} />
        </a>
      </div>
    </section>
  );
}
