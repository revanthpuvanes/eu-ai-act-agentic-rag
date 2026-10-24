"use client";

import { useEffect, useState } from "react";
import {
  Activity,
  ArrowDown,
  BookOpen,
  BrainCircuit,
  Check,
  CircleDot,
  FileText,
  GitBranch,
  Search,
  ShieldAlert,
  Sparkles,
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

type Props = {
  steps: Step[];
  loading: boolean;
};

const NODES = [
  {
    id: "retrieve",
    title: "Retrieve",
    subtitle: "Hybrid search",
    icon: Search,
  },
  {
    id: "assess",
    title: "Assess",
    subtitle: "Evidence decision",
    icon: BrainCircuit,
  },
  {
    id: "answer_directly",
    title: "Answer",
    subtitle: "Use current evidence",
    icon: FileText,
  },
  {
    id: "expand",
    title: "Expand",
    subtitle: "Full article context",
    icon: BookOpen,
  },
  {
    id: "search",
    title: "Search",
    subtitle: "Targeted retrieval",
    icon: Search,
  },
  {
    id: "abstain",
    title: "Abstain",
    subtitle: "No generation",
    icon: ShieldAlert,
  },
  {
    id: "generate",
    title: "Generate",
    subtitle: "Grounded response",
    icon: Sparkles,
  },
] as const;

type NodeId = (typeof NODES)[number]["id"];

function NodeCard({
  id,
  completed,
  selected,
  onClick,
}: {
  id: NodeId;
  completed: boolean;
  selected: boolean;
  onClick: () => void;
}) {
  const node = NODES.find((item) => item.id === id)!;
  const Icon = node.icon;

  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={selected}
      className={`relative flex min-h-21 w-full flex-col items-center justify-center gap-1 rounded-xl border px-2 py-3 text-center transition-all ${
        completed
          ? "border-blue-400/70 bg-blue-500/15 shadow-[0_0_24px_rgba(59,130,246,0.12)]"
          : "border-white/10 bg-[#0b1220]"
      } ${
        selected ? "ring-2 ring-cyan-400/60" : ""
      }`}
    >
      {completed && (
        <span className="absolute right-2 top-2 text-emerald-400">
          <Check size={13} />
        </span>
      )}

      <Icon
        size={19}
        className={
          completed ? "text-blue-300" : "text-slate-500"
        }
      />

      <span className="text-xs font-semibold">
        {node.title}
      </span>

      <span className="text-[10px] leading-4 text-slate-500">
        {node.subtitle}
      </span>
    </button>
  );
}

export default function LiveAgentGraph({
  steps,
  loading,
}: Props) {
  const [selectedNode, setSelectedNode] =
    useState<NodeId | null>(null);

  const completed = new Set(steps.map((step) => step.node));

  const latestNode = steps.at(-1)?.node;

  const selectedStep = [...steps]
    .reverse()
    .find((step) => step.node === selectedNode);

  useEffect(() => {
    if (latestNode && NODES.some((node) => node.id === latestNode)) {
      setSelectedNode(latestNode as NodeId);
    }
  }, [latestNode]);

  function renderNode(id: NodeId) {
    return (
      <NodeCard
        id={id}
        completed={completed.has(id)}
        selected={selectedNode === id}
        onClick={() => setSelectedNode(id)}
      />
    );
  }

  return (
    <section className="rounded-2xl border border-white/10 bg-[#101827] p-6">
      <div className="mb-5 flex items-start justify-between gap-3">
        <div>
          <h3 className="flex items-center gap-2 font-semibold">
            <GitBranch size={18} className="text-blue-400" />
            Live agent architecture
          </h3>
          <p className="mt-2 text-xs leading-5 text-slate-500">
            Actual LangGraph nodes. Blue nodes have completed
            during this investigation.
          </p>
        </div>

        <span className="shrink-0 rounded-full border border-white/10 px-2 py-1 text-[10px] text-slate-400">
          {loading ? "Executing" : steps.length ? "Finished" : "Idle"}
        </span>
      </div>

      <div className="mx-auto max-w-sm">
        <div className="mx-auto w-44">
          {renderNode("retrieve")}
        </div>

        <div className="flex justify-center py-2">
          <ArrowDown size={17} className="text-slate-600" />
        </div>

        <div className="mx-auto w-44">
          {renderNode("assess")}
        </div>

        <div className="flex justify-center py-2">
          <GitBranch size={18} className="text-slate-500" />
        </div>

        <div className="grid grid-cols-2 gap-3">
          {renderNode("answer_directly")}
          {renderNode("expand")}
          {renderNode("search")}
          {renderNode("abstain")}
        </div>

        <div className="flex justify-center py-2">
          <ArrowDown size={17} className="text-slate-600" />
        </div>

        <div className="mx-auto w-44">
          {renderNode("generate")}
        </div>

        <p className="mt-3 text-center text-[10px] text-slate-500">
          Answer / Expand / Search → Generate
          <br />
          Abstain → End
        </p>
      </div>

      <div className="mt-6 border-t border-white/10 pt-5">
        <div className="mb-3 flex items-center gap-2 text-xs font-semibold text-slate-300">
          <CircleDot size={14} className="text-cyan-400" />
          Node inspector
        </div>

        {!selectedNode && (
          <p className="text-xs leading-6 text-slate-500">
            Click a graph node to inspect its execution.
          </p>
        )}

        {selectedNode && (
          <div className="rounded-xl border border-white/10 bg-[#0b1220] p-4">
            <div className="flex items-center justify-between gap-3">
              <span className="font-mono text-xs text-blue-300">
                {selectedNode}
              </span>

              <span className="text-[10px] text-slate-400">
                {selectedStep ? "Executed" : "Not executed"}
              </span>
            </div>

            {selectedStep ? (
              <div className="mt-3 space-y-2">
                {typeof selectedStep.elapsed_seconds === "number" && (
                  <p className="text-xs text-slate-400">
                    Duration: {selectedStep.elapsed_seconds.toFixed(2)}s
                  </p>
                )}

                {selectedStep.assessment?.action && (
                  <p className="text-xs text-cyan-300">
                    Decision: {selectedStep.assessment.action}
                  </p>
                )}

                {selectedStep.assessment?.reason && (
                  <p className="text-xs leading-6 text-slate-300">
                    {selectedStep.assessment.reason}
                  </p>
                )}

                {selectedStep.action_taken && (
                  <p className="text-xs text-blue-300">
                    Action: {selectedStep.action_taken}
                  </p>
                )}

                {!selectedStep.assessment &&
                  !selectedStep.action_taken &&
                  typeof selectedStep.elapsed_seconds !== "number" && (
                    <p className="text-xs text-slate-400">
                      Node completed. No additional metadata
                      was provided by the streaming endpoint.
                    </p>
                  )}
              </div>
            ) : (
              <p className="mt-3 text-xs leading-6 text-slate-500">
                This node has not executed in the current investigation.
              </p>
            )}
          </div>
        )}

        <div className="mt-4 flex items-center gap-2 text-[10px] text-slate-500">
          <Activity size={12} />
          {steps.length} completed node events received
        </div>
      </div>
    </section>
  );
}
