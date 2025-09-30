"use client";
import type { AgentSummary } from "../hooks/useAgents";

interface AgentDirectoryProps {
  agents: AgentSummary[];
  selected: string;
  onSelect?: (slug: string) => void;
  loading: boolean;
}

export function AgentDirectory({
  agents,
  selected,
  onSelect,
  loading,
}: AgentDirectoryProps) {
  if (loading || agents.length === 0) {
    return null;
  }

  return (
    <div style={{ display: "grid", gap: 8 }}>
      <h4 style={{ marginBottom: 0 }}>Available agents</h4>
      <div
        style={{
          display: "grid",
          gap: 8,
          gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
        }}
      >
        {agents.map((agent) => {
          const isSelected = agent.slug === selected;
          return (
            <button
              key={agent.slug}
              type="button"
              onClick={() => onSelect?.(agent.slug)}
              style={{
                textAlign: "left",
                border: isSelected ? "2px solid #2563eb" : "1px solid #cbd5f5",
                borderRadius: 8,
                padding: 12,
                background: isSelected ? "#eff6ff" : "#ffffff",
                cursor: onSelect ? "pointer" : "default",
              }}
            >
              <div style={{ fontWeight: 600 }}>{agent.title || agent.slug}</div>
              <div style={{ fontSize: 12, color: "#4b5563" }}>{agent.slug}</div>
              <p style={{ fontSize: 13, lineHeight: 1.4 }}>
                {agent.description}
              </p>
            </button>
          );
        })}
      </div>
    </div>
  );
}
