"use client";
import axios from "axios";
import { useMemo, useState } from "react";

import { useAgents } from "../hooks/useAgents";
import { extractErrorMessage } from "../lib/errors";
import { AgentDirectory } from "./AgentDirectory";
import { AgentPicker } from "./AgentPicker";

type ChatProps = {
  apiBase: string;
  token: string | null;
};

export default function Chat({ apiBase, token }: ChatProps) {
  const [question, setQuestion] = useState<string>("");
  const [llm, setLlm] = useState<string>(process.env.NEXT_PUBLIC_DEFAULT_LLM || "gpt");
  const [selectedAgent, setSelectedAgent] = useState<string>("");
  const [topK, setTopK] = useState<number>(16);
  const [answer, setAnswer] = useState<string | null>(null);
  const [sources, setSources] = useState<string[]>([]);
  const [dataProvider, setDataProvider] = useState<string>("");
  const [dataModel, setDataModel] = useState<string>("");
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const {
    agents,
    loading: loadingAgents,
    error: agentsError,
  } = useAgents(apiBase, token);

  const sourceEntries = useMemo(() => {
    const occurrences = new Map<string, number>();
    return sources.map((source, index) => {
      const count = occurrences.get(source) ?? 0;
      occurrences.set(source, count + 1);
      return {
        id: `${source}-${count}`,
        order: index + 1,
        label: source,
      };
    });
  }, [sources]);

  const submitQuestion = async () => {
    const trimmed = question.trim();
    if (!trimmed) {
      setError("Please enter a question before asking.");
      return;
    }
    if (!token) {
      setError("Sign in again to ask questions.");
      return;
    }
    if (loadingAgents) {
      return;
    }
    if (!agents.length) {
      setError("No agents available for this account.");
      return;
    }

    setIsSubmitting(true);
    setError(null);
    setAnswer(null);
    setSources([]);

    try {
      const response = await axios.post(
        `${apiBase}/chat/ask`,
        {
          provider: llm,
          question,
          agent: selectedAgent || null,
          top_k: topK,
        },
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        },
      );
      const data = response.data ?? {};
      setDataProvider(typeof data.provider === 'string' ? data.provider : '');
      setDataModel(typeof data.model === 'string' ? data.model : '');
      setAnswer(data.answer ?? "");
      setSources(Array.isArray(data.sources) ? data.sources : []);
      if (!selectedAgent && typeof data.agent === "string") {
        setSelectedAgent(data.agent);
      }
    } catch (error) {
      setError(extractErrorMessage(error, "Failed to retrieve answer"));
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div style={{ display: "grid", gap: 16 }}>
      <div style={{ display: "grid", gap: 8 }}>
        <label style={{ display: "flex", gap: 8 }}>
          <span style={{ alignSelf: "flex-start", paddingTop: 4 }}>LLM:</span>
          <select value={llm} onChange={(e)=>setLlm(e.target.value)} style={{ minWidth: 140 }}>
            <option value="gpt">GPT</option>
            <option value="deepseek">DeepSeek</option>
          </select>
        </label>
        <label style={{ display: "flex", gap: 8 }}>
          <span style={{ alignSelf: "flex-start", paddingTop: 4 }}>Question:</span>
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="Ask a question..."
            style={{
              flex: 1,
              minHeight: 96,
              resize: "vertical",
              fontFamily: "inherit",
              fontSize: "inherit",
              lineHeight: 1.4,
              padding: 8,
            }}
            disabled={!token}
          />
        </label>
        <div
          style={{
            display: "flex",
            gap: 8,
            flexWrap: "wrap",
            alignItems: "center",
          }}
        >
          <AgentPicker
            value={selectedAgent}
            onChange={setSelectedAgent}
            agents={agents}
            loading={loadingAgents}
            allowAutoRoute
          />
          <label>
            Top K:
            <input
              type="number"
              min={1}
              max={24}
              value={topK}
              onChange={(event) => setTopK(Number(event.target.value) || 1)}
              style={{ width: 80, marginLeft: 8 }}
              disabled={!token}
            />
          </label>
          <button
            type="button"
            onClick={submitQuestion}
            disabled={isSubmitting || loadingAgents || !token}
          >
            {isSubmitting ? "Asking..." : "Ask"}
          </button>
        </div>
      </div>
      <AgentDirectory
        agents={agents}
        selected={selectedAgent}
        onSelect={setSelectedAgent}
        loading={loadingAgents}
      />
      {agentsError && <p style={{ color: "#b91c1c" }}>{agentsError}</p>}
      {error && <p style={{ color: "#b91c1c" }}>{error}</p>}
      {answer !== null && (
        <div style={{ display: "grid", gap: 8 }}>
          <h3>Answer</h3>
          {(dataProvider || dataModel) && (
            <div style={{ color: "#4b5563" }}>
              Using provider: <strong>{dataProvider || 'n/a'}</strong>
              {" "}| model: <strong>{dataModel || 'n/a'}</strong>
            </div>
          )}
          <div style={{ whiteSpace: "pre-wrap" }}>{answer}</div>
          {sourceEntries.length > 0 && (
            <>
              <h4>Sources</h4>
              <ul>
                {sourceEntries.map((entry) => (
                  <li key={entry.id}>
                    [{entry.order}] {entry.label}
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
    </div>
  );
}
