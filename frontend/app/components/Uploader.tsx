"use client";
import axios from "axios";
import { useEffect, useState } from "react";

import { useAgents } from "../hooks/useAgents";
import { extractErrorMessage } from "../lib/errors";
import { AgentDirectory } from "./AgentDirectory";
import { AgentPicker } from "./AgentPicker";

type UploaderProps = {
  apiBase: string;
  token: string | null;
};

export default function Uploader({ apiBase, token }: UploaderProps) {
  const [file, setFile] = useState<File | null>(null);
  const [agent, setAgent] = useState<string>("");
  const [status, setStatus] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);

  const {
    agents,
    loading: loadingAgents,
    error: agentsError,
    refresh,
  } = useAgents(apiBase, token);

  useEffect(() => {
    if (!agent && agents.length > 0) {
      setAgent(agents[0].slug);
    }
  }, [agent, agents]);

  const upload = async () => {
    if (!token) {
      setError("Sign in again to upload documents.");
      return;
    }
    if (!file) {
      setError("Pick a document to upload first.");
      return;
    }
    if (!agent) {
      setError("Select an agent to receive the document.");
      return;
    }

    setIsUploading(true);
    setError(null);
    setStatus("");

    try {
      const formData = new FormData();
      formData.append("agent_slug", agent);
      formData.append("file", file);

      const response = await axios.post(
        `${apiBase}/documents/upload`,
        formData,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        },
      );

      const data = response.data ?? {};
      const chunkCount =
        typeof data.chunks === "number" ? data.chunks : undefined;
      setStatus(
        `Uploaded successfully${chunkCount ? ` (${chunkCount} chunks)` : ""}.`,
      );
      setFile(null);
      refresh();
    } catch (error) {
      setError(extractErrorMessage(error, "Upload failed"));
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div style={{ display: "grid", gap: 16 }}>
      <div
        style={{
          display: "flex",
          gap: 8,
          alignItems: "center",
          flexWrap: "wrap",
        }}
      >
        <AgentPicker
          value={agent}
          onChange={setAgent}
          agents={agents}
          loading={loadingAgents}
          allowAutoRoute={false}
        />
        <input
          type="file"
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          multiple={false}
          disabled={!token}
        />
        <button
          type="button"
          onClick={upload}
          disabled={isUploading || loadingAgents || !agents.length || !token}
        >
          {isUploading ? "Uploading..." : "Upload"}
        </button>
      </div>
      <AgentDirectory
        agents={agents}
        selected={agent}
        onSelect={setAgent}
        loading={loadingAgents}
      />
      {agentsError && <p style={{ color: "#b91c1c" }}>{agentsError}</p>}
      {error && <p style={{ color: "#b91c1c" }}>{error}</p>}
      {status && <p style={{ color: "#047857" }}>{status}</p>}
    </div>
  );
}
