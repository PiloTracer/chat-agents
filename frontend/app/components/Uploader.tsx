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
  const [files, setFiles] = useState<File[]>([]);
  const [agent, setAgent] = useState<string>("");
  const [status, setStatus] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [fileInputKey, setFileInputKey] = useState<number>(0);

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
    if (!files.length) {
      setError("Pick at least one document to upload first.");
      return;
    }
    if (!agent) {
      setError("Select an agent to receive the documents.");
      return;
    }

    setIsUploading(true);
    setError(null);
    setStatus("");

    const filesToUpload = files;

    try {
      const formData = new FormData();
      formData.append("agent_slug", agent);
      filesToUpload.forEach((current) => {
        formData.append("files", current);
      });

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
      const documents = Array.isArray(data.documents) ? data.documents : [];
      const uploadedCount = documents.length || filesToUpload.length;
      const chunkCount =
        typeof data.total_chunks === "number" ? data.total_chunks : undefined;
      const uploadedNames = documents
        .map((entry: any) => entry?.filename)
        .filter((name: unknown): name is string => typeof name === "string");
      const fallbackNames = filesToUpload.map((file) => file.name);
      const names = uploadedNames.length ? uploadedNames : fallbackNames;
      const chunkSuffix = chunkCount ? ` (${chunkCount} chunks)` : "";
      setStatus(
        `Uploaded ${uploadedCount} file${uploadedCount === 1 ? "" : "s"}${chunkSuffix}. Files: ${names.join(", ")}`,
      );
      setFiles([]);
      setFileInputKey((value) => value + 1);
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
          key={fileInputKey}
          type="file"
          onChange={(event) =>
            setFiles(event.target.files ? Array.from(event.target.files) : [])
          }
          multiple
          disabled={!token}
        />
        <button
          type="button"
          onClick={upload}
          disabled={isUploading || loadingAgents || !agents.length || !token}
        >
          {isUploading ? "Uploading..." : "Upload"}
        </button>
        {files.length > 0 && (
          <div
            style={{
              flexBasis: "100%",
              fontSize: 13,
              color: "#374151",
            }}
          >
            <strong>Selected files:</strong>
            <ul style={{ margin: "4px 0 0", paddingLeft: 20 }}>
              {files.map((current) => (
                <li key={`${current.name}-${current.lastModified}`}>
                  {current.name}
                </li>
              ))}
            </ul>
          </div>
        )}
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
