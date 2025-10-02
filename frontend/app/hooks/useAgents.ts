"use client";
import axios from "axios";
import { useEffect, useState } from "react";

import { extractErrorMessage } from "../lib/errors";

export interface AgentSummary {
  id: number;
  slug: string;
  title: string;
  description: string;
}

interface UseAgentsResult {
  agents: AgentSummary[];
  loading: boolean;
  error: string | null;
  refresh: () => void;
}

export function useAgents(
  apiBase: string,
  token: string | null,
): UseAgentsResult {
  const [agents, setAgents] = useState<AgentSummary[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshToken, setRefreshToken] = useState<number>(0);

  useEffect(() => {
    if (!token) {
      setAgents([]);
      setLoading(false);
      setError(null);
      return;
    }

    let cancelled = false;
    const tokenSnapshot = refreshToken;

    async function fetchAgents() {
      setLoading(true);
      setError(null);
      try {
        const response = await axios.get<AgentSummary[]>(`${apiBase}/agents`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        if (!cancelled && tokenSnapshot === refreshToken) {
          setAgents(response.data ?? []);
        }
      } catch (error) {
        if (!cancelled && tokenSnapshot === refreshToken) {
          setAgents([]);
          setError(extractErrorMessage(error, "Failed to load agents"));
        }
      } finally {
        if (!cancelled && tokenSnapshot === refreshToken) {
          setLoading(false);
        }
      }
    }

    fetchAgents();

    return () => {
      cancelled = true;
    };
  }, [apiBase, token, refreshToken]);

  return {
    agents,
    loading,
    error,
    refresh: () => setRefreshToken((value) => value + 1),
  };
}
