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
  role: string,
  user: string,
): UseAgentsResult {
  const [agents, setAgents] = useState<AgentSummary[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [refreshToken, setRefreshToken] = useState<number>(0);

  useEffect(() => {
    let cancelled = false;
    const token = refreshToken;

    async function fetchAgents() {
      setLoading(true);
      setError(null);
      try {
        const response = await axios.get<AgentSummary[]>(`${apiBase}/agents`, {
          headers: {
            "X-Role": role,
            "X-User": user,
          },
        });
        if (!cancelled && token === refreshToken) {
          setAgents(response.data ?? []);
        }
      } catch (error) {
        if (!cancelled && token === refreshToken) {
          setAgents([]);
          setError(extractErrorMessage(error, "Failed to load agents"));
        }
      } finally {
        if (!cancelled && token === refreshToken) {
          setLoading(false);
        }
      }
    }

    fetchAgents();

    return () => {
      cancelled = true;
    };
  }, [apiBase, role, user, refreshToken]);

  return {
    agents,
    loading,
    error,
    refresh: () => setRefreshToken((token) => token + 1),
  };
}
