"use client";
import axios from "axios";
import { useEffect, useState } from "react";

import { extractErrorMessage } from "../lib/errors";

export interface PrincipalSummary {
  username: string;
  role: string;
  displayName: string;
  agentSlugs: string[];
}

interface UseDemoPrincipalsResult {
  principals: PrincipalSummary[];
  loading: boolean;
  error: string | null;
}

export function useDemoPrincipals(apiBase: string): UseDemoPrincipalsResult {
  const [principals, setPrincipals] = useState<PrincipalSummary[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function fetchPrincipals() {
      setLoading(true);
      setError(null);
      try {
        const response = await axios.get<PrincipalSummary[]>(
          `${apiBase}/auth/demo-users`,
        );
        if (!cancelled) {
          const data = response.data ?? [];
          setPrincipals(
            data.map((principal) => ({
              username: principal.username,
              role: principal.role,
              displayName:
                principal.display_name ??
                principal.displayName ??
                principal.username,
              agentSlugs: Array.isArray(principal.agent_slugs)
                ? principal.agent_slugs
                : Array.isArray(principal.agentSlugs)
                  ? principal.agentSlugs
                  : [],
            })),
          );
        }
      } catch (error) {
        if (!cancelled) {
          setPrincipals([]);
          setError(extractErrorMessage(error, "Failed to load demo users"));
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    fetchPrincipals();

    return () => {
      cancelled = true;
    };
  }, [apiBase]);

  return { principals, loading, error };
}
