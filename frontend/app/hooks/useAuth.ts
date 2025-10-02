"use client";
import axios from "axios";
import { useCallback, useEffect, useState } from "react";

import { extractErrorMessage } from "../lib/errors";

const STORAGE_KEY = "multiagent-rag:auth-token";

export type AuthUser = {
  username: string;
  role: string;
};

type MeResponse = {
  user?: AuthUser;
  agents?: string[];
};

type LoginResponse = {
  access_token?: string;
  token_type?: string;
  user?: AuthUser;
};

type UseAuthResult = {
  ready: boolean;
  token: string | null;
  user: AuthUser | null;
  agents: string[];
  login: (username: string, password: string) => Promise<boolean>;
  loginPending: boolean;
  loginError: string | null;
  logout: () => void;
  refresh: () => Promise<void>;
};

export function useAuth(apiBase: string): UseAuthResult {
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<AuthUser | null>(null);
  const [agents, setAgents] = useState<string[]>([]);
  const [ready, setReady] = useState<boolean>(false);
  const [loginPending, setLoginPending] = useState<boolean>(false);
  const [loginError, setLoginError] = useState<string | null>(null);

  const clear = useCallback(() => {
    setToken(null);
    setUser(null);
    setAgents([]);
    if (typeof window !== "undefined") {
      window.localStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  const fetchMe = useCallback(
    async (tokenValue: string) => {
      const response = await axios.get<MeResponse>(`${apiBase}/auth/me`, {
        headers: { Authorization: `Bearer ${tokenValue}` },
      });
      const data = response.data ?? {};
      setToken(tokenValue);
      setUser(data.user ?? null);
      setAgents(Array.isArray(data.agents) ? data.agents : []);
      if (typeof window !== "undefined") {
        window.localStorage.setItem(STORAGE_KEY, tokenValue);
      }
    },
    [apiBase],
  );

  useEffect(() => {
    if (typeof window === "undefined") {
      setReady(true);
      return;
    }
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (!stored) {
      setReady(true);
      return;
    }
    fetchMe(stored)
      .catch(() => {
        clear();
      })
      .finally(() => {
        setReady(true);
      });
  }, [clear, fetchMe]);

  const login = useCallback(
    async (username: string, password: string) => {
      setLoginPending(true);
      setLoginError(null);
      try {
        const response = await axios.post<LoginResponse>(
          `${apiBase}/auth/login`,
          { username, password },
        );
        const tokenValue = response.data?.access_token;
        if (!tokenValue) {
          throw new Error("Missing access token");
        }
        await fetchMe(tokenValue);
        return true;
      } catch (error) {
        setLoginError(extractErrorMessage(error, "Login failed"));
        clear();
        return false;
      } finally {
        setLoginPending(false);
        setReady(true);
      }
    },
    [apiBase, clear, fetchMe],
  );

  const logout = useCallback(() => {
    clear();
    setReady(true);
    axios.post(`${apiBase}/auth/logout`).catch(() => {
      /* ignore */
    });
  }, [apiBase, clear]);

  const refresh = useCallback(async () => {
    if (!token) {
      return;
    }
    await fetchMe(token);
  }, [fetchMe, token]);

  return {
    ready,
    token,
    user,
    agents,
    login,
    loginPending,
    loginError,
    logout,
    refresh,
  };
}

