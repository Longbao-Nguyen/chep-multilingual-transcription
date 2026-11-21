import { useEffect, useState } from 'react';
import { apiRequest, authToken, getUiSessionSocketUrl } from '../services/api';
import type { ExitChoice } from '../types';

export function useUiSession(departure: ExitChoice | null): boolean {
  const [sessionConnected, setSessionConnected] = useState(false);

  useEffect(() => {
    if (departure) return;
    let disposed = false;
    let socket: WebSocket | undefined;
    let timer: ReturnType<typeof setTimeout>;
    const connect = () => {
      socket = new WebSocket(getUiSessionSocketUrl());
      socket.onopen = () => socket?.send(JSON.stringify({ token: authToken }));
      socket.onmessage = () => {
        if (!disposed) setSessionConnected(true);
      };
      socket.onclose = () => {
        if (!disposed) {
          setSessionConnected(false);
          timer = setTimeout(connect, 1500);
        }
      };
      socket.onerror = () => socket?.close();
    };
    connect();
    return () => {
      disposed = true;
      clearTimeout(timer);
      socket?.close();
    };
  }, [departure]);

  return sessionConnected;
}

export function useServerStopped(departure: ExitChoice | null): boolean {
  const [serverStopped, setServerStopped] = useState(false);

  useEffect(() => {
    if (departure !== 'quit') return;
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    const check = async () => {
      try {
        await apiRequest('/status', { signal: AbortSignal.timeout(5000) });
      } catch (e) {
        if (!disposed && e instanceof TypeError) {
          setServerStopped(true);
          return;
        }
      }
      if (!disposed) {
        timer = setTimeout(check, 800);
      }
    };
    void check();
    return () => {
      disposed = true;
      clearTimeout(timer);
    };
  }, [departure]);

  return serverStopped;
}
