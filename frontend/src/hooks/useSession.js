import { useCallback, useEffect, useState } from 'react';
import api from '../api';
import { clearSession, getRefreshToken, getStoredUser, saveSession } from '../lib/storage';

export function useSession() {
  const [user, setUser] = useState(getStoredUser);

  useEffect(() => {
    const handler = () => setUser(null);
    window.addEventListener('diglit-session-expired', handler);
    return () => window.removeEventListener('diglit-session-expired', handler);
  }, []);

  const login = useCallback(async (username, password) => {
    const { data } = await api.post('/auth/login', { username, password });
    saveSession(data);
    setUser(data.user);
    return data.user;
  }, []);

  const register = useCallback(async (payload) => {
    const { data } = await api.post('/auth/register', payload);
    saveSession(data);
    setUser(data.user);
    return data.user;
  }, []);

  const logout = useCallback(async () => {
    const refresh = getRefreshToken();
    try {
      if (refresh) await api.post('/auth/logout', { refresh_token: refresh });
    } catch {
      // Local logout must remain available if backend is temporarily unreachable.
    }
    clearSession();
    setUser(null);
  }, []);

  return { user, login, register, logout };
}
