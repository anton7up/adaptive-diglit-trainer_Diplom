import axios from 'axios';
import { clearSession, getAccessToken, getRefreshToken, saveSession } from './lib/storage';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000',
  timeout: 15000,
});

let refreshPromise = null;

api.interceptors.request.use((config) => {
  const token = getAccessToken();
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const original = error.config;
    if (!error.response || error.response.status !== 401 || original?._retry || original?.url?.includes('/auth/')) {
      throw error;
    }
    const refreshToken = getRefreshToken();
    if (!refreshToken) {
      clearSession();
      window.dispatchEvent(new Event('diglit-session-expired'));
      throw error;
    }
    original._retry = true;
    if (!refreshPromise) {
      refreshPromise = axios.post(`${api.defaults.baseURL}/auth/refresh`, { refresh_token: refreshToken })
        .then(({ data }) => {
          saveSession(data);
          return data.access_token;
        })
        .finally(() => { refreshPromise = null; });
    }
    try {
      const token = await refreshPromise;
      original.headers.Authorization = `Bearer ${token}`;
      return api(original);
    } catch (refreshError) {
      clearSession();
      window.dispatchEvent(new Event('diglit-session-expired'));
      throw refreshError;
    }
  },
);

export default api;
