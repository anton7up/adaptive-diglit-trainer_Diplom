import { useState } from 'react';

export default function AuthPage({ onLogin, onRegister }) {
  const [mode, setMode] = useState('login');
  const [form, setForm] = useState({ username: 'student_demo', password: '1234', full_name: '', group_name: '' });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const update = (key) => (event) => setForm((prev) => ({ ...prev, [key]: event.target.value }));

  const submit = async (event) => {
    event.preventDefault();
    setLoading(true);
    setError('');
    try {
      if (mode === 'login') await onLogin(form.username, form.password);
      else await onRegister(form);
    } catch (err) {
      setError(err?.response?.data?.detail || 'Не удалось выполнить запрос');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-panel">
        <div className="auth-copy">
          <div className="eyebrow">ВКР · ML-адаптация</div>
          <h1>Адаптивный тренажёр цифровой грамотности</h1>
          <p>Система прогнозирует вероятность успешного ответа по истории обучения и выбирает задания в оптимальной зоне сложности.</p>
          <div className="auth-points">
            <span>Feature engineering</span>
            <span>Сравнение моделей</span>
            <span>Версионирование ML</span>
            <span>Экзамены и аналитика</span>
          </div>
        </div>
        <form className="auth-card" onSubmit={submit}>
          <div className="tabs">
            <button type="button" className={mode === 'login' ? 'tab active' : 'tab'} onClick={() => setMode('login')}>Вход</button>
            <button type="button" className={mode === 'register' ? 'tab active' : 'tab'} onClick={() => setMode('register')}>Регистрация</button>
          </div>
          <label>Логин<input value={form.username} onChange={update('username')} autoComplete="username" /></label>
          <label>Пароль<input type="password" value={form.password} onChange={update('password')} autoComplete={mode === 'login' ? 'current-password' : 'new-password'} /></label>
          {mode === 'register' ? <>
            <label>ФИО<input value={form.full_name} onChange={update('full_name')} /></label>
            <label>Группа<input value={form.group_name} onChange={update('group_name')} /></label>
          </> : null}
          {error ? <div className="error-box">{error}</div> : null}
          <button className="primary full" disabled={loading}>{loading ? 'Отправка…' : mode === 'login' ? 'Войти' : 'Создать аккаунт'}</button>
          <div className="demo-note">
            Демо: <code>student_demo / 1234</code>, <code>teacher_demo / 1234</code>, <code>admin_demo / 1234</code>
          </div>
        </form>
      </div>
    </div>
  );
}
