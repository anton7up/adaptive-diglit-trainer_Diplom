import { useEffect, useState } from 'react';
import api from '../api';

export default function AdminUsersPage() {
  const [users, setUsers] = useState([]);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');

  const load = async () => {
    try {
      const { data } = await api.get('/admin/users');
      setUsers(data);
    } catch (err) {
      setError(err?.response?.data?.detail || 'Не удалось загрузить пользователей');
    }
  };

  useEffect(() => { load(); }, []);

  const setRole = async (user, role) => {
    setError(''); setMessage('');
    try {
      await api.patch(`/admin/users/${user.id}/role`, null, { params: { role } });
      setMessage(`Роль ${user.username} изменена на ${role}.`);
      await load();
    } catch (err) { setError(err?.response?.data?.detail || 'Не удалось изменить роль'); }
  };

  const setActive = async (user, active) => {
    setError(''); setMessage('');
    try {
      await api.patch(`/admin/users/${user.id}/active`, null, { params: { active } });
      setMessage(`${user.username}: ${active ? 'аккаунт активирован' : 'аккаунт отключён'}.`);
      await load();
    } catch (err) { setError(err?.response?.data?.detail || 'Не удалось изменить статус'); }
  };

  return (
    <div>
      <header className="page-header"><div><div className="eyebrow">Администрирование</div><h1>Пользователи и роли</h1></div><button className="secondary" onClick={load}>Обновить</button></header>
      {error ? <div className="error-box">{error}</div> : null}{message ? <div className="success-box">{message}</div> : null}
      <section className="card"><div className="data-table"><div className="table-row head users-table"><span>Пользователь</span><span>Роль</span><span>Группа</span><span>Статус</span><span>Действия</span></div>{users.map((user) => <div className="table-row users-table" key={user.id}><span><strong>{user.full_name || user.username}</strong><small>{user.username}</small></span><span><select value={user.role} onChange={(e) => setRole(user, e.target.value)}><option value="student">student</option><option value="teacher">teacher</option><option value="admin">admin</option></select></span><span>{user.group_name || '—'}</span><span>{user.is_active ? 'активен' : 'отключён'}</span><span><button className="secondary small" onClick={() => setActive(user, !user.is_active)}>{user.is_active ? 'Отключить' : 'Включить'}</button></span></div>)}</div></section>
    </div>
  );
}
