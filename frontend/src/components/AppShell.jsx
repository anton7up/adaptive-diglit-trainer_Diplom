export default function AppShell({ user, onLogout, active, onNavigate, children }) {
  const studentItems = [
    ['practice', 'Практика'],
    ['progress', 'Прогресс'],
    ['exams', 'Экзамены'],
  ];
  const teacherItems = [
    ['dashboard', 'Аналитика'],
    ['ml', 'ML-модели'],
    ['research', 'Исследование'],
    ['tasks', 'Банк заданий'],
    ['exams', 'Экзамены'],
    ...(user.role === 'admin' ? [['users', 'Пользователи']] : []),
  ];
  const items = user.role === 'student' ? studentItems : teacherItems;
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">AD</div>
          <div>
            <strong>Adaptive Digital</strong>
            <span>ML Trainer</span>
          </div>
        </div>
        <nav>
          {items.map(([key, label]) => (
            <button key={key} className={active === key ? 'nav-item active' : 'nav-item'} onClick={() => onNavigate(key)}>
              {label}
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">
          <div className="user-chip">
            <strong>{user.full_name || user.username}</strong>
            <span>{user.role === 'student' ? 'Студент' : user.role === 'teacher' ? 'Преподаватель' : 'Администратор'}</span>
          </div>
          <button className="secondary full" onClick={onLogout}>Выйти</button>
        </div>
      </aside>
      <main className="main-area">{children}</main>
    </div>
  );
}
