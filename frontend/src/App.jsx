import { useState } from 'react';
import { useSession } from './hooks/useSession';
import AppShell from './components/AppShell';
import AuthPage from './pages/AuthPage';
import PracticePage from './pages/PracticePage';
import ProgressPage from './pages/ProgressPage';
import StudentExamsPage from './pages/StudentExamsPage';
import TeacherDashboardPage from './pages/TeacherDashboardPage';
import MLPage from './pages/MLPage';
import TaskBankPage from './pages/TaskBankPage';
import TeacherExamsPage from './pages/TeacherExamsPage';
import ResearchPage from './pages/ResearchPage';
import AdminUsersPage from './pages/AdminUsersPage';

export default function App() {
  const session = useSession();
  const [active, setActive] = useState('practice');

  if (!session.user) return <AuthPage onLogin={session.login} onRegister={session.register} />;

  const user = session.user;
  const teacher = user.role === 'teacher' || user.role === 'admin';
  const effectiveActive = teacher && ['practice', 'progress'].includes(active) ? 'dashboard' : active;

  let page;
  if (!teacher && effectiveActive === 'practice') page = <PracticePage />;
  else if (!teacher && effectiveActive === 'progress') page = <ProgressPage />;
  else if (!teacher && effectiveActive === 'exams') page = <StudentExamsPage />;
  else if (teacher && effectiveActive === 'dashboard') page = <TeacherDashboardPage />;
  else if (teacher && effectiveActive === 'ml') page = <MLPage user={user} />;
  else if (teacher && effectiveActive === 'research') page = <ResearchPage />;
  else if (teacher && effectiveActive === 'tasks') page = <TaskBankPage />;
  else if (teacher && effectiveActive === 'exams') page = <TeacherExamsPage />;
  else if (teacher && effectiveActive === 'users' && user.role === 'admin') page = <AdminUsersPage />;
  else page = teacher ? <TeacherDashboardPage /> : <PracticePage />;

  return <AppShell user={user} onLogout={session.logout} active={effectiveActive} onNavigate={setActive}>{page}</AppShell>;
}
