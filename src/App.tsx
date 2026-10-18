import {
  Activity,
  AlertCircle,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  Bell,
  Box,
  CalendarDays,
  ChevronDown,
  CircleHelp,
  Database,
  FlaskConical,
  Gauge,
  Home,
  LineChart,
  Menu,
  Settings,
  Users,
  Zap,
} from 'lucide-react';

const navItems = [
  { label: 'Overview', icon: Home, active: true },
  { label: 'Models', icon: Box },
  { label: 'Drift Detection', icon: BarChart3 },
  { label: 'Performance', icon: LineChart },
  { label: 'Alerts', icon: Bell, badge: '3' },
  { label: 'Experiments', icon: FlaskConical },
  { label: 'Data & Logs', icon: Database },
  { label: 'Settings', icon: Settings },
];

const driftRows = [
  { name: 'MonthlyCharges', score: '0.28', status: 'Moderate', tone: 'moderate', trend: [8, 10, 9, 14, 13, 20, 18, 26] },
  { name: 'Tenure', score: '0.12', status: 'Low', tone: 'low', trend: [10, 12, 11, 12, 10, 11, 10, 12] },
  { name: 'Contract', score: '0.42', status: 'High', tone: 'high', trend: [9, 11, 12, 10, 15, 16, 18, 22] },
  { name: 'PaymentMethod', score: '0.08', status: 'Low', tone: 'low', trend: [11, 9, 10, 8, 10, 9, 9, 8] },
  { name: 'TotalCharges', score: '0.15', status: 'Low', tone: 'low', trend: [10, 12, 9, 11, 10, 12, 11, 13] },
];

const alerts = [
  { title: 'Contract distribution drift detected', detail: 'Drift score (0.42) exceeded threshold (0.3)', severity: 'High', tone: 'high', time: '12 min ago', icon: AlertCircle },
  { title: 'MonthlyCharges drift increasing', detail: 'Drift score (0.28) shows upward trend', severity: 'Medium', tone: 'moderate', time: '1 hr ago', icon: AlertCircle },
  { title: 'Prediction latency returned to normal', detail: 'Latency decreased to 42 ms', severity: 'Low', tone: 'low', time: '1 hr ago', icon: Activity },
  { title: 'Model accuracy improved', detail: 'Accuracy increased from 92.4% to 94.2%', severity: 'Low', tone: 'low', time: '3 hr ago', icon: ArrowUpRight },
];

function MiniTrend({ points, color = '#1688ff' }: { points: number[]; color?: string }) {
  const max = Math.max(...points);
  const min = Math.min(...points);
  const path = points.map((point, index) => `${index === 0 ? 'M' : 'L'} ${index * 14} ${28 - ((point - min) / Math.max(max - min, 1)) * 21}`).join(' ');
  return (
    <svg className="mini-trend" viewBox="0 0 98 32" preserveAspectRatio="none" aria-hidden="true">
      <path d={path} fill="none" stroke={color} strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function PerformanceChart() {
  const accuracy = [70, 67, 73, 72, 75, 72, 76, 74, 78, 76, 80, 78, 79, 82, 80, 83, 82, 84, 82, 85, 84, 86, 85, 87, 86, 87, 88, 87, 89, 88, 89, 90, 89, 91, 90, 91, 92, 91, 90, 92];
  const precision = [57, 55, 58, 56, 60, 58, 61, 60, 62, 60, 63, 61, 66, 64, 67, 65, 69, 67, 70, 69, 72, 70, 73, 72, 75, 74, 75, 76, 75, 77, 78, 77, 79, 78, 80, 79, 80, 81, 80, 82];
  const recall = [45, 47, 44, 46, 43, 46, 48, 47, 50, 49, 51, 49, 52, 51, 53, 50, 54, 52, 49, 52, 54, 53, 55, 53, 56, 55, 57, 56, 54, 58, 57, 59, 57, 60, 59, 61, 60, 62, 61, 63];
  const makePath = (values: number[]) => values.map((value, index) => `${index === 0 ? 'M' : 'L'} ${index * 12.5} ${148 - value * 1.18}`).join(' ');
  const dots = (values: number[], color: string) => values.map((value, index) => <circle key={`${color}-${index}`} cx={index * 12.5} cy={148 - value * 1.18} r="2.2" fill="white" stroke={color} strokeWidth="1.8" />);
  return (
    <div className="chart-shell">
      <div className="chart-grid" />
      <div className="y-labels"><span>1.0</span><span>0.8</span><span>0.6</span><span>0.4</span><span>0.2</span><span>0</span></div>
      <svg className="performance-svg" viewBox="0 0 490 160" preserveAspectRatio="none" aria-label="Model performance over time">
        <path d={makePath(accuracy)} fill="none" stroke="#147ff0" strokeWidth="2" />
        <path d={makePath(precision)} fill="none" stroke="#0ab56b" strokeWidth="2" />
        <path d={makePath(recall)} fill="none" stroke="#8051e9" strokeWidth="2" />
        {dots(accuracy, '#147ff0')}{dots(precision, '#0ab56b')}{dots(recall, '#8051e9')}
      </svg>
      <div className="x-labels"><span>Apr 20</span><span>Apr 23</span><span>Apr 26</span><span>Apr 29</span><span>May 02</span><span>May 05</span><span>May 08</span><span>May 11</span><span>May 14</span><span>May 17</span><span>May 20</span></div>
      <div className="axis-title y-axis-title">Metric Value</div>
      <div className="legend"><span><i className="dot blue" />Accuracy</span><span><i className="dot green" />Precision</span><span><i className="dot purple" />Recall</span></div>
    </div>
  );
}

function DistributionChart() {
  const bars = Array.from({ length: 56 }, (_, index) => {
    const x = index / 55 * 100;
    const first = Math.exp(-Math.pow((x - 12) / 7, 2)) * 80 + Math.exp(-Math.pow((x - 38) / 8, 2)) * 70 + Math.exp(-Math.pow((x - 90) / 6, 2)) * 32;
    const second = Math.exp(-Math.pow((x - 16) / 7, 2)) * 68 + Math.exp(-Math.pow((x - 42) / 8, 2)) * 76 + Math.exp(-Math.pow((x - 92) / 8, 2)) * 29;
    return { x, first, second };
  });
  return (
    <div className="distribution-shell">
      <div className="distribution-grid" />
      <svg viewBox="0 0 520 150" preserveAspectRatio="none" aria-label="Feature distribution chart" className="distribution-svg">
        {bars.map((bar, index) => <g key={index}><rect x={bar.x * 5.05} y={140 - bar.first} width="4" height={bar.first} fill="#4d9aff" opacity=".55" /><rect x={bar.x * 5.05 + 3} y={140 - bar.second} width="4" height={bar.second} fill="#ff8250" opacity=".42" /></g>)}
        <path d="M0 139 C45 137 50 38 75 36 S115 135 170 137 S205 44 220 42 S255 139 310 138 S350 39 380 45 S414 136 450 138 S485 91 520 114" fill="none" stroke="#328bf5" strokeWidth="2" />
        <path d="M0 141 C55 139 72 52 95 55 S130 142 180 140 S220 49 245 53 S285 140 330 138 S365 47 395 55 S430 140 465 136 S500 99 520 119" fill="none" stroke="#ff7545" strokeWidth="2" />
      </svg>
      <div className="distribution-labels"><span>0</span><span>0.5</span><span>1.0</span><span>1.5</span><span>2.0</span><span>2.5</span><span>3.0</span></div>
      <div className="axis-title distribution-y-title">Density</div><div className="distribution-x-title">Feature value</div>
    </div>
  );
}

function MetricCard({ icon: Icon, label, value, note, trend, color, positive = true }: { icon: typeof Gauge; label: string; value: string; note: string; trend: number[]; color: string; positive?: boolean }) {
  return <div className="metric-card"><div className="metric-icon" style={{ color, background: `${color}12` }}><Icon size={21} strokeWidth={2.2} /></div><div className="metric-copy"><span>{label}</span><strong>{value}</strong><small className={positive ? 'positive' : 'negative'}>{positive ? <ArrowUpRight size={13} /> : <ArrowDownRight size={13} />}{note}</small></div><MiniTrend points={trend} color={color} /></div>;
}

function App() {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><div className="brand-mark"><BarChart3 size={23} /><LineChart size={14} /></div><div><strong>ModelWatch</strong><span>Monitor · Detect · Keep AI Reliable</span></div></div>
        <nav className="main-nav">{navItems.map(({ label, icon: Icon, active, badge }) => <button className={active ? 'nav-item active' : 'nav-item'} key={label}><Icon size={17} /><span>{label}</span>{badge && <b>{badge}</b>}</button>)}</nav>
        <div className="sidebar-footer"><div className="footer-art"><div><strong>ModelWatch</strong><span>Reliable AI for<br />Real-World Impact</span></div><LineChart size={42} /></div></div>
      </aside>
      <main className="main-content">
        <header className="topbar"><button className="mobile-menu"><Menu size={20} /></button><div className="environment"><span className="live-dot" />Production Monitoring <em><span className="tiny-dot" /> Monitoring active</em></div><div className="top-actions"><button><CalendarDays size={15} />Last 30 days <ChevronDown size={13} /></button><button>Customer Churn v1.4 <ChevronDown size={13} /></button><div className="avatar">RS</div></div></header>
        <div className="content-wrap">
          <section className="page-heading"><div><h1>Model health overview</h1><p>Monitor performance, detect drift, and investigate model changes.</p></div></section>
          <section className="metrics-grid"><MetricCard icon={Gauge} label="Model accuracy" value="94.2%" note="+1.8% vs previous period" trend={[11, 13, 12, 15, 14, 17, 15, 23]} color="#1688ff" /><MetricCard icon={Database} label="Data drift" value="3 features flagged" note="+2 vs previous period" trend={[10, 11, 12, 11, 16, 17, 25, 29]} color="#ff5d62" positive={false} /><MetricCard icon={Users} label="Prediction volume" value="24,891" note="+12.4% vs previous period" trend={[10, 13, 15, 18, 16, 23, 27, 30]} color="#438eff" /><MetricCard icon={Zap} label="Avg. inference latency" value="42 ms" note="−8 ms vs previous period" trend={[20, 16, 20, 13, 20, 15, 22, 17]} color="#8452ed" /> </section>
          <section className="dashboard-grid"><div className="panel performance-panel"><PanelHeader icon={LineChart} title="Model performance over time" /><PerformanceChart /></div><div className="panel drift-panel"><PanelHeader icon={Database} title="Drift status" action="View all" /><div className="drift-table"><div className="table-head"><span>Feature</span><span>Drift Score</span><span>Status</span><span>Trend</span></div>{driftRows.map(row => <div className="table-row" key={row.name}><span>{row.name}</span><span>{row.score}</span><span><em className={`status ${row.tone}`}>{row.status}</em></span><MiniTrend points={row.trend} color={row.tone === 'low' ? '#0ab56b' : row.tone === 'high' ? '#ff5d62' : '#ff7545'} /></div>)}</div></div><div className="panel distribution-panel"><PanelHeader icon={BarChart3} title="Feature distribution" /><div className="distribution-controls"><span><i className="dot blue" />Training data</span><span><i className="dot orange" />Production data</span></div><DistributionChart /></div><div className="panel alerts-panel"><PanelHeader icon={Bell} title="Recent alerts" action="View all" /><div className="alerts-list">{alerts.map(({ title, detail, severity, tone, time, icon: Icon }) => <div className="alert-row" key={title}><div className={`alert-icon ${tone}`}><Icon size={14} /></div><div className="alert-copy"><strong>{title}</strong><span>{detail}</span></div><em className={`status ${tone}`}>{severity}</em><time>{time}</time></div>)}</div></div></section>
        </div>
      </main>
    </div>
  );
}

function PanelHeader({ icon: Icon, title, action }: { icon: typeof LineChart; title: string; action?: string }) {
  return <div className="panel-header"><div className="panel-title"><div className="panel-icon"><Icon size={16} /></div><h2>{title}</h2><CircleHelp size={13} /></div>{action ? <button className="panel-action">{action} <ArrowUpRight size={12} /></button> : <button className="select-button">Last 30 days <ChevronDown size={12} /></button>}</div>;
}

export default App;
