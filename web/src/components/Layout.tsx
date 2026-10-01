import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import clsx from 'clsx'
import {
  BarChart3, BookOpenText, Boxes, Braces, Database, Download, FileSearch, FlaskConical, GitCompareArrows, Home,
  Layers, ListTree, Menu, Moon, Network, ScanText, Search, Sparkles, SplitSquareHorizontal, Sun, Tags, X,
} from 'lucide-react'
import { useMeta } from '../lib/api'
import { useTheme } from '../lib/theme'
import { runColor } from '../lib/palette'

export const NAV: { group: string; items: { to: string; label: string; icon: typeof Home; tag?: string }[] }[] = [
  { group: 'Overview', items: [{ to: '/', label: 'Home', icon: Home }] },
  { group: 'Corpus', items: [
    { to: '/corpus', label: 'Documents', icon: Database, tag: 'M1' },
    { to: '/statistics', label: 'Statistics', icon: BarChart3, tag: 'M1' },
    { to: '/scale', label: 'Heavy datasets', icon: Boxes },
  ] },
  { group: 'Text processing', items: [
    { to: '/tokenization', label: 'Tokenization', icon: SplitSquareHorizontal, tag: 'Ex 2·3·6' },
    { to: '/preprocessing', label: 'Preprocessing', icon: Layers, tag: 'Ex 1·7–11' },
    { to: '/bpe', label: 'BPE', icon: Braces, tag: 'Ex 4' },
  ] },
  { group: 'Linguistics', items: [
    { to: '/pos', label: 'POS tagging', icon: Tags, tag: 'Ex 5·12' },
    { to: '/custom-pos', label: 'Custom POS', icon: FlaskConical, tag: 'Ex 12' },
    { to: '/ner', label: 'Named entities', icon: ScanText, tag: 'Ex 13' },
    { to: '/ngrams', label: 'N-grams', icon: Network, tag: 'Ex 14' },
  ] },
  { group: 'Retrieval', items: [
    { to: '/index', label: 'Inverted index', icon: ListTree, tag: 'M4' },
    { to: '/search', label: 'Search', icon: Search, tag: 'M4·5' },
    { to: '/pipelines', label: 'Pipelines', icon: GitCompareArrows, tag: 'M3' },
    { to: '/evaluation', label: 'Evaluation', icon: FileSearch, tag: 'M6' },
  ] },
  { group: 'Research', items: [
    { to: '/justification', label: 'Justification', icon: BookOpenText, tag: 'M3' },
    { to: '/evidence', label: 'Evidence preview', icon: Sparkles },
    { to: '/downloads', label: 'Downloads', icon: Download },
  ] },
]

type Ctx = { pipeline: string; setPipeline: (p: string) => void }
const PipelineCtx = createContext<Ctx>({ pipeline: 'B', setPipeline: () => {} })
export const usePipeline = () => useContext(PipelineCtx)

export function PipelineProvider({ children }: { children: ReactNode }) {
  const meta = useMeta()
  const [pipeline, setP] = useState<string>(() => { try { return localStorage.getItem('pipeline') ?? '' } catch { return '' } })
  const setPipeline = (p: string) => { setP(p); try { localStorage.setItem('pipeline', p) } catch { /* ignore */ } }
  return <PipelineCtx.Provider value={{ pipeline: pipeline || meta.data?.winner || 'B', setPipeline }}>{children}</PipelineCtx.Provider>
}

function PipelinePicker() {
  const meta = useMeta()
  const { pipeline, setPipeline } = usePipeline()
  if (!meta.data) return null
  return (
    <label className="flex items-center gap-2 text-xs text-ink-3">
      <span className="hidden sm:inline">Pipeline</span>
      <span className="h-2.5 w-2.5 rounded-full" style={{ background: runColor(pipeline) }} />
      <select value={pipeline} onChange={(e) => setPipeline(e.target.value)} aria-label="Active pipeline"
        className="max-w-[170px] rounded-lg border border-line bg-surface px-2 py-1.5 text-[13px] text-ink outline-none sm:max-w-none">
        {meta.data.pipelines.map((p: { name: string; label: string }) => (
          <option key={p.name} value={p.name}>{p.name === meta.data.winner ? `${p.label} (selected)` : p.label}</option>
        ))}
      </select>
    </label>
  )
}

export default function Layout() {
  const { theme, toggle } = useTheme()
  const [open, setOpen] = useState(false)
  const loc = useLocation()
  useEffect(() => { setOpen(false); window.scrollTo(0, 0) }, [loc.pathname])
  const nav = (
    <nav className="scroll-thin flex h-full flex-col gap-6 overflow-y-auto px-3 py-5">
      {NAV.map((g) => (
        <div key={g.group}>
          <div className="px-3 pb-1.5 text-[11px] font-semibold uppercase tracking-[0.12em] text-ink-3">{g.group}</div>
          <ul className="space-y-0.5">
            {g.items.map((it) => (
              <li key={it.to}>
                <NavLink to={it.to} end={it.to === '/'}
                  className={({ isActive }) => clsx('group flex items-center gap-2.5 rounded-xl px-3 py-2 text-[13.5px] transition',
                    isActive ? 'bg-accent-soft font-medium text-accent-ink' : 'text-ink-2 hover:bg-surface-2 hover:text-ink')}>
                  <it.icon size={16} className="shrink-0 opacity-80" />
                  <span className="truncate">{it.label}</span>
                  {it.tag && <span className="ml-auto text-[10px] text-ink-3 opacity-70 group-hover:opacity-100">{it.tag}</span>}
                </NavLink>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </nav>
  )
  return (
    <div className="min-h-full">
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-64 border-r border-line bg-surface lg:block">
        <Brand />
        <div className="h-[calc(100%-64px)]">{nav}</div>
      </aside>
      {open && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-black/30" onClick={() => setOpen(false)} />
          <aside className="absolute inset-y-0 left-0 w-72 border-r border-line bg-surface">
            <div className="flex items-center justify-between pr-3"><Brand /><button onClick={() => setOpen(false)} aria-label="Close menu"><X size={18} /></button></div>
            <div className="h-[calc(100%-64px)]">{nav}</div>
          </aside>
        </div>
      )}
      <div className="lg:pl-64">
        <header className="sticky top-0 z-20 flex h-16 items-center gap-3 border-b border-line bg-surface/85 px-4 backdrop-blur sm:px-8">
          <button className="lg:hidden" onClick={() => setOpen(true)} aria-label="Open menu"><Menu size={20} /></button>
          <div className="hidden truncate text-[13px] text-ink-3 md:block">NLP Assessment-1 · Domain-specific text analysis and retrieval</div>
          <div className="ml-auto flex min-w-0 items-center gap-2 sm:gap-3">
            <PipelinePicker />
            <button onClick={toggle} aria-label="Toggle theme" className="rounded-lg border border-line p-2 text-ink-2 hover:bg-surface-2">
              {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
            </button>
          </div>
        </header>
        <main className="mx-auto max-w-[1280px] px-4 py-8 sm:px-8 sm:py-10">
          <Outlet />
        </main>
        <footer className="mx-auto max-w-[1280px] px-4 pb-10 text-xs text-ink-3 sm:px-8">
          Every number on this site is computed by <code className="font-mono">nlp_core</code> and matches the notebook and <code className="font-mono">results/</code> files.
        </footer>
      </div>
    </div>
  )
}

function Brand() {
  return (
    <div className="flex h-16 items-center gap-2.5 px-5">
      <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent text-sm font-bold text-white">½</div>
      <div className="leading-tight">
        <div className="text-[14px] font-semibold text-ink">Learner Language Lab</div>
        <div className="text-[11px] text-ink-3">Text analysis &amp; retrieval</div>
      </div>
    </div>
  )
}
