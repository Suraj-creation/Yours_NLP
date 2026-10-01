import { useMemo, useState, type ReactNode } from 'react'
import clsx from 'clsx'
import { ArrowDownUp, ChevronDown, Download, Info, Loader2, TriangleAlert } from 'lucide-react'
import { fileUrl } from '../lib/api'
import { fmt } from '../lib/palette'

export function PageHeader({ eyebrow, title, lead, children }: { eyebrow?: string; title: string; lead?: ReactNode; children?: ReactNode }) {
  return (
    <header className="mb-8">
      {eyebrow && <div className="mb-2 text-xs font-semibold uppercase tracking-[0.12em] text-accent-ink">{eyebrow}</div>}
      <h1 className="text-[28px] font-semibold leading-tight tracking-tight text-ink">{title}</h1>
      {lead && <p className="mt-3 max-w-3xl text-[15px] leading-relaxed text-ink-2">{lead}</p>}
      {children}
    </header>
  )
}

export function Card({ title, subtitle, children, actions, className, pad = true }: {
  title?: ReactNode; subtitle?: ReactNode; children: ReactNode; actions?: ReactNode; className?: string; pad?: boolean
}) {
  return (
    <section className={clsx('rounded-2xl border border-line bg-surface shadow-[var(--shadow)]', className)}>
      {(title || actions) && (
        <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2 border-b border-line px-5 py-4">
          <div className="min-w-0">
            {title && <h2 className="text-[15px] font-semibold text-ink">{title}</h2>}
            {subtitle && <p className="mt-0.5 text-[13px] leading-snug text-ink-3">{subtitle}</p>}
          </div>
          {actions && <div className="flex max-w-full flex-wrap items-center gap-2">{actions}</div>}
        </div>
      )}
      <div className={clsx(pad && 'p-5')}>{children}</div>
    </section>
  )
}

export function Stat({ label, value, note, tone }: { label: string; value: ReactNode; note?: ReactNode; tone?: 'good' | 'bad' }) {
  return (
    <div className="rounded-2xl border border-line bg-surface px-5 py-4 shadow-[var(--shadow)]">
      <div className="text-[13px] text-ink-3">{label}</div>
      <div className="mt-1 text-[26px] font-semibold tracking-tight text-ink">{value}</div>
      {note && <div className={clsx('mt-1 text-xs', tone === 'good' ? 'text-good-ink' : tone === 'bad' ? 'text-bad' : 'text-ink-3')}>{note}</div>}
    </div>
  )
}

export function Grid({ cols = 2, children, className }: { cols?: number; children: ReactNode; className?: string }) {
  const c = { 1: 'lg:grid-cols-1', 2: 'lg:grid-cols-2', 3: 'lg:grid-cols-3', 4: 'lg:grid-cols-4', 5: 'lg:grid-cols-5', 6: 'lg:grid-cols-6' }[cols]
  return <div className={clsx('grid grid-cols-1 gap-5 sm:grid-cols-2', c, className)}>{children}</div>
}

export function Section({ title, lead, children }: { title: string; lead?: ReactNode; children: ReactNode }) {
  return (
    <section className="mt-10">
      <h2 className="text-lg font-semibold tracking-tight text-ink">{title}</h2>
      {lead && <p className="mt-1 mb-4 max-w-3xl text-sm leading-relaxed text-ink-2">{lead}</p>}
      {!lead && <div className="mb-4" />}
      <div className="space-y-5">{children}</div>
    </section>
  )
}

/** "Why this step, where it sits, what it depends on, what happens if it moves" (Module 3). */
export function Why({ why, where, depends, moved, open: initial = false }: { why: ReactNode; where: ReactNode; depends: ReactNode; moved: ReactNode; open?: boolean }) {
  const [open, setOpen] = useState(initial)
  return (
    <div className="mt-5 rounded-2xl border border-accent/25 bg-accent-soft/60">
      <button onClick={() => setOpen(!open)} className="flex w-full items-center gap-2 px-5 py-3 text-left text-sm font-medium text-accent-ink">
        <Info size={16} /> Why this step, where it sits, and what happens if it moves
        <ChevronDown size={16} className={clsx('ml-auto transition-transform', open && 'rotate-180')} />
      </button>
      {open && (
        <dl className="grid gap-x-8 gap-y-4 px-5 pb-5 text-sm leading-relaxed sm:grid-cols-2">
          {[['Why it is needed', why], ['Where it is placed', where], ['What it depends on', depends], ['If its position changes', moved]].map(([k, v]) => (
            <div key={k as string}>
              <dt className="text-xs font-semibold uppercase tracking-wide text-ink-3">{k}</dt>
              <dd className="mt-1 text-ink-2">{v}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  )
}

export function Callout({ children, tone = 'info' }: { children: ReactNode; tone?: 'info' | 'warn' }) {
  return (
    <div className={clsx('flex gap-3 rounded-xl border px-4 py-3 text-sm leading-relaxed',
      tone === 'warn' ? 'border-warn/40 bg-warn/10 text-ink-2' : 'border-line bg-surface-2 text-ink-2')}>
      {tone === 'warn' ? <TriangleAlert size={16} className="mt-0.5 shrink-0 text-warn" /> : <Info size={16} className="mt-0.5 shrink-0 text-ink-3" />}
      <div>{children}</div>
    </div>
  )
}

export function Pill({ children, tone = 'neutral', className }: { children: ReactNode; tone?: 'neutral' | 'accent' | 'good' | 'bad' | 'warn'; className?: string }) {
  const t = {
    neutral: 'bg-surface-2 text-ink-2 border-line', accent: 'bg-accent-soft text-accent-ink border-accent/20',
    good: 'bg-good/10 text-good-ink border-good/25', bad: 'bg-bad/10 text-bad border-bad/25', warn: 'bg-warn/15 text-ink-2 border-warn/30',
  }[tone]
  return <span className={clsx('inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium', t, className)}>{children}</span>
}

export function Button({ children, onClick, variant = 'primary', disabled, type = 'button', className }: {
  children: ReactNode; onClick?: () => void; variant?: 'primary' | 'ghost' | 'outline'; disabled?: boolean; type?: 'button' | 'submit'; className?: string
}) {
  const v = {
    primary: 'bg-accent text-white hover:brightness-110 disabled:opacity-50',
    outline: 'border border-line-strong bg-surface text-ink hover:bg-surface-2',
    ghost: 'text-ink-2 hover:bg-surface-2',
  }[variant]
  return (
    <button type={type} onClick={onClick} disabled={disabled}
      className={clsx('inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2 text-sm font-medium transition', v, className)}>
      {children}
    </button>
  )
}

export function Select({ value, onChange, options, label, className }: {
  value: string; onChange: (v: string) => void; options: { value: string; label: string }[]; label?: string; className?: string
}) {
  return (
    <label className={clsx('flex flex-col gap-1 text-xs font-medium text-ink-3', className)}>
      {label}
      <select value={value} onChange={(e) => onChange(e.target.value)} aria-label={label}
        className="rounded-xl border border-line-strong bg-surface px-3 py-2 text-sm text-ink outline-none focus:border-accent">
        {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </label>
  )
}

export function Segmented({ value, onChange, options }: { value: string; onChange: (v: string) => void; options: { value: string; label: string }[] }) {
  return (
    <div className="inline-flex max-w-full flex-wrap rounded-xl border border-line bg-surface-2 p-0.5">
      {options.map((o) => (
        <button key={o.value} onClick={() => onChange(o.value)}
          className={clsx('rounded-[10px] px-3 py-1.5 text-xs font-medium transition',
            value === o.value ? 'bg-surface text-ink shadow-sm' : 'text-ink-3 hover:text-ink')}>
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function TextArea({ value, onChange, rows = 3, placeholder }: { value: string; onChange: (v: string) => void; rows?: number; placeholder?: string }) {
  return (
    <textarea value={value} onChange={(e) => onChange(e.target.value)} rows={rows} placeholder={placeholder}
      className="w-full resize-y rounded-xl border border-line-strong bg-surface px-4 py-3 text-[15px] leading-relaxed text-ink outline-none placeholder:text-ink-3 focus:border-accent" />
  )
}

export function Examples({ items, onPick }: { items: string[]; onPick: (s: string) => void }) {
  return (
    <div className="mt-2 flex flex-wrap gap-1.5">
      {items.map((s) => (
        <button key={s} onClick={() => onPick(s)}
          className="max-w-full truncate rounded-full border border-line bg-surface-2 px-2.5 py-1 text-xs text-ink-2 hover:border-accent/40 hover:text-accent-ink">
          {s}
        </button>
      ))}
    </div>
  )
}

export function Loading({ label = 'Loading' }: { label?: string }) {
  return <div className="flex items-center gap-2 py-10 text-sm text-ink-3"><Loader2 size={16} className="animate-spin" /> {label}</div>
}

export function ErrorBox({ error }: { error: unknown }) {
  return <Callout tone="warn">{String((error as Error)?.message ?? error)}</Callout>
}

export function DownloadLink({ file, label }: { file: string; label?: string }) {
  return (
    <a href={fileUrl(file)} className="inline-flex max-w-full items-center gap-1.5 break-all rounded-lg px-2 py-1 text-xs font-medium text-ink-3 hover:bg-surface-2 hover:text-ink">
      <Download size={14} /> {label ?? file}
    </a>
  )
}

export type Col<T> = { key: string; label: string; render?: (row: T) => ReactNode; num?: boolean; width?: string; digits?: number }

export function DataTable<T extends Record<string, unknown>>({ rows, cols, max = 400, dense, initialSort }: {
  rows: T[]; cols: Col<T>[]; max?: number; dense?: boolean; initialSort?: { key: string; dir: 1 | -1 }
}) {
  const [sort, setSort] = useState<{ key: string; dir: 1 | -1 } | null>(initialSort ?? null)
  const sorted = useMemo(() => {
    if (!sort) return rows
    return [...rows].sort((a, b) => {
      const x = a[sort.key] as never, y = b[sort.key] as never
      if (x === y) return 0
      if (x === null || x === undefined) return 1
      if (y === null || y === undefined) return -1
      return (x > y ? 1 : -1) * sort.dir
    })
  }, [rows, sort])
  // a numeric column with any fractional value is shown with the same decimals on every row
  const floatCol = useMemo(() => Object.fromEntries(cols.map((c) => [c.key,
    rows.some((r) => typeof r[c.key] === 'number' && !Number.isInteger(r[c.key] as number))])), [rows, cols])
  return (
    <div className="scroll-thin max-h-[560px] overflow-auto rounded-xl border border-line">
      <table className="w-full border-collapse text-left text-[13px]">
        <thead className="sticky top-0 z-10 bg-surface-2">
          <tr>
            {cols.map((c) => (
              <th key={c.key} style={{ width: c.width }}
                className={clsx('whitespace-nowrap border-b border-line px-3 py-2 text-xs font-semibold text-ink-3', c.num && 'text-right')}>
                <button className="inline-flex items-center gap-1 hover:text-ink"
                  onClick={() => setSort(sort?.key === c.key ? { key: c.key, dir: (sort.dir * -1) as 1 | -1 } : { key: c.key, dir: c.num ? -1 : 1 })}>
                  {c.label}<ArrowDownUp size={11} className="opacity-40" />
                </button>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.slice(0, max).map((r, i) => (
            <tr key={i} className="border-b border-line/70 last:border-0 hover:bg-surface-2/60">
              {cols.map((c) => {
                const v = r[c.key]
                return (
                  <td key={c.key} className={clsx(dense ? 'px-3 py-1.5' : 'px-3 py-2', 'align-top text-ink-2', c.num && 'num text-right')}>
                    {c.render ? c.render(r) : c.num && typeof v === 'number' ? fmt(v, c.digits ?? (floatCol[c.key] ? 3 : 0)) : String(v ?? '')}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > max && <div className="px-3 py-2 text-xs text-ink-3">Showing {max} of {rows.length} rows — download the CSV for all.</div>}
    </div>
  )
}

export function Bool({ v }: { v: unknown }) {
  const yes = v === true || v === 'True' || v === 'true'
  return <Pill tone={yes ? 'good' : 'bad'}>{yes ? 'yes' : 'no'}</Pill>
}
