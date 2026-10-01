/* Small display components shared by the NLP pages: token chips, entity highlights,
   search snippets, multi-select pills and a metric strip. */
import { useState, type ReactNode } from 'react'
import clsx from 'clsx'
import { Check, Copy } from 'lucide-react'
import { fmt, layerColor, slot } from '../lib/palette'

export function TokenChips({ tokens, mark, max = 400 }: { tokens: string[]; mark?: (t: string, i: number) => boolean; max?: number }) {
  return (
    <div className="flex flex-wrap gap-1">
      {tokens.slice(0, max).map((t, i) => (
        <span key={i}
          className={clsx('rounded-md border px-1.5 py-0.5 font-mono text-[12.5px] leading-5',
            mark?.(t, i) ? 'border-accent/50 bg-accent-soft text-accent-ink' : 'border-line bg-surface-2 text-ink')}>
          {t === ' ' ? '␠' : t.replace(/\n/g, '⏎')}
        </span>
      ))}
      {tokens.length > max && <span className="px-1 text-xs text-ink-3">+{tokens.length - max} more</span>}
    </div>
  )
}

/** Entity types: the five domain types get categorical slots; the seven standard types
    share one neutral style. Every highlight also prints its label, so colour is never alone. */
const DOMAIN_SLOT: Record<string, number> = { CONCEPT: 1, MISCONCEPTION: 8, KT_MODEL: 7, DATASET: 3, METRIC: 4 }
export const entityColor = (label: string) => (DOMAIN_SLOT[label] ? slot(DOMAIN_SLOT[label]) : 'var(--ink-3)')

export type Ent = { text: string; label: string; start: number; end: number; source?: string }

export function Highlight({ text, ents }: { text: string; ents: Ent[] }) {
  const sorted = [...ents].sort((a, b) => a.start - b.start)
  const out: ReactNode[] = []
  let cur = 0
  sorted.forEach((e, i) => {
    if (e.start < cur) return
    if (e.start > cur) out.push(<span key={`t${i}`}>{text.slice(cur, e.start)}</span>)
    const c = entityColor(e.label)
    out.push(
      <mark key={`e${i}`} className="mx-0.5 inline-flex items-baseline gap-1 rounded-md px-1.5 py-0.5"
        style={{ background: `color-mix(in oklab, ${c} 16%, transparent)`, boxShadow: `inset 0 0 0 1px color-mix(in oklab, ${c} 45%, transparent)`, color: 'var(--ink)' }}>
        {text.slice(e.start, e.end)}
        <span className="text-[10px] font-semibold tracking-wide" style={{ color: 'var(--ink-2)' }}>{e.label}</span>
      </mark>,
    )
    cur = e.end
  })
  out.push(<span key="end">{text.slice(cur)}</span>)
  return <p className="text-[15px] leading-9 text-ink">{out}</p>
}

export function Snippet({ parts }: { parts: { text: string; hit: boolean }[] }) {
  return (
    <p className="text-[13.5px] leading-relaxed text-ink-2">
      {parts.map((p, i) => (p.hit
        ? <mark key={i} className="rounded bg-[var(--hit)] px-0.5 text-ink">{p.text}</mark>
        : <span key={i}>{p.text}</span>))}
    </p>
  )
}

export function LayerDot({ layer }: { layer: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-ink-2">
      <span className="h-2 w-2 rounded-full" style={{ background: layerColor(layer) }} />{layer}
    </span>
  )
}

export function MultiPills({ options, value, onChange }: { options: { value: string; label: string }[]; value: string[]; onChange: (v: string[]) => void }) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {options.map((o) => {
        const on = value.includes(o.value)
        return (
          <button key={o.value} onClick={() => onChange(on ? value.filter((x) => x !== o.value) : [...value, o.value])}
            aria-pressed={on}
            className={clsx('inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs font-medium transition',
              on ? 'border-accent/40 bg-accent-soft text-accent-ink' : 'border-line bg-surface text-ink-3 hover:text-ink')}>
            {on && <Check size={12} />}{o.label}
          </button>
        )
      })}
    </div>
  )
}

export function Metrics({ items }: { items: { label: string; value: number | string | null | undefined; digits?: number; note?: string }[] }) {
  return (
    <div className="grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-line bg-line sm:grid-cols-3 lg:grid-cols-6">
      {items.map((m) => (
        <div key={m.label} className="bg-surface px-4 py-3">
          <div className="text-[11px] font-medium uppercase tracking-wide text-ink-3">{m.label}</div>
          <div className="num mt-0.5 text-lg font-semibold text-ink">
            {typeof m.value === 'number' ? fmt(m.value, m.digits ?? (Number.isInteger(m.value) ? 0 : 3)) : (m.value ?? '—')}
          </div>
          {m.note && <div className="text-[11px] text-ink-3">{m.note}</div>}
        </div>
      ))}
    </div>
  )
}

export function JsonView({ data }: { data: unknown }) {
  const [copied, setCopied] = useState(false)
  const s = JSON.stringify(data, null, 2)
  return (
    <div className="relative">
      <button onClick={() => { navigator.clipboard?.writeText(s); setCopied(true); setTimeout(() => setCopied(false), 1200) }}
        className="absolute right-2 top-2 inline-flex items-center gap-1 rounded-md border border-line bg-surface px-2 py-1 text-xs text-ink-3 hover:text-ink">
        {copied ? <Check size={12} /> : <Copy size={12} />} {copied ? 'Copied' : 'Copy'}
      </button>
      <pre className="scroll-thin max-h-[520px] overflow-auto rounded-xl border border-line bg-surface-2 p-4 font-mono text-[12px] leading-relaxed text-ink-2">{s}</pre>
    </div>
  )
}

export function Bar({ value, max = 1, color = 'var(--accent)' }: { value: number; max?: number; color?: string }) {
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-surface-3">
      <div className="h-full rounded-full" style={{ width: `${Math.max(0, Math.min(1, value / max)) * 100}%`, background: color }} />
    </div>
  )
}

export function Mono({ children }: { children: ReactNode }) {
  return <code className="rounded bg-surface-2 px-1 py-0.5 font-mono text-[12px] text-ink">{children}</code>
}
