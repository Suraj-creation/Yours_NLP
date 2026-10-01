import { Link } from 'react-router-dom'
import { ArrowRight } from 'lucide-react'
import { useMeta, useResult } from '../lib/api'
import { compact, fmt } from '../lib/palette'
import { Card, Grid, Loading, PageHeader, Pill, Stat } from '../components/ui'
import { NAV } from '../components/Layout'

function LoopDiagram() {
  const box = (x: number, label: string, sub: string, on: boolean) => (
    <g key={label}>
      <rect x={x} y={34} width={138} height={60} rx={12} fill={on ? 'var(--accent-soft)' : 'var(--surface)'}
        stroke={on ? 'var(--accent)' : 'var(--line-strong)'} strokeWidth={on ? 2 : 1.25} />
      <text x={x + 69} y={60} textAnchor="middle" fontSize={13} fontWeight={600} fill="var(--ink)">{label}</text>
      <text x={x + 69} y={79} textAnchor="middle" fontSize={11.5} fill="var(--ink-3)">{sub}</text>
    </g>
  )
  const xs = [10, 170, 330, 490, 650]
  return (
    <svg viewBox="0 0 800 150" role="img" aria-label="Assessment-1 builds the first three stages of the research loop" className="w-full">
      <defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="var(--line-strong)" /></marker></defs>
      <path d="M10 22V14H468V22" fill="none" stroke="var(--accent)" strokeWidth={1.25} />
      <text x={239} y={10} textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--accent-ink)">Built in Assessment-1</text>
      {[0, 1, 2, 3].map((i) => <path key={i} d={`M${xs[i] + 138} 64H${xs[i + 1]}`} stroke="var(--line-strong)" strokeWidth={1.25} markerEnd="url(#ar)" />)}
      <path d="M719 94V126H79V94" fill="none" stroke="var(--line-strong)" strokeWidth={1.25} markerEnd="url(#ar)" />
      <text x={400} y={144} textAnchor="middle" fontSize={11} fill="var(--ink-3)">the next diagnostic question produces new observations</text>
      {box(xs[0], 'Observations', 'student text, docs', true)}
      {box(xs[1], 'NLP pipeline', 'tokens to entities', true)}
      {box(xs[2], 'Retrieval', 'index + Boolean', true)}
      {box(xs[3], 'Cognitive evidence', 'preview only', false)}
      {box(xs[4], 'Learner state', 'research phase', false)}
    </svg>
  )
}

export default function Home() {
  const meta = useMeta()
  const tok = useResult('tokenization')
  const ner = useResult('ner')
  const pos = useResult('pos')
  const pl = useResult('pipelines')
  const scale = useResult('scale')
  if (!meta.data || !tok.data || !ner.data || !pos.data || !pl.data) return <Loading />
  const t = meta.data.tiles
  const sc = (n: string) => tok.data.scores.find((s: { tokenizer: string }) => s.tokenizer === n)
  const acc = (n: string) => pos.data.accuracy.find((s: { tagger: string }) => s.tagger === n)
  const win = pl.data.table_h.find((r: { pipeline: string }) => r.pipeline === pl.data.winner)
  const heavyDocs = scale.data ? scale.data.summary.reduce((a: number, s: { documents: number }) => a + s.documents, 0) : null
  const findings = [
    { to: '/tokenization', tag: 'Exercise 3', title: `Hybrid tokenizer F1 ${sc('hybrid').f1.toFixed(3)} vs NLTK ${sc('nltk').f1.toFixed(3)}`,
      body: 'Scored on 40 hand-segmented sentences. The domain rules fix fractions, mixed numbers, currency and hyphenated terms.' },
    { to: '/ner', tag: 'Exercise 13', title: `NER F1 ${ner.data.model.overall.f1.toFixed(2)} → ${ner.data.ruler.overall.f1.toFixed(2)} with the rule layer`,
      body: 'spaCy alone calls BKT an ORG and finds none of the 13 money amounts once "$3,650" is a single token.' },
    { to: '/custom-pos', tag: 'Exercise 12', title: `Rule tagger ${(acc('rules_nltk').accuracy_test * 100).toFixed(1)}% vs NLTK ${(acc('nltk').accuracy_test * 100).toFixed(1)}%`,
      body: 'NLTK tags the instruction "Simplify" as a noun most of the time; the dictionary and rule layer corrects it.' },
    { to: '/pipelines', tag: 'Module 3', title: `Selected pipeline: ${pl.data.winner} (F1 ${win.f1.toFixed(3)})`,
      body: 'Chosen by a rule fixed before the runs. The hybrid design H lost on one query; the post-hoc H2 is reported separately.' },
  ]
  return (
    <div>
      <PageHeader eyebrow="Vidyashilp University · NLP Assessment-1" title="Learner language about school mathematics, analysed and searchable"
        lead={<>A domain-specific text analysis and retrieval system over <b>30 documents in 10 formats</b> and two heavy datasets
          ({heavyDocs ? fmt(heavyDocs) : '3,427'} tutoring dialogues and classroom transcripts). Every step follows the brief's loop:
          select, order, implement, compare, evaluate, justify.</>} />
      <Grid cols={5}>
        <Stat label="Documents" value={t.documents} note="D01–D30, 10 formats" />
        <Stat label="Tokens" value={compact(t.tokens)} note="NLTK tokens after cleaning" />
        <Stat label="Vocabulary" value={fmt(t.vocabulary)} note="lower-cased word types" />
        <Stat label="Heavy datasets" value={heavyDocs ? fmt(heavyDocs) : '—'} note="MathDial + TalkMoves" />
        <Stat label="Best mean F1" value={t.best_f1.toFixed(3)} note={`pipeline ${t.best_pipeline}, 15 queries`} />
      </Grid>

      <Card className="mt-6" title="Where this assessment sits in the research" subtitle="Language-grounded learner state modelling for misconception-aware adaptive learning">
        <LoopDiagram />
        <p className="mt-3 text-sm leading-relaxed text-ink-2">
          This assessment builds the text front end: loading, cleaning, tokenization, linguistic analysis and retrieval that grounds a
          learner's words in concept and misconception documents. It builds no learner model and no probabilities; the Evidence preview
          page only shows what a first evidence record could look like.
        </p>
      </Card>

      <h2 className="mt-10 mb-4 text-lg font-semibold">Four findings worth defending in the viva</h2>
      <Grid cols={2}>
        {findings.map((f) => (
          <Link key={f.title} to={f.to} className="group rounded-2xl border border-line bg-surface p-5 shadow-[var(--shadow)] transition hover:border-accent/40">
            <Pill tone="accent">{f.tag}</Pill>
            <div className="mt-3 text-[16px] font-semibold text-ink">{f.title}</div>
            <p className="mt-1.5 text-sm leading-relaxed text-ink-2">{f.body}</p>
            <div className="mt-3 inline-flex items-center gap-1 text-xs font-medium text-accent-ink">Open <ArrowRight size={13} className="transition group-hover:translate-x-0.5" /></div>
          </Link>
        ))}
      </Grid>

      <h2 className="mt-10 mb-4 text-lg font-semibold">Every requirement, one page each</h2>
      <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
        {NAV.slice(1).map((g) => (
          <Card key={g.group} title={g.group}>
            <ul className="space-y-1.5">
              {g.items.map((it) => (
                <li key={it.to}>
                  <Link to={it.to} className="flex items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-ink-2 hover:bg-surface-2 hover:text-ink">
                    <it.icon size={15} className="text-ink-3" /> {it.label}
                    {it.tag && <span className="ml-auto text-[11px] text-ink-3">{it.tag}</span>}
                  </Link>
                </li>
              ))}
            </ul>
          </Card>
        ))}
      </div>
    </div>
  )
}
