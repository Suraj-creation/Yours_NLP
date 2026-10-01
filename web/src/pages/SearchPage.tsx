import { useEffect, useState } from 'react'
import { Search as SearchIcon } from 'lucide-react'
import clsx from 'clsx'
import { usePost, useResult, type Any } from '../lib/api'
import { fmt, runColor } from '../lib/palette'
import { usePipeline } from '../components/Layout'
import { Button, Callout, Card, ErrorBox, Grid, Loading, PageHeader, Pill, Section, Segmented } from '../components/ui'
import { Metrics, Snippet } from '../components/nlp'

const SYNTAX = [
  ['denominator', 'Keyword', 'one term, ranked by tf-idf'],
  ['common denominator', 'Phrase', 'adjacent words in order (positions)'],
  ['fraction AND misconception', 'Boolean AND', 'both terms'],
  ['LCD OR "least common denominator"', 'Boolean OR', 'either; quotes make a phrase inside Boolean'],
  ['denominator AND NOT common', 'Boolean NOT', 'NOT is the complement within the collection'],
  ['dialogue AND (tutor OR teacher)', 'Advanced', 'brackets group; operators must be upper case'],
]

export default function SearchPage() {
  const { pipeline } = usePipeline()
  const pl = useResult('pipelines')
  const sc = useResult('scale')
  const run = usePost('/api/search')
  const [q, setQ] = useState('fraction AND misconception')
  const [mode, setMode] = useState('auto')
  const [coll, setColl] = useState('assessment')
  const go = (query = q, m = mode, c = coll) => { if (query.trim()) run.mutate({ query, mode: m, collection: c, pipeline, limit: 30 }) }
  useEffect(() => { go() }, [pipeline]) // eslint-disable-line react-hooks/exhaustive-deps
  const d = run.data
  const ev = d?.evaluation
  const examples: string[] = coll === 'assessment' ? (pl.data?.queries ?? []).map((x: Any) => x.query) : (sc.data?.query_list ?? [])
  const rel = new Set<string>(ev?.relevant_docs ?? [])

  return (
    <div>
      <PageHeader eyebrow="Modules 4 and 5 · Query processing and search" title="Search the corpus"
        lead="Keyword, phrase and Boolean queries over the positional index. The query goes through the same analyser as the documents; results are ranked by tf-idf. For the 15 gold queries the page also scores the result list against the relevance judgements." />

      <Card>
        <form className="flex flex-col gap-3 sm:flex-row" onSubmit={(e) => { e.preventDefault(); go() }}>
          <div className="relative flex-1">
            <SearchIcon size={18} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-ink-3" />
            <input value={q} onChange={(e) => setQ(e.target.value)} aria-label="Query" placeholder='e.g. fraction AND misconception'
              className="w-full rounded-xl border border-line-strong bg-surface py-3 pl-11 pr-4 font-mono text-[15px] text-ink outline-none focus:border-accent" />
          </div>
          <Button type="submit" className="px-6">Search</Button>
        </form>
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <Segmented value={coll} onChange={(v) => { setColl(v); go(q, mode, v) }} options={[{ value: 'assessment', label: '30 documents' }, { value: 'mathdial', label: 'MathDial · 2,861' }, { value: 'talkmoves', label: 'TalkMoves · 566' }]} />
          <Segmented value={mode} onChange={(v) => { setMode(v); go(q, v) }} options={[{ value: 'auto', label: 'Auto' }, { value: 'keyword', label: 'Keywords (AND)' }, { value: 'phrase', label: 'Phrase' }]} />
          <span className="inline-flex items-center gap-1.5 text-xs text-ink-3"><span className="h-2 w-2 rounded-full" style={{ background: runColor(d?.pipeline ?? pipeline) }} />pipeline {d?.pipeline ?? pipeline}{coll !== 'assessment' && !['A', 'B', 'H2'].includes(pipeline) ? ' (heavy datasets are indexed with A, B and H2 only)' : ''}</span>
        </div>
        <div className="mt-3 flex flex-wrap gap-1.5">
          {examples.map((s) => (
            <button key={s} onClick={() => { setQ(s); go(s) }}
              className={clsx('rounded-full border px-2.5 py-1 font-mono text-xs', s === q ? 'border-accent/40 bg-accent-soft text-accent-ink' : 'border-line bg-surface-2 text-ink-2 hover:text-ink')}>{s}</button>
          ))}
        </div>
      </Card>

      {run.isPending && <Loading label="Searching" />}
      {run.error && <div className="mt-5"><ErrorBox error={run.error} /></div>}
      {d && !run.isPending && (
        <div className="mt-6 space-y-5">
          <div className="flex flex-wrap items-center gap-2 text-sm text-ink-2">
            <Pill tone="accent">{d.type}</Pill>
            <span><b className="num text-ink">{fmt(d.count)}</b> documents</span>
            <span className="text-ink-3">· {d.time_ms.toFixed(3)} ms (median of 5)</span>
            <span className="ml-2 flex flex-wrap items-center gap-1.5 text-xs text-ink-3">analysed as
              {d.analysed.map((a: Any, i: number) => (
                <span key={i} className={clsx('rounded-md border px-1.5 py-0.5 font-mono', a.negated ? 'border-bad/30 bg-bad/10 text-bad line-through' : 'border-line bg-surface-2 text-ink')}>
                  {a.terms.join(' ')}
                </span>
              ))}
            </span>
          </div>

          {ev && (
            <Card title={`Gold query ${ev.qid}`} subtitle={ev.information_need}>
              <Metrics items={[
                { label: 'Precision', value: ev.precision }, { label: 'Recall', value: ev.recall }, { label: 'F1', value: ev.f1 },
                { label: 'P@5', value: ev['p@5'] }, { label: 'R@10', value: ev['r@10'] }, { label: 'Average precision', value: ev.ap },
              ]} />
              <div className="mt-3 flex flex-wrap items-center gap-1.5 text-xs text-ink-3">
                Relevant ({ev.relevant_docs.length}):
                {ev.relevant_docs.map((r: string) => <Pill key={r} tone={d.doc_ids.includes(r) ? 'good' : 'bad'}>{r} {d.doc_ids.includes(r) ? 'found' : 'missed'}</Pill>)}
              </div>
            </Card>
          )}

          {d.results.length === 0 && <Callout>No document matches. Operators must be upper case (AND, OR, NOT); lower-case "and" is treated as a word.</Callout>}
          <div className="space-y-3">
            {d.results.map((x: Any, i: number) => (
              <article key={x.doc_id} className="rounded-2xl border border-line bg-surface p-4 shadow-[var(--shadow)]">
                <div className="flex flex-wrap items-baseline gap-2">
                  <span className="num w-6 text-sm text-ink-3">{i + 1}</span>
                  <span className="font-mono text-sm font-semibold text-accent-ink">{x.doc_id}</span>
                  <h3 className="flex-1 text-[15px] font-medium text-ink">{x.title}</h3>
                  {ev && <Pill tone={rel.has(x.doc_id) ? 'good' : 'neutral'}>{rel.has(x.doc_id) ? 'relevant' : 'not judged relevant'}</Pill>}
                  <span className="num text-xs text-ink-3">tf-idf {x.score.toFixed(3)}</span>
                </div>
                <div className="mt-2 pl-8"><Snippet parts={x.snippet} /></div>
              </article>
            ))}
          </div>
          {d.count > d.results.length && <p className="text-xs text-ink-3">Showing the top {d.results.length} of {fmt(d.count)}.</p>}
        </div>
      )}

      <Section title="Query syntax">
        <Grid cols={2}>
          <Card>
            <table className="w-full text-left text-sm">
              <tbody>
                {SYNTAX.map(([ex, kind, what]) => (
                  <tr key={ex} className="border-b border-line/70 last:border-0">
                    <td className="py-2 pr-3"><button className="text-left font-mono text-[12.5px] text-accent-ink hover:underline" onClick={() => { setQ(ex); setColl('assessment'); go(ex, mode, 'assessment') }}>{ex}</button></td>
                    <td className="py-2 pr-3 text-ink">{kind}</td><td className="py-2 text-ink-3">{what}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
          <Card title="How a query is processed">
            <ol className="list-decimal space-y-1.5 pl-5 text-sm leading-relaxed text-ink-2">
              <li>Lex: upper-case AND, OR, NOT and brackets are operators; runs of words and quoted strings are units; adjacent units get an implicit AND.</li>
              <li>Each unit is analysed by the pipeline's tokenizer, stop list and normaliser, exactly as documents were.</li>
              <li>A one-term unit looks up its postings; a multi-term unit is a phrase and needs consecutive positions.</li>
              <li>Shunting-yard turns the expression into postfix; set operations evaluate it. NOT A is every document without A.</li>
              <li>Matches are ranked by the summed tf-idf of the non-negated terms; ties go to the lower document id.</li>
            </ol>
          </Card>
        </Grid>
      </Section>
    </div>
  )
}
