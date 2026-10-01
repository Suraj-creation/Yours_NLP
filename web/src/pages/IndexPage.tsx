import { useState } from 'react'
import { Search } from 'lucide-react'
import { useGet, useMeta, type Any } from '../lib/api'
import { fmt, runColor } from '../lib/palette'
import Chart from '../components/Chart'
import { bars, heatmap } from '../components/charts'
import { usePipeline } from '../components/Layout'
import { Button, Card, DataTable, DownloadLink, ErrorBox, Examples, Grid, Loading, PageHeader, Pill, Section, Segmented, Stat, Why } from '../components/ui'
import { Mono } from '../components/nlp'

const TERMS = ['denominators', 'misconception', 'tracing', 'added', 'LLM', 'halves', 'teacher', 'percent']

export default function IndexPage() {
  const { pipeline } = usePipeline()
  const meta = useMeta()
  const W = meta.data?.winner ?? 'B'
  const [norm, setNorm] = useState(true)
  const sum = useGet(`/api/index/${pipeline}/summary`)
  const [q, setQ] = useState('denominators')
  const [term, setTerm] = useState('denominators')
  const t = useGet(term ? `/api/index/${pipeline}/term?q=${encodeURIComponent(term)}` : null)

  return (
    <div>
      <PageHeader eyebrow="Module 4 · Positional inverted index" title="The inverted index"
        lead={<>Every index term maps to the documents that contain it and the positions where it occurs. Positions make phrase queries possible:
          "common denominator" matches only where <i>denominator</i> sits one position after <i>common</i>. Sentence ends leave a gap of one
          position so phrases never match across sentences. The index shown is built by pipeline <b>{pipeline}</b>; change it in the header.</>} />

      {sum.isLoading && <Loading label={`Building or loading the index for pipeline ${pipeline}`} />}
      {sum.error && <ErrorBox error={sum.error} />}
      {sum.data && (
        <>
          <Grid cols={4}>
            <Stat label="Index terms" value={fmt(sum.data.vocabulary)} note={<span className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: runColor(pipeline) }} />pipeline {pipeline}</span>} />
            <Stat label="Documents" value={sum.data.documents} />
            <Stat label="Postings" value={fmt(sum.data.postings)} note="(term, document) pairs" />
            <Stat label="Positions" value={fmt(sum.data.positions)} note="term occurrences stored" />
          </Grid>

          <Section title="Look up a term" lead="The term goes through the same analyser as the documents, so 'denominators' finds the lemma or stem the pipeline stored.">
            <Card>
              <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); setTerm(q.trim()) }}>
                <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="a word" aria-label="Term"
                  className="min-w-0 flex-1 rounded-xl border border-line-strong bg-surface px-4 py-2 text-[15px] text-ink outline-none focus:border-accent" />
                <Button type="submit"><Search size={15} /> Look up</Button>
              </form>
              <Examples items={TERMS} onPick={(s) => { setQ(s); setTerm(s) }} />
              {t.error && <div className="mt-4"><ErrorBox error={t.error} /></div>}
              {t.data && (
                <div className="mt-5 space-y-4">
                  <div className="flex flex-wrap items-center gap-2 text-sm text-ink-2">
                    <Mono>{t.data.input}</Mono> → index term <Mono>{t.data.term}</Mono>
                    {t.data.analysed.length > 1 && <span className="text-ink-3">(analysed as {t.data.analysed.join(' + ')}; the first part is looked up)</span>}
                  </div>
                  <div className="grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-line bg-line sm:grid-cols-4">
                    {[['Document frequency', `${t.data.df} of ${t.data.N}`], ['Collection frequency', fmt(t.data.cf)], ['IDF (log10 N/df)', t.data.idf.toFixed(3)], ['Surface forms', t.data.surface_forms.length]].map(([k, v]) => (
                      <div key={k as string} className="bg-surface px-4 py-3"><div className="text-[11px] uppercase tracking-wide text-ink-3">{k}</div><div className="num text-lg font-semibold text-ink">{v}</div></div>
                    ))}
                  </div>
                  {t.data.surface_forms.length > 0 && (
                    <div className="flex flex-wrap items-center gap-1.5 text-xs text-ink-3">Merged forms:
                      {t.data.surface_forms.map(([s, c]: [string, number]) => <Pill key={s}><span className="font-mono">{s}</span> {fmt(c)}</Pill>)}
                    </div>
                  )}
                  {t.data.df === 0 ? <p className="text-sm text-ink-3">Not in the index.</p> : (
                    <DataTable rows={t.data.postings} dense cols={[
                      { key: 'doc_id', label: 'Document' }, { key: 'tf', label: 'tf', num: true }, { key: 'tfidf', label: 'tf-idf', num: true, digits: 4 },
                      { key: 'positions', label: 'Positions (first 50)', render: (x) => <span className="font-mono text-[11.5px] text-ink-3">{(x.positions as number[]).join(', ')}</span> },
                    ]} />
                  )}
                </div>
              )}
            </Card>
          </Section>

          <Section title="How terms spread over documents">
            <Grid cols={2}>
              <Card title="Terms by document frequency" subtitle="log scale; most terms appear in one document">
                <Chart height={300} label="Document frequency histogram" option={() => bars({
                  categories: sum.data.df_histogram.map((x: Any) => String(x.df)), logValue: true, valueName: 'terms',
                  series: [{ name: 'terms', data: sum.data.df_histogram.map((x: Any) => x.terms), color: runColor(pipeline) }],
                })} />
              </Card>
              <Card title="Share of terms in exactly one document">
                <div className="flex h-[300px] flex-col justify-center">
                  <div className="num text-5xl font-semibold text-ink">{((sum.data.df_histogram[0]?.terms / sum.data.vocabulary) * 100).toFixed(0)}%</div>
                  <p className="mt-3 max-w-sm text-sm leading-relaxed text-ink-2">
                    {fmt(sum.data.df_histogram[0]?.terms)} of {fmt(sum.data.vocabulary)} terms occur in a single document. They get the highest IDF
                    ({Math.log10(sum.data.documents).toFixed(2)}), so a query word that appears in one document ranks it first.
                  </p>
                </div>
              </Card>
            </Grid>
            <Card title="Term frequency by document" subtitle="30 most frequent content terms × 30 documents; stronger colour means more occurrences"
              actions={<Segmented value={norm ? 'rel' : 'raw'} onChange={(v) => setNorm(v === 'rel')} options={[{ value: 'rel', label: 'Per 1,000 terms' }, { value: 'raw', label: 'Raw count' }]} />}>
              <Chart height={640} label="Term by document heatmap" option={() => {
                const cells: [number, number, number][] = sum.data.heatmap.cells.map((c: number[]) => {
                  const len = sum.data.doc_length?.[sum.data.heatmap.docs[c[1]]] ?? 1
                  return [c[1], c[0], norm ? (1000 * c[2]) / Math.max(1, len) : c[2]]
                })
                const vals = cells.map((c) => c[2]).sort((a, b) => a - b)
                return heatmap({ xLabels: sum.data.heatmap.docs, yLabels: sum.data.heatmap.terms, xRotate: 45, visual: true, data: cells,
                  digits: norm ? 1 : 0, max: vals[Math.floor(vals.length * 0.98)] || 1 })
              }} />
              <p className="mt-2 text-xs text-ink-3">{norm ? 'Occurrences per 1,000 index terms of the document, so long files (D16–D18) do not dominate.' : 'Raw counts: the three longest files dominate.'} Colour scale capped at the 98th percentile.</p>
            </Card>
          </Section>

          <Section title="Structure on disk">
            <Card actions={<DownloadLink file="inverted_index.json" />}>
              <pre className="overflow-x-auto rounded-xl bg-surface-2 p-4 font-mono text-[12px] leading-relaxed text-ink-2">{`{
  "documents": ["D01", "D02", …, "D30"],
  "doc_length": { "D01": ${sum.data.doc_length?.D01 ?? '…'}, … },
  "vocabulary_size": ${sum.data.vocabulary},
  "terms": {
    "${t.data?.term ?? 'denominator'}": {
      "df": ${t.data?.df ?? '…'}, "cf": ${t.data?.cf ?? '…'},
      "postings": { ${(t.data?.postings ?? []).slice(0, 2).map((p: Any) => `"${p.doc_id}": [${p.positions.slice(0, 5).join(', ')}, …]`).join(', ')}, … }
    },
    …
  }
}`}</pre>
              <p className="mt-3 text-xs text-ink-3">results/inverted_index.json holds the index of the selected pipeline ({W}); the values above are for pipeline {pipeline}. Ranking uses tf-idf = (1 + log10 tf) × log10(N / df), summed over the query's non-negated terms.</p>
            </Card>
            <Why why="Boolean and phrase search need to find documents by term in constant time; positions are what separates a phrase query from an AND query."
              where="After every normalisation step: the index stores exactly the analysed terms, and queries pass through the same analyser."
              depends="The whole pipeline: a different tokenizer, stop list or lemmatizer gives a different index (compare pipelines in the header)."
              moved="Indexing before normalisation would need the query to guess every surface form; normalising queries differently from documents silently drops matches (the H pipeline's 'tracing' vs 'trace' problem)." />
          </Section>
        </>
      )}
    </div>
  )
}
