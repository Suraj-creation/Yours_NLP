import { useState } from 'react'
import { Play, Trophy } from 'lucide-react'
import { usePost, useResult, type Any } from '../lib/api'
import { fmt, runColor } from '../lib/palette'
import Chart from '../components/Chart'
import { bars, lines } from '../components/charts'
import { Button, Callout, Card, DataTable, DownloadLink, ErrorBox, Grid, Loading, PageHeader, Pill, Section, Select, Why } from '../components/ui'
import { Metrics } from '../components/nlp'

const STAGE_LABEL: Record<string, string> = { tokenized: 'Tokenized', lowercased: 'Lower-cased', stopwords_removed: 'Stop words removed', normalized: 'Normalised', index_terms: 'Index terms' }
const RunDot = ({ run }: { run: string }) => <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: runColor(run) }} />

export default function Pipelines() {
  const r = useResult('pipelines')
  const design = usePost('/api/pipelines/run')
  const [cfg, setCfg] = useState({ tokenizer: 'hybrid', stopwords: 'custom', normalizer: 'wordnet_cf', order: 'norm_then_stop', compound_parts: false })
  if (!r.data) return <Loading />
  const p = r.data
  const th: Any[] = p.table_h
  const runs = th.map((x) => x.pipeline as string)
  const win = th.find((x) => x.pipeline === p.winner)
  const qids: string[] = p.queries.map((q: Any) => q.qid)
  const f1 = (run: string, qid: string) => p.evaluation.find((e: Any) => e.pipeline === run && e.qid === qid)?.f1 ?? 0
  const diffs = qids.map((q) => ({ qid: q, query: p.queries.find((x: Any) => x.qid === q).query, B: f1('B', q), H: f1('H', q), H2: f1('H2', q), A: f1('A', q) }))
    .filter((x) => Math.abs(x.H - x.B) > 1e-9 || Math.abs(x.H2 - x.B) > 1e-9 || Math.abs(x.A - x.B) > 1e-9)
  const set = (k: string) => (v: string) => setCfg({ ...cfg, [k]: k === 'compound_parts' ? v === 'yes' : v })

  return (
    <div>
      <PageHeader eyebrow="Module 3 · Pipeline design and comparison" title="Seven pipelines, one selection rule"
        lead={<>A and B are the brief's two branches. H is the hybrid design proposed before any results were seen. C1–C3 each change one thing to
          test a claim (order, POS for lemmas, BPE terms). H2 was designed after seeing H's errors, so it is reported but <b>not eligible</b> for selection.</>} />

      <Card className="border-accent/30" title={<span className="flex items-center gap-2"><Trophy size={16} className="text-accent-ink" /> Selected: pipeline {p.winner}</span>}
        subtitle={`Rule, fixed before the runs: ${p.selection_rule}.`}>
        <Metrics items={[
          { label: 'Mean F1', value: win.f1, digits: 4 }, { label: 'Precision', value: win.precision }, { label: 'Recall', value: win.recall },
          { label: 'P@5', value: win['p@5'] }, { label: 'Domain-term fidelity', value: win.domain_term_fidelity }, { label: 'Vocabulary', value: win.vocabulary_size },
        ]} />
        <p className="mt-3 text-sm leading-relaxed text-ink-2">{win.description}.</p>
      </Card>

      <Section title="Table H: pipeline comparison">
        <Card pad={false} title="pipeline_comparison.csv" actions={<><DownloadLink file="pipeline_comparison.csv" /><DownloadLink file="table_h_brief_format.csv" /></>}>
          <DataTable rows={th} dense initialSort={{ key: 'f1', dir: -1 }} cols={[
            { key: 'pipeline', label: 'Pipeline', render: (x) => <span className="flex items-center gap-2"><RunDot run={x.pipeline as string} /><b className="text-ink">{x.pipeline as string}</b>{x.selected ? <Pill tone="accent">selected</Pill> : null}{x.posthoc ? <Pill tone="warn">post-hoc</Pill> : null}</span> },
            { key: 'tokenizer', label: 'Tokenizer' }, { key: 'stopwords', label: 'Stop list' }, { key: 'normalizer', label: 'Normaliser' },
            { key: 'order', label: 'Order', render: (x) => (x.order === 'stop_then_norm' ? 'stop → norm' : 'norm → stop') },
            { key: 'token_count', label: 'Index tokens', num: true }, { key: 'vocabulary_size', label: 'Vocabulary', num: true },
            { key: 'domain_term_fidelity', label: 'Fidelity', num: true }, { key: 'meaningful_ngrams_share', label: 'Meaningful n-grams', num: true, digits: 2 },
            { key: 'precision', label: 'P', num: true }, { key: 'recall', label: 'R', num: true }, { key: 'f1', label: 'F1', num: true, digits: 4 },
            { key: 'p@5', label: 'P@5', num: true }, { key: 'mean_query_time_ms', label: 'ms/query', num: true, digits: 2 },
          ]} />
        </Card>
        <Grid cols={2}>
          <Card title="Mean F1 over 15 queries">
            <Chart height={300} label="Mean F1 per pipeline" option={() => bars({
              horizontal: true, categories: runs, labels: true, digits: 3, max: 0.8,
              series: [{ name: 'mean F1', data: th.map((x) => x.f1), color: runColor('A') }], itemColors: runs.map(runColor),
            })} />
          </Card>
          <Card title="Vocabulary size" subtitle="index terms after every step">
            <Chart height={300} label="Vocabulary per pipeline" option={() => bars({
              horizontal: true, categories: runs, labels: true,
              series: [{ name: 'vocabulary', data: th.map((x) => x.vocabulary_size), color: runColor('A') }], itemColors: runs.map(runColor),
            })} />
          </Card>
        </Grid>
      </Section>

      <Section title="Where the pipelines disagree" lead={`The queries where A, B, H and H2 disagree. The other ${qids.length - diffs.length} score the same in all four; the Evaluation page has every query and pipeline.`}>
        <Card title="F1 per query">
          <Chart height={320} label="F1 per disputed query" option={() => bars({
            categories: diffs.map((x) => x.qid), max: 1, digits: 2,
            series: (['A', 'B', 'H', 'H2'] as const).map((run) => ({ name: run, data: diffs.map((x) => x[run]), color: runColor(run) })),
          })} />
        </Card>
        <DataTable rows={diffs} dense cols={[
          { key: 'qid', label: 'Query' }, { key: 'query', label: 'Text', render: (x) => <span className="font-mono text-[12px]">{x.query as string}</span> },
          ...(['A', 'B', 'H', 'H2'] as const).map((k) => ({ key: k, label: k, num: true, digits: 3 })),
        ]} />
        <Callout>
          <b>Why H lost.</b> H differs from B on one query only: Q15 (LLM AND "knowledge tracing"), F1 0 against 0.667. spaCy's lemmas depend on
          context: in D10 "Knowledge Tracing" is tagged as a proper noun and keeps the lemma "tracing", while the query is lemmatized to "trace", so
          the phrase is not found there; and "LLM-based" is a single hybrid token, so "llm" is not indexed on its own. The same mismatch swaps D10
          for D11 on Q10 (F1 unchanged by coincidence). H2 uses context-free lemmas and indexes compound parts, finds D10 again, but also retrieves
          D09 and D11 (F1 0.5). It was designed after seeing these errors, so it is not eligible and the rule keeps B.
        </Callout>
      </Section>

      <Section title="Vocabulary through the stages">
        <Card>
          <Chart height={320} label="Vocabulary per stage per pipeline" option={() => lines({
            xs: Object.values(STAGE_LABEL), markers: true,
            series: runs.map((run) => ({ name: run, color: runColor(run), data: Object.keys(STAGE_LABEL).map((st) => p.stages.find((x: Any) => x.pipeline === run && x.stage === st)?.vocabulary ?? null) })),
          })} />
        </Card>
        <DownloadLink file="pipeline_stages.csv" />
      </Section>

      <Section title="Design your own pipeline" lead="Runs the whole 30-document corpus through your choices, indexes it and scores the 15 gold queries. Uncached combinations take up to a minute.">
        <Card>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-6">
            <Select label="Tokenizer" value={cfg.tokenizer} onChange={set('tokenizer')} options={['nltk', 'spacy', 'custom', 'hybrid', 'tiktoken_cl100k'].map((v) => ({ value: v, label: v }))} />
            <Select label="Stop list" value={cfg.stopwords} onChange={set('stopwords')} options={['none', 'nltk', 'spacy', 'custom'].map((v) => ({ value: v, label: v }))} />
            <Select label="Normaliser" value={cfg.normalizer} onChange={set('normalizer')} options={['none', 'porter', 'snowball', 'lancaster', 'regexp', 'wordnet_nopos', 'wordnet_pos', 'wordnet_cf', 'spacy'].map((v) => ({ value: v, label: v }))} />
            <Select label="Order" value={cfg.order} onChange={set('order')} options={[{ value: 'stop_then_norm', label: 'stop → norm' }, { value: 'norm_then_stop', label: 'norm → stop' }]} />
            <Select label="Compound parts" value={cfg.compound_parts ? 'yes' : 'no'} onChange={set('compound_parts')} options={[{ value: 'no', label: 'no' }, { value: 'yes', label: 'yes' }]} />
            <div className="flex items-end"><Button onClick={() => design.mutate(cfg)} disabled={design.isPending} className="w-full"><Play size={15} /> {design.isPending ? 'Running…' : 'Run'}</Button></div>
          </div>
          {design.isPending && <Loading label="Tokenizing, normalising, indexing and scoring" />}
          {design.error && <div className="mt-4"><ErrorBox error={design.error} /></div>}
          {design.data && !design.isPending && (
            <div className="mt-5 space-y-4">
              <Metrics items={[
                { label: 'Mean F1', value: design.data.mean.f1, digits: 4, note: `${design.data.mean.f1 >= win.f1 ? '+' : ''}${(design.data.mean.f1 - win.f1).toFixed(4)} vs ${p.winner}` },
                { label: 'Precision', value: design.data.mean.precision }, { label: 'Recall', value: design.data.mean.recall }, { label: 'P@5', value: design.data.mean['p@5'] },
                { label: 'Vocabulary', value: design.data.summary.vocabulary_size ?? design.data.summary.vocabulary }, { label: 'Seconds', value: design.data.seconds, digits: 1 },
              ]} />
              <DataTable rows={design.data.per_query.map((x: Any) => ({ ...x, winner_f1: f1(p.winner, x.qid) }))} dense cols={[
                { key: 'qid', label: 'Query' }, { key: 'query', label: 'Text', render: (x) => <span className="font-mono text-[12px]">{x.query as string}</span> },
                { key: 'precision', label: 'P', num: true }, { key: 'recall', label: 'R', num: true }, { key: 'f1', label: 'F1', num: true },
                { key: 'winner_f1', label: `F1 of ${p.winner}`, num: true },
                { key: 'delta', label: 'Δ', num: true, render: (x) => { const dd = (x.f1 as number) - (x.winner_f1 as number); return <span className={dd > 0 ? 'text-good-ink' : dd < 0 ? 'text-bad' : 'text-ink-3'}>{dd === 0 ? '0' : dd.toFixed(3)}</span> } },
              ]} />
              <p className="text-xs text-ink-3">Stages: {design.data.stages.map((s: Any) => `${STAGE_LABEL[s.stage] ?? s.stage} ${fmt(s.vocabulary)}`).join(' → ')}</p>
            </div>
          )}
        </Card>
        <Why why="The brief asks for at least two pipelines built from different choices at each stage, compared on the same queries, with the final one justified."
          where="Each pipeline is a full path: cleaned text → tokens → lower case → stop words → stems or lemmas → index terms → index."
          depends="The same cleaned corpus, the same 15 queries and the same qrels for every pipeline, and a selection rule written before the runs."
          moved="Changing one stage at a time (C1: order, C2: POS for lemmas, C3: BPE terms) isolates its effect; see the Justification page for each decision." />
      </Section>
    </div>
  )
}
