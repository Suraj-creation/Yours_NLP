import { useResult, type Any } from '../lib/api'
import { runColor } from '../lib/palette'
import Chart from '../components/Chart'
import { heatmap, lines } from '../components/charts'
import { usePipeline } from '../components/Layout'
import { Callout, Card, DataTable, DownloadLink, Grid, Loading, PageHeader, Pill, Section, Why } from '../components/ui'

export default function Evaluation() {
  const r = useResult('pipelines')
  const { pipeline } = usePipeline()
  if (!r.data) return <Loading />
  const p = r.data
  const runs: string[] = p.overall.map((x: Any) => x.pipeline)
  const queries: Any[] = p.queries
  const f1 = (run: string, qid: string) => p.evaluation.find((e: Any) => e.pipeline === run && e.qid === qid)?.f1 ?? 0
  const perQ: Any[] = p.evaluation.filter((e: Any) => e.pipeline === pipeline)
  const retr: Any[] = p.retrieval.filter((e: Any) => e.pipeline === pipeline)
  const ks = p.curves[runs[0]].map((x: Any) => x.k)

  return (
    <div>
      <PageHeader eyebrow="Module 6 · Evaluation" title="Precision, recall and ranking quality"
        lead={<>Fifteen queries, each with a written information need, judged against all 30 documents (450 judgements, {Object.values(p.qrels as Record<string, string[]>).reduce((a, v) => a + v.length, 0)} relevant).
          Set metrics (P, R, F1) score the whole result list; P@K, R@K and average precision score the ranking.</>} />

      <Section title="Table J: overall performance">
        <Card pad={false} title="overall_performance.csv" actions={<><DownloadLink file="overall_performance.csv" /><DownloadLink file="table_j_brief_format.csv" /></>}>
          <DataTable rows={p.overall} dense initialSort={{ key: 'f1', dir: -1 }} cols={[
            { key: 'pipeline', label: 'Pipeline', render: (x) => <span className="flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full" style={{ background: runColor(x.pipeline as string) }} /><b className="text-ink">{x.pipeline as string}</b>{x.pipeline === p.winner && <Pill tone="accent">selected</Pill>}{x.pipeline === 'H2' && <Pill tone="warn">post-hoc</Pill>}</span> },
            { key: 'precision', label: 'P', num: true }, { key: 'recall', label: 'R', num: true }, { key: 'f1', label: 'F1', num: true, digits: 4 },
            { key: 'p@3', label: 'P@3', num: true }, { key: 'p@5', label: 'P@5', num: true }, { key: 'p@10', label: 'P@10', num: true },
            { key: 'r@5', label: 'R@5', num: true }, { key: 'r@10', label: 'R@10', num: true }, { key: 'ap', label: 'MAP', num: true },
            { key: 'mean_query_time_ms', label: 'ms/query', num: true, digits: 3 }, { key: 'index_seconds', label: 'Index s', num: true, digits: 1 },
          ]} />
        </Card>
      </Section>

      <Section title="Ranking curves" lead="Mean over the 15 queries at each cut-off K.">
        <Grid cols={2}>
          <Card title="Precision at K">
            <Chart height={300} label="Precision at K" option={() => lines({ xs: ks, xName: 'K', yPercent: true, markers: true,
              series: runs.map((run) => ({ name: run, color: runColor(run), data: p.curves[run].map((x: Any) => x.p) })) })} />
          </Card>
          <Card title="Recall at K">
            <Chart height={300} label="Recall at K" option={() => lines({ xs: ks, xName: 'K', yPercent: true, markers: true,
              series: runs.map((run) => ({ name: run, color: runColor(run), data: p.curves[run].map((x: Any) => x.r) })) })} />
          </Card>
        </Grid>
      </Section>

      <Section title="F1 by query and pipeline" lead="Most queries score the same everywhere; the few that differ decide the ranking of pipelines.">
        <Card>
          <Chart height={520} label="F1 heatmap" option={() => heatmap({
            xLabels: runs, yLabels: queries.map((q) => `${q.qid} ${q.query}`), max: 1, digits: 2, showValues: true, visual: true, showZero: true,
            data: queries.flatMap((q, y) => runs.map((run, x) => [x, y, f1(run, q.qid)] as [number, number, number])),
          })} />
        </Card>
      </Section>

      <Section title={`Table I: per-query results for pipeline ${pipeline}`} lead="Change the pipeline in the header to compare.">
        <Card pad={false} title="evaluation_results.csv" actions={<><DownloadLink file="evaluation_results.csv" /><DownloadLink file="retrieval_results.csv" /></>}>
          <DataTable rows={perQ.map((e) => ({ ...e, type: retr.find((x) => x.qid === e.qid)?.type, analysed: retr.find((x) => x.qid === e.qid)?.analysed_terms }))} dense cols={[
            { key: 'qid', label: 'Query' }, { key: 'query', label: 'Text', render: (x) => <span className="font-mono text-[12px]">{x.query as string}</span> },
            { key: 'type', label: 'Type' }, { key: 'analysed', label: 'Analysed', render: (x) => <span className="font-mono text-[11.5px] text-ink-3">{x.analysed as string}</span> },
            { key: 'retrieved', label: 'Retrieved', num: true }, { key: 'relevant', label: 'Relevant', num: true }, { key: 'hits', label: 'Hits', num: true },
            { key: 'precision', label: 'P', num: true }, { key: 'recall', label: 'R', num: true }, { key: 'f1', label: 'F1', num: true },
            { key: 'p@5', label: 'P@5', num: true }, { key: 'ap', label: 'AP', num: true },
          ]} />
        </Card>
      </Section>

      <Section title="The queries and their judgements" lead="Each query has an information need written before judging; a document is relevant if it would help someone with that need.">
        <Card pad={false}>
          <DataTable rows={queries} dense cols={[
            { key: 'qid', label: 'ID' }, { key: 'query', label: 'Query', render: (x) => <span className="font-mono text-[12.5px] text-ink">{x.query as string}</span> },
            { key: 'information_need', label: 'Information need' },
            { key: 'n_relevant', label: 'Relevant', render: (x) => <div className="flex flex-wrap gap-1">{(p.qrels[x.qid as string] ?? []).map((d: string) => <span key={d} className="rounded bg-surface-2 px-1 font-mono text-[11px] text-ink-2">{d}</span>)}</div> },
          ]} />
        </Card>
        <Callout tone="warn">The relevance judgements were drafted with Claude and must be reviewed by the student before submission. gold/qrels_second_judge_template.csv lets a second person judge a sample blind; Cohen's kappa between the two is computed by nlp_core.evaluate.cohen_kappa and should be reported with the results.</Callout>
        <Why why="Without judged queries, 'better pipeline' has no meaning; the brief's Table I and J need precision, recall and F1 per query and overall."
          where="Last: every pipeline's index answers the same queries and is scored against the same qrels."
          depends="The qrels (gold/qrels.csv), the query analyser matching the document analyser, and deterministic tie-breaking so P@K is repeatable."
          moved="Choosing the selection rule after seeing these numbers would make the comparison circular; it is fixed in code (nlp_core/experiments.py) and was written before the runs." />
      </Section>
    </div>
  )
}
