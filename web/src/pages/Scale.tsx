import { Link } from 'react-router-dom'
import { ArrowRight } from 'lucide-react'
import { useResult, type Any } from '../lib/api'
import { chrome, compact, fmt, runColor, slot } from '../lib/palette'
import Chart, { legend } from '../components/Chart'
import { bars, lines } from '../components/charts'
import { Callout, Card, DataTable, DownloadLink, Grid, Loading, PageHeader, Pill, Section, Stat } from '../components/ui'

const COLL: Record<string, { label: string; color: () => string; about: string }> = {
  mathdial: { label: 'MathDial', color: () => slot(4), about: '2,861 one-to-one tutoring dialogues. A teacher (crowd worker) tutors an LLM-simulated student who holds a real incorrect solution to a GSM8K word problem. Dialogue acts are kept as metadata.' },
  talkmoves: { label: 'TalkMoves', color: () => slot(6), about: '566 transcripts of real K-12 mathematics lessons, coded turn by turn with teacher and student talk moves. Speaker roles are kept; codes go to metadata.' },
}

export default function Scale() {
  const r = useResult('scale')
  if (!r.data) return <Loading label="Loading heavy-dataset results" />
  const d = r.data
  const cols = ['mathdial', 'talkmoves']
  const sp = d.collections.talkmoves.speakers
  const runs: string[] = d.runs
  const pl: Any[] = d.pipelines
  const q: Any[] = d.queries
  const per = (coll: string, key: string) => runs.map((run) => pl.find((x) => x.collection === coll && x.pipeline === run)?.[key] ?? null)

  return (
    <div>
      <PageHeader eyebrow="Scale tier · beyond the 30-document corpus" title="Two heavy datasets, the same pipeline"
        lead={<>The assessment's exercises and relevance judgements use the 30-document corpus. To show the pipeline holds at scale, the
          full <b>MathDial</b> and <b>TalkMoves</b> collections ({fmt(d.words_total)} words) are cleaned, tokenized and indexed with pipelines
          {' '}{runs.join(', ')}, and searched from the same search page.</>} />

      <Grid cols={4}>
        <Stat label="MathDial dialogues" value={fmt(d.collections.mathdial.summary.documents)} note={`${compact(d.collections.mathdial.summary.characters)} characters`} />
        <Stat label="TalkMoves transcripts" value={fmt(d.collections.talkmoves.summary.documents)} note={`${compact(d.collections.talkmoves.summary.characters)} characters`} />
        <Stat label="Words, both collections" value={compact(d.words_total)} note="whitespace tokens" />
        <Stat label="Estimated sentences" value={compact(cols.reduce((a, k) => a + d.collections[k].summary.estimated_sentences, 0))} />
      </Grid>

      <Grid cols={2} className="mt-5">
        {cols.map((k) => (
          <Card key={k} title={<span className="inline-flex items-center gap-2"><span className="h-2.5 w-2.5 rounded-full" style={{ background: COLL[k].color() }} />{COLL[k].label}</span>}>
            <p className="text-sm leading-relaxed text-ink-2">{COLL[k].about}</p>
            <dl className="mt-4 grid grid-cols-3 gap-3 text-sm">
              <div><dt className="text-xs text-ink-3">Tokens</dt><dd className="num font-semibold text-ink">{compact(d.collections[k].summary.tokens_whitespace)}</dd></div>
              <div><dt className="text-xs text-ink-3">Types (B)</dt><dd className="num font-semibold text-ink">{fmt(d.collections[k].zipf.types)}</dd></div>
              <div><dt className="text-xs text-ink-3">Zipf slope</dt><dd className="num font-semibold text-ink">{d.collections[k].zipf.slope}</dd></div>
            </dl>
          </Card>
        ))}
      </Grid>

      <Section title="Building the indexes" lead="Each pipeline was run over each full collection and its positional index cached. B keeps stop words, so its index holds roughly twice as many positions.">
        <Grid cols={2}>
          <Card title="Index positions" subtitle="terms stored in the positional index">
            <Chart height={260} label="Index terms per pipeline" option={() => bars({
              categories: cols.map((k) => COLL[k].label), valueName: 'terms',
              series: runs.map((run) => ({ name: run, color: runColor(run), data: cols.map((k) => pl.find((x) => x.collection === k && x.pipeline === run)?.index_terms ?? null) })),
            })} />
          </Card>
          <Card title="Build time" subtitle="seconds, from cleaned text to a finished index (single process)">
            <Chart height={260} label="Build seconds per pipeline" option={() => bars({
              categories: cols.map((k) => COLL[k].label), valueName: 's', digits: 0,
              series: runs.map((run) => ({ name: run, color: runColor(run), data: cols.map((k) => pl.find((x) => x.collection === k && x.pipeline === run)?.build_seconds ?? null) })),
            })} />
          </Card>
        </Grid>
        <Card pad={false} title="Index summary" actions={<DownloadLink file="scale/scale_pipelines.csv" label="scale_pipelines.csv" />}>
          <DataTable rows={pl} dense cols={[
            { key: 'collection', label: 'Collection' }, { key: 'pipeline', label: 'Pipeline', render: (x) => <Pill><span className="h-2 w-2 rounded-full" style={{ background: runColor(x.pipeline as string) }} />{x.pipeline as string}</Pill> },
            { key: 'index_terms', label: 'Index terms', num: true }, { key: 'vocabulary', label: 'Vocabulary', num: true },
            { key: 'build_seconds', label: 'Build s', num: true, digits: 1 }, { key: 'index_json_mb', label: 'Index JSON MB', num: true, digits: 2 },
            { key: 'median_query_ms', label: 'Median query ms', num: true, digits: 3 },
          ]} />
        </Card>
        <p className="text-xs text-ink-3">Vocabulary per pipeline: MathDial {per('mathdial', 'vocabulary').map((v, i) => `${runs[i]} ${fmt(v)}`).join(' · ')}; TalkMoves {per('talkmoves', 'vocabulary').map((v, i) => `${runs[i]} ${fmt(v)}`).join(' · ')}.</p>
      </Section>

      <Section title="Zipf and Heaps at scale">
        <Grid cols={2}>
          <Card title="Zipf: frequency against rank" subtitle="log-log, pipeline B terms">
            <Chart height={300} label="Zipf at scale" option={() => ({
              ...lines({ xs: [], series: [], logX: true, logY: true }),
              legend: legend(),
              series: cols.map((k) => ({ name: COLL[k].label, type: 'scatter', symbolSize: 5, itemStyle: { color: COLL[k].color(), opacity: 0.85 },
                data: d.collections[k].zipf.points.map((p: Any) => [p.rank, p.freq, p.term]) })) as never,
              tooltip: { trigger: 'item', formatter: (p: Any) => `<b>${p.data[2]}</b><br/>rank ${fmt(p.data[0])} · ${fmt(p.data[1])} occurrences` },
              xAxis: { type: 'log', name: 'rank', nameLocation: 'middle', nameGap: 28, axisLabel: { color: chrome().ink3 }, splitLine: { show: false }, axisLine: { lineStyle: { color: chrome().lineStrong } } },
            })} />
          </Card>
          <Card title="Heaps: vocabulary growth" subtitle={cols.map((k) => `${COLL[k].label} K ${d.collections[k].heaps.K}, β ${d.collections[k].heaps.beta}`).join(' · ')}>
            <Chart height={300} label="Heaps at scale" option={() => ({
              ...lines({ xs: [], series: [], xValue: true }),
              legend: legend(),
              xAxis: { type: 'value', name: 'tokens read', nameLocation: 'middle', nameGap: 28, axisLabel: { color: chrome().ink3, formatter: (v: number) => compact(v) }, splitLine: { show: false }, axisLine: { lineStyle: { color: chrome().lineStrong } } },
              series: cols.map((k) => ({ name: COLL[k].label, type: 'line', showSymbol: false, lineStyle: { width: 2, color: COLL[k].color() }, itemStyle: { color: COLL[k].color() },
                data: d.collections[k].heaps.points.map((p: Any) => [p.tokens, p.vocabulary]) })) as never,
            })} />
          </Card>
        </Grid>
        <Callout>MathDial's vocabulary saturates early (β {d.collections.mathdial.heaps.beta}): its dialogues reuse the same word-problem language. TalkMoves keeps adding words (β {d.collections.talkmoves.heaps.beta}) because every lesson brings its own topic and names.</Callout>
      </Section>

      <Section title="How students and teachers talk (TalkMoves)" lead="Speaker roles are kept when the transcripts are loaded, so the n-gram module can compare them. Student turns are shorter and full of number words and hedges.">
        <Grid cols={3}>
          {(['student', 'teacher'] as const).map((k) => (
            <Stat key={k} label={`${k[0].toUpperCase()}${k.slice(1)} turns`} value={compact(Number(sp[k].turns))} note={`${compact(Number(sp[k].words))} words · ${sp[k].mean_turn_length} words per turn`} />
          ))}
          <Stat label="Teacher words per student word" value={(Number(sp.teacher.words) / Number(sp.student.words)).toFixed(1)} note="teacher talk dominates the lessons" />
        </Grid>
        <Grid cols={2}>
          {(['student', 'teacher'] as const).map((k) => (
            <Card key={k} title={`Top ${k} bigrams`} subtitle="stop words removed, counts over all 566 lessons">
              <Chart height={330} label={`Top ${k} bigrams`} option={() => bars({
                horizontal: true, categories: sp[k].bigrams.slice(0, 12).map((b: Any) => b.ngram), labels: true,
                series: [{ name: 'count', data: sp[k].bigrams.slice(0, 12).map((b: Any) => b.count), color: slot(k === 'student' ? 3 : 1) }],
              })} />
            </Card>
          ))}
        </Grid>
        <Grid cols={2}>
          {(['student', 'teacher'] as const).map((k) => (
            <Card key={k} title={`Top ${k} trigrams`}>
              <div className="flex flex-wrap gap-1.5">{sp[k].trigrams.slice(0, 15).map((t: Any) => <Pill key={t.ngram}>{t.ngram} · {fmt(t.count)}</Pill>)}</div>
            </Card>
          ))}
        </Grid>
      </Section>

      <Section title="Queries at scale" lead="The same Boolean engine answers queries over 3,427 documents in under a millisecond; there are no relevance judgements at this tier, so only counts and latency are reported.">
        <Card title="Median query time" subtitle="milliseconds, 10 queries per collection">
          <Chart height={260} label="Median query time" option={() => bars({
            categories: cols.map((k) => COLL[k].label), digits: 2, valueName: 'ms',
            series: runs.map((run) => ({ name: run, color: runColor(run), data: cols.map((k) => pl.find((x) => x.collection === k && x.pipeline === run)?.median_query_ms ?? null) })),
          })} />
        </Card>
        <Card pad={false} title="Every query, every pipeline" actions={<DownloadLink file="scale/scale_queries.csv" label="scale_queries.csv" />}>
          <DataTable rows={q} dense cols={[
            { key: 'collection', label: 'Collection' }, { key: 'pipeline', label: 'Pipeline' }, { key: 'query', label: 'Query', render: (x) => <span className="font-mono text-[12px]">{x.query as string}</span> },
            { key: 'type', label: 'Type' }, { key: 'results', label: 'Results', num: true }, { key: 'time_ms', label: 'ms', num: true, digits: 3 },
            { key: 'top5', label: 'Top 5', render: (x) => <span className="font-mono text-[11.5px] text-ink-3">{x.top5 as string}</span> },
          ]} />
        </Card>
        <Link to="/search" className="inline-flex items-center gap-1 text-sm font-medium text-accent-ink hover:underline">Search MathDial or TalkMoves yourself <ArrowRight size={14} /></Link>
      </Section>

      <Section title="BPE at scale" lead="Byte-level BPE trained on both heavy collections with the Hugging Face tokenizers library (Rust), against the three OpenAI encodings. Fewer tokens per word means cheaper model input.">
        <Card title="Tokens per word">
          <Chart height={280} label="Tokens per word by BPE vocabulary" option={() => bars({
            categories: d.bpe.map((b: Any) => b.method.replace('our byte-level BPE, ', 'ours · ').replace('tiktoken ', '')), labels: true, digits: 3,
            series: [{ name: 'tokens per word', data: d.bpe.map((b: Any) => b.tokens_per_word), color: slot(1) }],
          })} />
        </Card>
        <Callout>A 32,000-token vocabulary trained on these lessons needs fewer tokens per word ({d.bpe[2]?.tokens_per_word}) than o200k_base ({d.bpe[5]?.tokens_per_word}) with a vocabulary six times smaller. Two caveats: our BPE is measured on the text it was trained on, which flatters it, and token counts are estimated from every tenth document.</Callout>
      </Section>
    </div>
  )
}
