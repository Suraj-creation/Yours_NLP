import { useEffect, useState } from 'react'
import clsx from 'clsx'
import { useMeta, usePost, useResult, type Any } from '../lib/api'
import { fmt, runColor, slot } from '../lib/palette'
import Chart from '../components/Chart'
import { bars, lines } from '../components/charts'
import { Button, Callout, Card, DataTable, DownloadLink, ErrorBox, Examples, Grid, Loading, PageHeader, Pill, Section, Select, TextArea, Why } from '../components/ui'
import { Metrics } from '../components/nlp'

const EXAMPLES = [
  'The students were adding the numerators and the denominators, which is not the same as finding a common denominator.',
  'She simplified 12/8 and was comparing the fractions on a number line before multiplying.',
  'Knowledge tracing models predict whether learners will answer the next question correctly.',
]
const STAGE_LABEL: Record<string, string> = { tokenized: 'Tokenized', lowercased: 'Lower-cased', stopwords_removed: 'Stop words removed', normalized: 'Normalised', index_terms: 'Index terms' }
const METHOD_LABEL: Record<string, string> = { porter: 'Porter', snowball: 'Snowball', lancaster: 'Lancaster', regexp: 'Regexp', wordnet_nopos: 'WordNet (no POS)', wordnet_pos: 'WordNet + POS', spacy: 'spaCy lemma' }

export default function Preprocessing() {
  const meta = useMeta()
  const r = useResult('preprocessing')
  const pl = useResult('pipelines')
  const run = usePost('/api/preprocess')
  const [text, setText] = useState(EXAMPLES[0])
  const [opt, setOpt] = useState({ stopwords: 'custom', stemmer: 'porter', lemmatizer: 'spacy', order: 'stop_then_norm' })
  const go = (s = text, o = opt) => run.mutate({ text: s, ...o })
  useEffect(() => { go() }, []) // eslint-disable-line react-hooks/exhaustive-deps
  if (!r.data || !pl.data || !meta.data) return <Loading />
  const p = r.data
  const set = (k: string) => (v: string) => { const o = { ...opt, [k]: v }; setOpt(o); go(text, o) }
  const runs = ['A', 'B', 'H', 'C1', 'C2', 'C3', 'H2']
  const stageNames = Object.keys(STAGE_LABEL)
  const stem = p.stemmers as Any[]

  return (
    <div>
      <PageHeader eyebrow="Module 2 · Exercises 1 and 7 to 11" title="Stop words, stemming and lemmatization"
        lead={<>The brief's two branches are compared end to end: <b>A</b> removes NLTK stop words and stems with Porter; <b>B</b> keeps every word
          and lemmatizes with WordNet using the POS tag. Every choice is measured by what it does to the vocabulary, to domain words and to retrieval F1.</>} />

      <Card title="Try it" subtitle="Hybrid tokens, then your chosen stop list, stemmer and lemmatizer">
        <TextArea value={text} onChange={setText} rows={3} />
        <Examples items={EXAMPLES} onPick={(s) => { setText(s); go(s) }} />
        <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-5">
          <Select label="Stop list" value={opt.stopwords} onChange={set('stopwords')} options={['none', 'nltk', 'spacy', 'custom'].map((v) => ({ value: v, label: v }))} />
          <Select label="Stemmer" value={opt.stemmer} onChange={set('stemmer')} options={meta.data.stemmers.map((v: string) => ({ value: v, label: METHOD_LABEL[v] ?? v }))} />
          <Select label="Lemmatizer" value={opt.lemmatizer} onChange={set('lemmatizer')} options={meta.data.lemmatizers.map((v: string) => ({ value: v, label: METHOD_LABEL[v] ?? v }))} />
          <Select label="Order" value={opt.order} onChange={set('order')} options={[{ value: 'stop_then_norm', label: 'stop words, then stem' }, { value: 'norm_then_stop', label: 'stem, then stop words' }]} />
          <div className="flex items-end"><Button onClick={() => go()} className="w-full">Run</Button></div>
        </div>
        {run.error && <div className="mt-4"><ErrorBox error={run.error} /></div>}
        {run.data && (
          <div className="mt-5 space-y-4">
            <Metrics items={[
              { label: 'Tokens', value: run.data.counts.tokens }, { label: 'Words', value: run.data.counts.words },
              { label: 'Kept', value: run.data.counts.kept }, { label: 'Unique stems', value: run.data.counts.unique_stems },
              { label: 'Unique lemmas', value: run.data.counts.unique_lemmas },
              { label: 'Stem leaks', value: run.data.rows.filter((x: Any) => x.stem_leak).length, note: 'stop words that survive as stems' },
            ]} />
            <div className="scroll-thin overflow-x-auto rounded-xl border border-line">
              <table className="w-full text-left text-[13px]">
                <thead className="bg-surface-2 text-xs text-ink-3"><tr>{['Token', 'Lower', 'Stop word', 'Stem', 'Lemma'].map((h) => <th key={h} className="px-3 py-2 font-semibold">{h}</th>)}</tr></thead>
                <tbody>
                  {run.data.rows.map((x: Any, i: number) => (
                    <tr key={i} className={clsx('border-t border-line/70', x.stopword && 'text-ink-3', !x.is_word && 'opacity-60')}>
                      <td className="px-3 py-1.5 font-mono">{x.token}</td>
                      <td className="px-3 py-1.5 font-mono">{x.lower}</td>
                      <td className="px-3 py-1.5">{x.stopword ? <Pill tone="neutral">removed</Pill> : x.stem_leak ? <Pill tone="warn">leaks as “{x.stem}”</Pill> : x.is_word ? <span className="text-good-ink">kept</span> : <span>punct</span>}</td>
                      <td className={clsx('px-3 py-1.5 font-mono', x.stem !== x.lower && 'text-accent-ink')}>{x.stem}</td>
                      <td className={clsx('px-3 py-1.5 font-mono', x.lemma.toLowerCase() !== x.lower && 'text-accent-ink')}>{x.lemma}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </Card>

      <Section title="Table A: before and after preprocessing" lead={`For the selected pipeline ${p.final_pipeline}: raw NLTK tokens before, index terms after.`}>
        <Card pad={false} title="preprocessing_results.csv" actions={<DownloadLink file="preprocessing_results.csv" />}>
          <DataTable rows={p.table_a.map((x: Any) => ({ ...x, change: x.before ? (x.after - x.before) / x.before : 0 }))} cols={[
            { key: 'measure', label: 'Measure' }, { key: 'before', label: 'Before', num: true }, { key: 'after', label: 'After', num: true },
            { key: 'change', label: 'Change', num: true, render: (x) => `${((x.change as number) * 100).toFixed(1)}%` },
          ]} />
        </Card>
        <Card title="Vocabulary at each stage, every pipeline" subtitle="Where each pipeline shrinks its dictionary">
          <Chart height={320} label="Vocabulary per stage" option={() => lines({
            xs: stageNames.map((s) => STAGE_LABEL[s]), markers: true,
            series: runs.map((run) => ({ name: run, color: runColor(run), data: stageNames.map((st) => pl.data.stages.find((x: Any) => x.pipeline === run && x.stage === st)?.vocabulary ?? null) })),
          })} />
        </Card>
      </Section>

      <Section title="Exercise 7: which stop list" lead="Standard lists delete words that carry meaning in mathematics: not, more, than, over, under, same, each. The custom list starts from NLTK, keeps those, and adds corpus noise (et, al, fig, um, uh).">
        <Grid cols={2}>
          <Card title="Index size by stop list" subtitle="index tokens with pipeline B's tokens and lemmas">
            <Chart height={260} label="Index tokens per stop list" option={() => bars({
              categories: p.stopwords.map((x: Any) => x.stopword_list), labels: true,
              series: [{ name: 'index tokens', data: p.stopwords.map((x: Any) => x.index_tokens), color: slot(1) }],
            })} />
          </Card>
          <Card title="Mean F1 by stop list" subtitle="15 gold queries">
            <Chart height={260} label="F1 per stop list" option={() => bars({
              categories: p.stopwords.map((x: Any) => x.stopword_list), labels: true, digits: 4, max: 0.8,
              series: [{ name: 'mean F1', data: p.stopwords.map((x: Any) => x.mean_f1), color: slot(2) }],
            })} />
          </Card>
        </Grid>
        <Card pad={false} title="stopword_comparison.csv" actions={<DownloadLink file="stopword_comparison.csv" />}>
          <DataTable rows={p.stopwords} cols={[
            { key: 'stopword_list', label: 'List' }, { key: 'list_size', label: 'Size', num: true }, { key: 'index_tokens', label: 'Index tokens', num: true },
            { key: 'vocabulary', label: 'Vocabulary', num: true }, { key: 'mean_f1', label: 'Mean F1', num: true, digits: 4 },
            { key: 'domain_words_removed', label: 'Meaningful words it removes', render: (x) => <span className="text-[12px] text-ink-3">{(x.domain_words_removed as string) || 'none'}</span> },
          ]} />
        </Card>
        <Card title="The custom stop list" subtitle={`${p.custom_stoplist.length} words; words kept back from NLTK are highlighted`}>
          <div className="flex flex-wrap gap-1">
            {p.kept_words.map((w: string) => <span key={w} className="rounded-md border border-good/30 bg-good/10 px-1.5 py-0.5 font-mono text-[12px] text-good-ink">{w}</span>)}
            {p.custom_stoplist.map((w: string) => <span key={w} className="rounded-md border border-line bg-surface-2 px-1.5 py-0.5 font-mono text-[12px] text-ink-3 line-through decoration-ink-3/40">{w}</span>)}
          </div>
          <p className="mt-3 text-xs text-ink-3">Green words are removed by NLTK but kept here; struck-through words are removed.</p>
        </Card>
        <Callout>On these 15 queries the stop list barely moves F1, because no query depends on a stop word. It still matters for n-grams and for learner language: "not the same" and "more than" survive only with the custom list.</Callout>
      </Section>

      <Section title="Exercises 8 and 9: stemmers against lemmatizers" lead="64 word pairs from the corpus: 40 that should merge (denominator / denominators) and 24 that should not (numerator / numerical).">
        <Grid cols={2}>
          <Card title="Merge errors" subtitle="over-stemming merges words that differ; under-stemming leaves forms apart">
            <Chart height={290} label="Stemming errors" option={() => bars({
              categories: stem.map((x) => METHOD_LABEL[x.method] ?? x.method), rotate: 20,
              series: [
                { name: 'over-stemming', data: stem.map((x) => x.over_stemming_errors), color: slot(8) },
                { name: 'under-stemming', data: stem.map((x) => x.under_stemming_errors), color: slot(4) },
              ],
            })} />
          </Card>
          <Card title="Distinct forms after normalising the corpus" subtitle="fewer forms means a smaller dictionary">
            <Chart height={290} label="Unique forms" option={() => bars({
              categories: stem.map((x) => METHOD_LABEL[x.method] ?? x.method), rotate: 20, labels: true,
              series: [{ name: 'unique forms', data: stem.map((x) => x.unique_forms), color: slot(1) }],
            })} />
          </Card>
        </Grid>
        <Card pad={false} title="Every merge error" actions={<DownloadLink file="stemmer_errors.csv" />}>
          <DataTable rows={p.stem_errors} dense cols={[
            { key: 'method', label: 'Method', render: (x) => METHOD_LABEL[x.method as string] ?? (x.method as string) },
            { key: 'error', label: 'Error', render: (x) => <Pill tone={x.error === 'over' ? 'bad' : 'warn'}>{x.error as string}</Pill> },
            { key: 'pair', label: 'Pair' }, { key: 'forms', label: 'Forms produced', render: (x) => <span className="font-mono">{x.forms as string}</span> },
          ]} />
        </Card>
        <Card pad={false} title="Table C: stemming and lemmatization of 30 domain words" actions={<DownloadLink file="stemming_lemmatization.csv" />}>
          <DataTable rows={p.table_c} dense cols={[
            { key: 'word', label: 'Word', render: (x) => <b className="font-mono text-ink">{x.word as string}</b> },
            ...['stem_porter', 'stem_snowball', 'stem_lancaster', 'stem_regexp', 'lemma_wordnet_nopos', 'lemma_wordnet_pos', 'lemma_spacy'].map((k) => ({
              key: k, label: METHOD_LABEL[k.replace(/^stem_|^lemma_/, '')] ?? k,
              render: (x: Any) => <span className={clsx('font-mono', x[k] === x.word ? 'text-ink-3' : 'text-ink')}>{x[k]}</span>,
            })),
          ]} />
        </Card>
      </Section>

      <Section title="Exercise 10: lemmatizer accuracy" lead="220 word tokens from the POS gold sentences, each with a hand-written lemma.">
        <Grid cols={3}>
          {p.lemmas.map((x: Any) => (
            <Card key={x.lemmatizer} title={METHOD_LABEL[x.lemmatizer]} subtitle={`${x.correct} of ${x.tokens} correct`}>
              <div className="num text-3xl font-semibold text-ink">{(x.accuracy * 100).toFixed(1)}%</div>
              <div className="mt-3 flex flex-wrap gap-1">{String(x.errors).split('; ').map((e) => <span key={e} className="rounded-md bg-surface-2 px-1.5 py-0.5 font-mono text-[11.5px] text-ink-3">{e}</span>)}</div>
            </Card>
          ))}
        </Grid>
        <Callout>WordNet without a POS tag treats every word as a noun, so "is", "added" and "bigger" stay unchanged and "uses" becomes "us". Passing the tag fixes almost all of it, which is why pipeline B tags before lemmatizing.</Callout>
      </Section>

      <Section title="Exercise 11: does the order matter?" lead="Stop-word removal before stemming against stemming first. Stemming first changes stop words into forms the list no longer recognises.">
        <Card pad={false} title="order_experiment.csv" actions={<DownloadLink file="order_experiment.csv" />}>
          <DataTable rows={p.order} cols={[
            { key: 'run', label: 'Run' }, { key: 'order', label: 'Order' }, { key: 'index_tokens', label: 'Index tokens', num: true },
            { key: 'stop_word_stems_leaked', label: 'Leaked stems', num: true }, { key: 'leaked_occurrences', label: 'Leaked occurrences', num: true },
            { key: 'examples', label: 'Examples', render: (x) => <span className="font-mono text-[12px]">{x.examples as string}</span> },
            { key: 'mean_f1', label: 'Mean F1', num: true, digits: 4 },
          ]} />
        </Card>
        <Callout tone="warn">Stemming first turns "this" into "thi", "has" into "ha" and "was" into "wa"; none of those is on the stop list, so {fmt(p.order[1]?.leaked_occurrences)} extra noise tokens reach the index. Pipeline A therefore removes stop words before stemming.</Callout>
        <Why why="Stop words dominate counts and n-grams; stems and lemmas merge the inflected forms a query should match (denominators → denominator)."
          where="After tokenization and lower-casing. In A, stop words go first, then Porter; in B, spaCy tags each sentence and WordNet lemmatizes with the tag."
          depends="Stemmers need lower-cased tokens; the WordNet lemmatizer needs a POS tag, so B depends on the tagger (Exercise 5)."
          moved="Stemming before stop-word removal leaks stems like 'thi' and 'wa' (above). Lemmatizing without POS drops accuracy from 98% to 87%. Removing stop words before POS tagging would take away the context the tagger uses." />
      </Section>

      <Section title="Most frequent terms" lead={`Pipeline ${p.final_pipeline} keeps stop words (it won on retrieval), so function words stay at the top of its index; lemmatization merges 'is', 'are' and 'was' into 'be'.`}>
        <Grid cols={2}>
          {(['top_before', 'top_after'] as const).map((k) => (
            <Card key={k} title={k === 'top_before' ? 'Before: lower-cased NLTK word tokens' : `After: pipeline ${p.final_pipeline} index terms`}>
              <Chart height={420} label={k} option={() => bars({
                horizontal: true, categories: p[k].slice(0, 20).map((x: Any) => x.term), labels: true,
                series: [{ name: 'count', data: p[k].slice(0, 20).map((x: Any) => x.count), color: slot(k === 'top_before' ? 1 : 2) }],
              })} />
            </Card>
          ))}
        </Grid>
        <DownloadLink file="preprocessing_stages.csv" />
      </Section>
    </div>
  )
}
