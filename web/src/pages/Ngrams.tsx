import { useState } from 'react'
import { useResult, type Any } from '../lib/api'
import { chrome, fmt, slot } from '../lib/palette'
import Chart, { tooltip } from '../components/Chart'
import { bars, lines } from '../components/charts'
import { Callout, Card, DataTable, DownloadLink, Grid, Loading, PageHeader, Section, Segmented, Why } from '../components/ui'
import { Metrics } from '../components/nlp'

const VIEWS = [
  { value: 'all', label: 'All n-grams' },
  { value: 'meaningful_filter', label: 'Meaningful filter' },
  { value: 'after_stopword_removal', label: 'After stop-word removal' },
]
const NAMES = ['', 'Unigrams', 'Bigrams', 'Trigrams', '4-grams', '5-grams']

export default function Ngrams() {
  const r = useResult('ngrams')
  const [n, setN] = useState('2')
  const [view, setView] = useState('meaningful_filter')
  const [coll, setColl] = useState<'bigram' | 'trigram'>('bigram')
  if (!r.data) return <Loading />
  const g = r.data
  const cur = g.per_n[n][view]
  const dict: Any[] = g.dictionary
  const net = g.network
  const maxW = Math.max(...net.nodes.map((x: Any) => x.weight))

  return (
    <div>
      <PageHeader eyebrow="Module 2 · Exercise 14" title="N-grams and collocations"
        lead={<>Unigrams to 5-grams over pipeline {g.final_pipeline}'s index terms, counted within sentences. Two ways of cleaning them are compared:
          a <b>meaningful filter</b> that drops n-grams starting or ending with a stop word or punctuation (keeping "not the same"), and plain
          <b> stop-word removal</b> before counting, which glues words that were never adjacent.</>} />

      <Card title={`Top ${NAMES[Number(n)].toLowerCase()}`} actions={<>
        <Segmented value={n} onChange={setN} options={['1', '2', '3', '4', '5'].map((v) => ({ value: v, label: `n=${v}` }))} />
        <Segmented value={view} onChange={setView} options={VIEWS} />
      </>}>
        <Metrics items={[
          { label: 'Total', value: cur.total }, { label: 'Unique', value: cur.unique }, { label: 'Seen once', value: cur.singletons },
          { label: 'Singleton share', value: `${(cur.singleton_share * 100).toFixed(1)}%` },
          { label: 'Top count', value: cur.top[0]?.count }, { label: 'Top share', value: `${((cur.top[0]?.count / cur.total) * 100).toFixed(2)}%` },
        ]} />
        <div className="mt-4">
          <Chart height={340} label="Top n-grams" option={() => bars({
            horizontal: true, categories: cur.top.map((x: Any) => x.ngram), labels: true, categoryWidth: 220,
            series: [{ name: 'count', data: cur.top.map((x: Any) => x.count), color: slot(1) }],
          })} />
        </div>
      </Card>

      <Section title="Dictionary size against n" lead="Every extra word makes n-grams rarer: most 4- and 5-grams occur once, so their dictionaries are huge and nearly useless as features.">
        <Grid cols={2}>
          <Card title="Distinct n-grams">
            <Chart height={280} label="Dictionary size by n" option={() => lines({
              xs: dict.map((d) => `n=${d.n}`), markers: true,
              series: [
                { name: 'all', data: dict.map((d) => d.dictionary_size_all), color: slot(1) },
                { name: 'meaningful filter', data: dict.map((d) => d.dictionary_size_filtered), color: slot(3) },
                { name: 'after stop-word removal', data: dict.map((d) => d.dictionary_size_after_removal), color: slot(4) },
              ],
            })} />
          </Card>
          <Card title="N-grams invented by stop-word removal" subtitle="present after removal but never adjacent in the text">
            <Chart height={280} label="Invented n-grams" option={() => bars({
              categories: dict.map((d) => `n=${d.n}`), labels: true,
              series: [{ name: 'invented n-grams', data: dict.map((d) => d.invented_by_removal), color: slot(8) }],
            })} />
          </Card>
        </Grid>
        <Callout tone="warn">Removing stop words first creates {fmt(dict[1]?.invented_by_removal)} bigrams that never occur in the text. The most frequent is "numerator denominator" (101 times, from "the numerator and the denominator"); others, such as "operation student", join words from different clauses. The meaningful filter drops edge stop words without joining distant words, which is why it is the default here.</Callout>
      </Section>

      <Section title="Collocations" lead="Pairs and triples that occur together more than chance predicts. PMI rewards rare exclusive pairs; the log-likelihood ratio is stabler for rare counts, so the table ranks by it.">
        <Card pad={false} title={coll === 'bigram' ? 'Bigram collocations' : 'Trigram collocations'} actions={<>
          <Segmented value={coll} onChange={(v) => setColl(v as 'bigram' | 'trigram')} options={[{ value: 'bigram', label: 'Bigrams' }, { value: 'trigram', label: 'Trigrams' }]} />
          <DownloadLink file="collocations.csv" /></>}>
          <DataTable rows={g.collocations[coll]} dense initialSort={{ key: 'log_likelihood', dir: -1 }} cols={[
            { key: 'ngram', label: 'Collocation', render: (x) => <b className="text-ink">{x.ngram as string}</b> }, { key: 'count', label: 'Count', num: true },
            { key: 'log_likelihood', label: 'Log-likelihood', num: true, digits: 1 }, { key: 'pmi', label: 'PMI', num: true, digits: 2 },
          ]} />
        </Card>
      </Section>

      <Section title="Bigram network" lead="The 70 strongest meaningful bigrams as a graph. Node size is the word's count; drag nodes to untangle, scroll to zoom.">
        <Card pad={false}>
          <Chart height={620} label="Bigram network graph" option={() => {
            const c = chrome()
            return {
              tooltip: tooltip({ formatter: (p: Any) => p.dataType === 'edge' ? `<b>${p.data.source} ${p.data.target}</b><br/>${fmt(p.data.value)} times` : `<b>${p.data.name}</b><br/>${fmt(p.data.value)} in bigrams` }),
              series: [{
                type: 'graph', layout: 'force', roam: true, draggable: true,
                force: { repulsion: 120, edgeLength: [36, 90], gravity: 0.2, friction: 0.25 },
                data: net.nodes.map((x: Any) => ({ name: x.id, value: x.weight, symbolSize: 8 + 26 * Math.sqrt(x.weight / maxW),
                  itemStyle: { color: slot(1), borderColor: c.surface, borderWidth: 2 } })),
                links: net.edges.map((e: Any) => ({ source: e.source, target: e.target, value: e.weight })),
                lineStyle: { color: c.lineStrong, width: 1.2, opacity: 0.8, curveness: 0.08 },
                label: { show: true, position: 'right', fontSize: 11, color: c.ink2 },
                labelLayout: { hideOverlap: true },
                emphasis: { focus: 'adjacency', lineStyle: { width: 2.5, color: c.accent } },
              }],
            }
          }} />
        </Card>
      </Section>

      <Section title="Table F: n-gram summary">
        <Card pad={false} title="ngram_results.csv" actions={<><DownloadLink file="unigram_results.csv" /><DownloadLink file="bigram_results.csv" /><DownloadLink file="trigram_results.csv" /><DownloadLink file="ngram_results.csv" /></>}>
          <DataTable rows={g.table_f} cols={[
            { key: 'ngram', label: 'N-gram' }, { key: 'total', label: 'Total', num: true }, { key: 'unique', label: 'Unique', num: true },
            { key: 'top_ngrams_filtered', label: 'Top 10 (meaningful filter)', render: (x) => <span className="text-[12px] leading-relaxed text-ink-2">{x.top_ngrams_filtered as string}</span> },
          ]} />
        </Card>
        <Why why="N-grams capture phrases a single word cannot ('common denominator', 'add the numerators'); they drive phrase queries and the choice of domain terms for the lexicon."
          where="On the final index terms, within sentence boundaries, so no n-gram crosses a full stop or a dialogue turn."
          depends="Tokenization (a split 1/2 would create '1 /' and '/ 2' bigrams), lemmatization (merges 'numerators' and 'numerator') and sentence splitting."
          moved="Counting after stop-word removal invents n-grams (chart above). Counting before lemmatization splits one phrase across its inflections." />
      </Section>
    </div>
  )
}
