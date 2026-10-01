import { useState } from 'react'
import { useResult, type Any } from '../lib/api'
import { chrome, fmt, layerColor, slot } from '../lib/palette'
import Chart, { legend } from '../components/Chart'
import { bars, lines, scatter } from '../components/charts'
import { Callout, Card, DataTable, DownloadLink, Grid, Loading, PageHeader, Section, Segmented, Stat, Why } from '../components/ui'
import { LayerDot } from '../components/nlp'

const LAYERS = ['concept', 'research', 'learner', 'synthetic']

export default function Statistics() {
  const c = useResult('corpus')
  const [measure, setMeasure] = useState<'tokens' | 'vocabulary' | 'type_token_ratio' | 'sentences'>('tokens')
  if (!c.data) return <Loading />
  const s = c.data.summary
  const docs: Any[] = c.data.documents
  const zipf: Any[] = c.data.zipf.points
  const heaps: Any[] = c.data.heaps.points
  // Line with the slope fitted over every rank (reported by stats.zipf), anchored on the sampled points.
  const zb = c.data.zipf.slope as number
  const za = Math.exp(zipf.reduce((a, p) => a + Math.log(p.freq) - zb * Math.log(p.rank), 0) / zipf.length)
  const { K, beta } = c.data.heaps
  const sl = c.data.sentence_lengths
  const byLayer = LAYERS.map((l) => {
    const d = docs.filter((x) => x.layer === l)
    return { layer: l, docs: d.length, tokens: d.reduce((a, x) => a + x.tokens, 0) }
  })
  const mLabel = { tokens: 'Tokens', vocabulary: 'Vocabulary', type_token_ratio: 'Type-token ratio', sentences: 'Sentences' }[measure]

  return (
    <div>
      <PageHeader eyebrow="Module 1 · Corpus statistics" title="How big the corpus is and how its words are spread"
        lead="Counts use NLTK word tokens after cleaning, the same basis the brief's Table A uses. Vocabulary is lower-cased word types." />

      <Grid cols={4}>
        <Stat label="Documents" value={s.documents} />
        <Stat label="Sentences" value={fmt(s.sentences)} note="NLTK Punkt, lines as hard boundaries" />
        <Stat label="Tokens" value={fmt(s.tokens)} note={`${fmt(s.word_tokens)} are words`} />
        <Stat label="Vocabulary" value={fmt(s.vocabulary)} note="lower-cased word types" />
        <Stat label="Characters" value={fmt(s.characters)} />
        <Stat label="Mean document length" value={fmt(s.avg_doc_length_tokens, 0)} note="tokens per document" />
        <Stat label="Mean document length" value={fmt(s.avg_doc_length_characters, 0)} note="characters per document" />
        <Stat label="Hapax share" value={`${((docs.reduce((a, d) => a + d.hapax, 0) / docs.reduce((a, d) => a + d.vocabulary, 0)) * 100).toFixed(0)}%`} note="of per-document types seen once" />
      </Grid>

      <Section title="Per document" lead="Colour shows the corpus layer. The MaE file (D16) and the two MathDial files (D17, D18) are by far the longest; the synthetic artefacts are short by design.">
        <Card title={`${mLabel} per document`} actions={
          <Segmented value={measure} onChange={(v) => setMeasure(v as typeof measure)}
            options={[{ value: 'tokens', label: 'Tokens' }, { value: 'vocabulary', label: 'Vocabulary' }, { value: 'sentences', label: 'Sentences' }, { value: 'type_token_ratio', label: 'TTR' }]} />}>
          <div className="mb-3 flex flex-wrap gap-4 text-xs">{LAYERS.map((l) => <LayerDot key={l} layer={l} />)}</div>
          <Chart height={320} label={`${mLabel} per document`} option={() => bars({
            categories: docs.map((d) => d.doc_id), series: [{ name: mLabel, data: docs.map((d) => d[measure]), color: slot(1) }],
            itemColors: docs.map((d) => layerColor(d.layer)), digits: measure === 'type_token_ratio' ? 2 : 0, rotate: 45,
          })} />
        </Card>
        <Grid cols={4}>
          {byLayer.map((b) => (
            <div key={b.layer} className="rounded-2xl border border-line bg-surface p-4">
              <LayerDot layer={b.layer} />
              <div className="num mt-2 text-xl font-semibold text-ink">{fmt(b.tokens)}</div>
              <div className="text-xs text-ink-3">tokens in {b.docs} documents · {((b.tokens / s.tokens) * 100).toFixed(0)}% of the corpus</div>
            </div>
          ))}
        </Grid>
      </Section>

      <Section title="Zipf and Heaps" lead="Both laws hold, which is a sanity check that cleaning did not distort the text.">
        <Grid cols={2}>
          <Card title="Zipf's law: frequency against rank" subtitle={`Log-log. Least-squares slope over all ${fmt(c.data.zipf.types)} ranks: ${zb}`}>
            <Chart height={300} label="Zipf plot" option={() => ({
              ...scatter({ series: [{ name: 'Observed', data: zipf.map((p) => [p.rank, p.freq, p.term]), color: slot(1), size: 6 }], xName: 'rank', logX: true, logY: true }),
              series: [
                { name: 'Observed', type: 'scatter', data: zipf.map((p) => [p.rank, p.freq, p.term]), symbolSize: 6, itemStyle: { color: slot(1), opacity: 0.85 } },
                { name: 'Power-law fit', type: 'line', showSymbol: false, data: [1, c.data.zipf.types].map((r: number) => [r, za * r ** zb]), lineStyle: { width: 2, color: chrome().ink3, type: 'dashed' } },
              ] as never,
              legend: legend(),
            })} />
          </Card>
          <Card title="Heaps' law: vocabulary against tokens" subtitle={`V = K·N^β with K = ${K}, β = ${beta}`}>
            <Chart height={300} label="Heaps plot" option={() => lines({
              xs: heaps.map((p) => p.tokens), xValue: true, xName: 'tokens read',
              series: [
                { name: 'Observed', data: heaps.map((p) => p.vocabulary), color: slot(1) },
                { name: 'K·N^β', data: heaps.map((p) => K * p.tokens ** beta), color: chrome().ink3 },
              ],
            })} />
          </Card>
        </Grid>
        <Callout>A β of {beta} means vocabulary keeps growing with every new document: fractions like "240/4" and ratios like "3:5" are types the hybrid tokenizer keeps whole, and most of them occur once.</Callout>
      </Section>

      <Section title="Sentence lengths" lead="Punkt and spaCy's sentencizer agree on all but a few sentences because newlines are treated as hard boundaries first.">
        <Card title="Sentences by length (tokens, bins of 5)" subtitle={`Punkt ${fmt(c.data.sentence_totals.punkt)} sentences · spaCy ${fmt(c.data.sentence_totals.spacy)}`}>
          <Chart height={280} label="Sentence length histogram" option={() => bars({
            categories: sl.punkt.map((b: Any) => (b.bin >= 80 ? '80+' : `${b.bin}–${b.bin + 4}`)),
            series: [
              { name: 'NLTK Punkt', data: sl.punkt.map((b: Any) => b.count), color: slot(1) },
              { name: 'spaCy sentencizer', data: sl.spacy.map((b: Any) => b.count), color: slot(2) },
            ],
          })} />
        </Card>
        <Why why="Sentence boundaries decide where n-grams stop, what the POS tagger sees as context and how positions are gapped in the index."
          where="After cleaning, before tokenization; each line is split separately so dialogue turns and table rows never merge."
          depends="Cleaned text with line breaks kept (the whitespace step collapses spaces but keeps newlines)."
          moved="Splitting before cleaning would break sentences at abbreviations inside markup and at page numbers." />
      </Section>

      <Section title="Document statistics table">
        <Card pad={false} title="document_statistics.csv" actions={<DownloadLink file="document_statistics.csv" />}>
          <DataTable rows={docs} dense cols={[
            { key: 'doc_id', label: 'ID' }, { key: 'format', label: 'Format' }, { key: 'layer', label: 'Layer', render: (r) => <LayerDot layer={r.layer as string} /> },
            { key: 'sentences', label: 'Sentences', num: true }, { key: 'tokens', label: 'Tokens', num: true }, { key: 'word_tokens', label: 'Words', num: true },
            { key: 'vocabulary', label: 'Vocabulary', num: true }, { key: 'type_token_ratio', label: 'TTR', num: true, digits: 3 },
            { key: 'hapax', label: 'Hapax', num: true }, { key: 'characters', label: 'Characters', num: true }, { key: 'raw_characters', label: 'Raw chars', num: true },
          ]} />
        </Card>
      </Section>
    </div>
  )
}
