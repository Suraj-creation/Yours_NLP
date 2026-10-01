import { useState } from 'react'
import { useResult, type Any } from '../lib/api'
import { slot } from '../lib/palette'
import Chart from '../components/Chart'
import { bars, heatmap } from '../components/charts'
import { Bool, Callout, Card, DataTable, DownloadLink, Grid, Loading, PageHeader, Pill, Section, Select, Why } from '../components/ui'

const RULES = [
  ['number/fraction pattern', 'A token like 3/4, 3 1/2, 0.75, 25% or 1/2+1/3=2/5 is CD, whatever the base tagger said.'],
  ['operator between numbers', 'x, ×, +, =, ÷ or * between two numbers is SYM (NLTK calls "x" a noun).'],
  ['domain dictionary', '37 entries: LCD, GCF → NN; BKT, DKT, MathDial, ASSIST2009 → NNP; halves, thirds → NNS; idk → UH; u → PRP.'],
  ['math instruction verb (imperative)', 'Simplify, Round, Subtract… at the start of a sentence or after "then" / "and" / "first" is VB.'],
  ['top/bottom as modifier', 'top or bottom before number(s) or part(s) is JJ ("the top number").'],
  ["'over' as fraction bar", '"over" after a number or number word is IN ("one over two").'],
]

export default function CustomPos() {
  const r = useResult('pos')
  const [cm, setCm] = useState('rules_nltk')
  if (!r.data) return <Loading />
  const p = r.data
  const acc: Any[] = p.accuracy
  const labels: string[] = p.labels
  const matrix: number[][] = p.confusion[cm]
  const cells: [number, number, number][] = matrix.flatMap((row, i) => row.map((v, j) => [j, i, v] as [number, number, number]))
  const nonDiag = matrix.reduce((a, row, i) => a + row.reduce((b, v, j) => b + (i === j ? 0 : v), 0), 0)

  return (
    <div>
      <PageHeader eyebrow="Module 2 · Exercise 12" title="A custom POS tagger for mathematics talk"
        lead={<>Two kinds of custom tagger, as the brief asks: a <b>rules and dictionary</b> layer over a default tagger, and <b>machine-learned</b>{" "}
          taggers (an NLTK trigram backoff chain and a CRF) trained on the Penn Treebank sample plus 20 domain sentences weighted five times.
          All are scored on 12 held-out domain sentences.</>} />

      <Grid cols={3}>
        {acc.filter((a) => ['nltk', 'rules_nltk', 'crf'].includes(a.tagger)).map((a) => (
          <div key={a.tagger} className="rounded-2xl border border-line bg-surface px-5 py-4 shadow-[var(--shadow)]">
            <div className="text-[13px] text-ink-3">{a.label}</div>
            <div className="num mt-1 text-[26px] font-semibold text-ink">{(a.accuracy_test * 100).toFixed(1)}%</div>
            <div className="text-xs text-ink-3">{a.correct_test} of {a.tokens_test} test tokens</div>
          </div>
        ))}
      </Grid>

      <Section title="Accuracy on the held-out sentences" lead="97 tokens in 12 sentences the ML taggers never saw. One token is one point, so differences of one or two tokens are within noise; the rule layer's gain over NLTK (7 tokens) is not.">
        <Grid cols={2}>
          <Card title="Test accuracy by tagger">
            <Chart height={300} label="POS accuracy" option={() => bars({
              horizontal: true, categories: acc.map((a) => a.label), percent: true, digits: 1, labels: true, max: 1, categoryWidth: 170,
              series: [{ name: 'accuracy', data: acc.map((a) => a.accuracy_test), color: slot(1) }],
              itemColors: acc.map((a) => (a.tagger.startsWith('rules') ? slot(3) : ['backoff', 'crf'].includes(a.tagger) ? slot(7) : slot(1))),
            })} />
            <div className="mt-2 flex flex-wrap gap-4 text-xs text-ink-2">
              {[[1, 'default'], [3, 'rules + dictionary'], [7, 'machine-learned']].map(([s, l]) => <span key={l} className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: slot(s as number) }} />{l}</span>)}
            </div>
          </Card>
          <Card pad={false} title="pos_tagger_accuracy.csv" actions={<DownloadLink file="pos_tagger_accuracy.csv" />}>
            <DataTable rows={acc} dense cols={[
              { key: 'label', label: 'Tagger' }, { key: 'accuracy_test', label: 'Test', num: true, render: (x) => `${((x.accuracy_test as number) * 100).toFixed(1)}%` },
              { key: 'accuracy_all_32', label: 'All 32', num: true, render: (x) => (x.accuracy_all_32 === '' ? '—' : `${((x.accuracy_all_32 as number) * 100).toFixed(1)}%`) },
              { key: 'top_errors', label: 'Top errors (gold→predicted)', render: (x) => <span className="font-mono text-[11.5px] text-ink-3">{x.top_errors as string}</span> },
            ]} />
          </Card>
        </Grid>
        <p className="text-xs text-ink-3">"All 32" includes the 20 training sentences, so it is reported only for taggers that did not train on them. spaCy's UPOS accuracy on the test set is {(p.upos_spacy_test * 100).toFixed(1)}%, lower than its Penn tags because the gold was written in Penn tags and mapped.</p>
      </Section>

      <Section title="Confusion matrix" lead="Rows are gold tags, columns predicted tags, test sentences only.">
        <Card title={`${acc.find((a) => a.tagger === cm)?.label} · ${nonDiag} errors`} actions={
          <Select value={cm} onChange={setCm} options={acc.map((a) => ({ value: a.tagger, label: a.label }))} />}>
          <Chart height={560} label="POS confusion matrix" option={() => heatmap({ xLabels: labels, yLabels: labels, data: cells, showValues: true, xRotate: 45 })} />
        </Card>
      </Section>

      <Section title="Table D: words the default tagger gets wrong" lead="Each row is a test-sentence token where the default and custom taggers differ or both are wrong.">
        <Card pad={false} title="pos_tagging_results.csv" actions={<DownloadLink file="pos_tagging_results.csv" />}>
          <DataTable rows={p.table_d} cols={[
            { key: 'sentence', label: 'Sentence' }, { key: 'word', label: 'Word', render: (x) => <b className="font-mono text-ink">{x.word as string}</b> },
            { key: 'default_pos_nltk', label: 'Default (NLTK)', render: (x) => <span className="font-mono">{x.default_pos_nltk as string}</span> },
            { key: 'custom_pos_rules', label: 'Custom (rules)', render: (x) => <span className="font-mono">{x.custom_pos_rules as string}</span> },
            { key: 'custom_pos_crf', label: 'Custom (CRF)', render: (x) => <span className="font-mono">{x.custom_pos_crf as string}</span> },
            { key: 'gold', label: 'Gold', render: (x) => <b className="font-mono">{x.gold as string}</b> },
            { key: 'default_correct', label: 'Default right', render: (x) => <Bool v={x.default_correct} /> },
            { key: 'custom_correct', label: 'Custom right', render: (x) => <Bool v={x.custom_correct} /> },
            { key: 'rule_fired', label: 'Rule', render: (x) => <span className="text-xs text-ink-3">{x.rule_fired as string}</span> },
          ]} />
        </Card>
      </Section>

      <Section title="How the custom taggers work">
        <Grid cols={2}>
          <Card title="Rule layer" subtitle="Applied in this order over NLTK or spaCy tags; the first matching rule wins">
            <ol className="space-y-3 text-sm">
              {RULES.map(([k, v], i) => (
                <li key={k} className="flex gap-3"><span className="num mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-accent-soft text-[11px] font-semibold text-accent-ink">{i + 1}</span>
                  <div><div className="font-medium text-ink">{k}</div><div className="text-ink-2">{v}</div></div></li>
              ))}
            </ol>
          </Card>
          <Card title="Machine-learned taggers" subtitle="Both trained on 3,914 Treebank sentences plus the 20 domain training sentences × 5">
            <div className="space-y-4 text-sm leading-relaxed text-ink-2">
              <p><b className="text-ink">Backoff chain.</b> Trigram → bigram → unigram → regular-expression tagger (fractions and numbers to CD, -ing to VBG, -ed to VBD, capitalised to NNP, else NN), as in the lab.</p>
              <p><b className="text-ink">CRF.</b> sklearn-crfsuite with L-BFGS (c1 0.1, c2 0.05). Features: the word, suffixes, prefix, shape (Xxx, d/d), is-fraction, has-hyphen, in-domain-lexicon, math-imperative, and the same for the neighbouring words.</p>
              <p>The CRF learns the rules' patterns from features but still calls sentence-initial "Round" a noun: with 20 domain sentences it has too few imperatives to override the Treebank.</p>
            </div>
          </Card>
        </Grid>
        <Card title="Domain dictionary" subtitle={`${Object.keys(p.lexicon).length} entries`}>
          <div className="flex flex-wrap gap-1.5">
            {Object.entries(p.lexicon as Record<string, string>).map(([w, t]) => <Pill key={w}><span className="font-mono">{w}</span><span className="text-ink-3">{t}</span></Pill>)}
          </div>
        </Card>
        <Card title="The 12 held-out test sentences">
          <ol className="list-decimal space-y-1 pl-5 text-sm text-ink-2">{p.test_sentences.map((s: string, i: number) => <li key={i}>{s}</li>)}</ol>
        </Card>
        <Callout>The rule layer wins because the errors are systematic: the same few words in the same positions. A learned tagger would need many more labelled domain sentences to see them often enough.</Callout>
        <Why why="Better tags give better lemmas (Simplify → simplify as a verb) and cleaner noun phrases for concept extraction."
          where="Directly on top of the default tagger's output, before lemmatization."
          depends="Hybrid tokens (the fraction rule assumes 3/4 is one token) and the default tagger's tags as the starting point."
          moved="Running the rules before a statistical tagger would be undone by it; running them after lemmatization would be too late for WordNet's POS-dependent lemmas." />
      </Section>
    </div>
  )
}
