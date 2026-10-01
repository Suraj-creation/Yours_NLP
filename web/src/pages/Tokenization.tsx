import { useEffect, useState } from 'react'
import { useMeta, usePost, useResult, type Any } from '../lib/api'
import { TOKENIZER_SLOT, fmt, slot } from '../lib/palette'
import Chart from '../components/Chart'
import { bars, heatmap } from '../components/charts'
import { Button, Callout, Card, DataTable, DownloadLink, ErrorBox, Examples, Grid, Loading, PageHeader, Pill, Section, TextArea, Why } from '../components/ui'
import { MultiPills, TokenChips } from '../components/nlp'

const EXAMPLES = [
  'I did 1/2+1/3=2/5 because you add the tops and then add the bottoms.',
  '3 1/2 is the same as 7/2, idk why its 3/2 on the worksheet.',
  'The fees rose from Rs 2,500 to $3,650 (a 46% increase) on 2026-09-01.',
  'Our LLM-based knowledge-tracing model beats BKT and DKT on ASSIST2009.',
  'Round 3.456 to 2 d.p. and write the ratio 3:5 as a fraction.',
]
const TYPE_LABEL: Record<string, string> = {
  fraction: 'Fractions', mixed_number: 'Mixed numbers', expression: 'Expressions', currency_percent: 'Currency & %',
  decimal_ratio_date: 'Decimals, ratios, dates', hyphenated: 'Hyphenated', acronym: 'Acronyms', informal: 'Informal',
}

export default function Tokenization() {
  const meta = useMeta()
  const res = useResult('tokenization')
  const run = usePost('/api/tokenize')
  const [text, setText] = useState(EXAMPLES[0])
  const [sel, setSel] = useState(['nltk', 'spacy', 'custom', 'hybrid'])
  const [rules, setRules] = useState<string[] | null>(null)
  useEffect(() => { run.mutate({ text: EXAMPLES[0], tokenizers: ['nltk', 'spacy', 'custom', 'hybrid'] }) }, []) // eslint-disable-line react-hooks/exhaustive-deps
  if (!res.data || !meta.data) return <Loading />
  const t = res.data
  const allRules: string[] = meta.data.rules
  const go = (s = text) => run.mutate({ text: s, tokenizers: sel.length ? sel : ['hybrid'], rules })
  const scores: Any[] = t.scores
  const types = Object.keys(TYPE_LABEL).filter((k) => `f1_${k}` in scores[0])
  const ref = run.data?.tokenizers.find((x: Any) => x.name === 'hybrid') ?? run.data?.tokenizers.at(-1)
  const refSpans = new Set(ref?.tokens.map((x: Any) => `${x.start}:${x.end}`) ?? [])

  return (
    <div>
      <PageHeader eyebrow="Module 2 · Exercises 2, 3 and 6" title="Tokenization for learner mathematics"
        lead={<>General tokenizers split <b>1/2</b>, <b>3 1/2</b>, <b>$3,650</b> and <b>LLM-based</b> in different, often wrong, places. Four
          tokenizers are compared on the corpus and scored against 40 hand-segmented sentences; the hybrid design keeps spaCy's tokenizer and
          adds protected spans from the custom rules.</>} />

      <Card title="Try it" subtitle="Tokens that differ from the hybrid tokenizer's spans are highlighted">
        <TextArea value={text} onChange={setText} rows={3} />
        <Examples items={EXAMPLES} onPick={(s) => { setText(s); run.mutate({ text: s, tokenizers: sel, rules }) }} />
        <div className="mt-4 flex flex-wrap items-end gap-4">
          <div>
            <div className="mb-1.5 text-xs font-medium text-ink-3">Tokenizers</div>
            <MultiPills value={sel} onChange={setSel} options={meta.data.tokenizers.map((x: Any) => ({ value: x.name, label: x.label }))} />
          </div>
          <Button onClick={() => go()} className="ml-auto">Tokenize</Button>
        </div>
        <details className="mt-4 rounded-xl border border-line px-4 py-3 text-sm">
          <summary className="cursor-pointer text-ink-2">Custom rules used by the custom and hybrid tokenizers ({rules ? rules.length : allRules.length} of {allRules.length} on)</summary>
          <div className="mt-3"><MultiPills value={rules ?? allRules} onChange={(v) => setRules(v.length === allRules.length ? null : v)} options={allRules.map((r) => ({ value: r, label: r.replace(/_/g, ' ') }))} /></div>
          <p className="mt-2 text-xs text-ink-3">Switch a rule off and tokenize again to see what it protects. The same switch-off is measured on the gold set in the ablation chart below.</p>
        </details>
        {run.error && <div className="mt-4"><ErrorBox error={run.error} /></div>}
        {run.data && (
          <div className="mt-5 space-y-4">
            {run.data.tokenizers.map((x: Any) => (
              <div key={x.name}>
                <div className="mb-1.5 flex items-center gap-2 text-sm">
                  <span className="h-2.5 w-2.5 rounded-full" style={{ background: slot(TOKENIZER_SLOT[x.name] ?? 8) }} />
                  <span className="font-medium text-ink">{x.label}</span><span className="text-ink-3">· {x.count} tokens</span>
                </div>
                <TokenChips tokens={x.tokens.map((k: Any) => k.text)} mark={(_, i) => x.name !== 'hybrid' && !refSpans.has(`${x.tokens[i].start}:${x.tokens[i].end}`)} />
              </div>
            ))}
            {run.data.rule_hits.length > 0 && (
              <div className="flex flex-wrap items-center gap-1.5 border-t border-line pt-3 text-xs text-ink-3">
                Rules that matched:
                {run.data.rule_hits.map((h: Any, i: number) => <Pill key={i}>{h.rule} · <span className="font-mono">{h.text}</span></Pill>)}
              </div>
            )}
          </div>
        )}
      </Card>

      <Section title="Exercise 3: scored against hand-segmented gold" lead="40 sentences (651 gold tokens) chosen for the hard cases, each tagged with a problem type. Token-level precision, recall and F1 over exact spans.">
        <Grid cols={2}>
          <Card title="F1 on the gold sentences">
            <Chart height={320} label="Tokenizer F1" option={() => bars({
              horizontal: true, categories: scores.map((s) => s.label), labels: true, digits: 3, max: 1,
              series: [{ name: 'F1', data: scores.map((s) => s.f1), color: slot(1) }],
              itemColors: scores.map((s) => slot(TOKENIZER_SLOT[s.tokenizer] ?? 8)), categoryWidth: 150,
            })} />
          </Card>
          <Card title="F1 by problem type" subtitle="stronger colour is better (scale starts at 0.6); custom is near 1 partly by construction, since gold spans follow its rules">
            <Chart height={320} label="F1 by problem type heatmap" option={() => heatmap({
              xLabels: types.map((k) => TYPE_LABEL[k]), yLabels: scores.map((s) => s.label), min: 0.6, max: 1, digits: 2, showValues: true, xRotate: 30, visual: true,
              data: scores.flatMap((s, y) => types.map((k, x) => [x, y, s[`f1_${k}`]] as [number, number, number])),
            })} />
          </Card>
        </Grid>
        <Card pad={false} title="tokenizer_scores.csv" actions={<DownloadLink file="tokenizer_scores.csv" />}>
          <DataTable rows={scores} dense cols={[
            { key: 'label', label: 'Tokenizer' }, { key: 'precision', label: 'P', num: true }, { key: 'recall', label: 'R', num: true },
            { key: 'f1', label: 'F1', num: true }, { key: 'tp', label: 'TP', num: true }, { key: 'fp', label: 'FP', num: true }, { key: 'fn', label: 'FN', num: true },
            { key: 'sentence_exact_match', label: 'Sentences exact', num: true },
          ]} initialSort={{ key: 'f1', dir: -1 }} />
        </Card>
      </Section>

      <Section title="Which rule earns its place" lead="Each custom rule removed in turn from the hybrid tokenizer, re-scored on the gold set. Seven rules show no drop: spaCy's own tokenizer already keeps a lone 1/2, 3:5 or 1st whole, and URLs and emails are not in the gold set. They stay because the standalone custom tokenizer needs them.">
        <Card title="F1 drop when one rule is removed">
          <Chart height={380} label="Rule ablation" option={() => bars({
            horizontal: true, categories: t.ablation.map((a: Any) => a.rule_removed), labels: true, digits: 4,
            series: [{ name: 'F1 drop', data: t.ablation.map((a: Any) => a.f1_drop), color: slot(4) }],
          })} />
        </Card>
      </Section>

      <Section title="Exercise 2: on the whole corpus">
        <Grid cols={2}>
          <Card title="Dictionary size by tokenizer" subtitle="distinct token types over all 30 documents">
            <Chart height={300} label="Dictionary size" option={() => bars({
              horizontal: true, categories: t.corpus.map((c: Any) => c.label), categoryWidth: 150,
              series: [{ name: 'dictionary size', data: t.corpus.map((c: Any) => c.dictionary_size), color: slot(1) }],
              itemColors: t.corpus.map((c: Any) => slot(TOKENIZER_SLOT[c.tokenizer] ?? 8)), labels: true,
            })} />
          </Card>
          <Card pad={false} title="tokenizer_corpus_counts.csv" actions={<DownloadLink file="tokenizer_corpus_counts.csv" />}>
            <DataTable rows={t.corpus} dense cols={[
              { key: 'label', label: 'Tokenizer' }, { key: 'tokens', label: 'Tokens', num: true }, { key: 'dictionary_size', label: 'Dictionary', num: true },
              { key: 'dictionary_lowercased', label: 'Lower-cased', num: true }, { key: 'punctuation_tokens', label: 'Punctuation', num: true },
            ]} />
          </Card>
        </Grid>
        <Card pad={false} title="Where the general tokenizers fail" subtitle="Problems found by scanning the corpus, with the rule that fixes each" actions={<DownloadLink file="tokenizer_audit.csv" />}>
          <DataTable rows={t.audit} dense cols={[
            { key: 'tokenizer', label: 'Tokenizer' }, { key: 'problem', label: 'Problem' }, { key: 'occurrences', label: 'Occurrences', num: true },
            { key: 'examples', label: 'Examples', render: (r) => <span className="font-mono text-[12px]">{r.examples as string}</span> }, { key: 'fix', label: 'Fix' },
          ]} />
        </Card>
      </Section>

      <Section title="Table B: domain problem sentences" lead="The eight sentences in the brief's Table B format. Tokens that the hybrid tokenizer does not produce are highlighted.">
        <div className="space-y-3">
          {t.table_b.map((r: Any) => (
            <Card key={r.id} title={<span className="flex items-center gap-2"><span className="font-mono text-ink-3">{r.id}</span> <Pill>{r.problem_type}</Pill></span>}>
              <p className="mb-3 text-[15px] text-ink">{r.input}</p>
              <div className="space-y-2">
                {['nltk', 'spacy', 'custom', 'hybrid'].map((k) => (
                  <div key={k} className="grid grid-cols-[80px_1fr] items-start gap-3">
                    <span className="pt-0.5 text-xs font-medium text-ink-3">{k}</span>
                    <TokenChips tokens={String(r[k]).split(' | ')} mark={(tk) => k !== 'hybrid' && !String(r.hybrid).split(' | ').includes(tk)} />
                  </div>
                ))}
              </div>
            </Card>
          ))}
        </div>
        <DownloadLink file="tokenization_comparison.csv" />
      </Section>

      <Section title="Exercise 6: numbers and dates" lead="Share of each kind of number kept as one token (exact span match), across the whole corpus.">
        <Card pad={false} title="numbers_dates.csv" actions={<DownloadLink file="numbers_dates.csv" />}>
          <DataTable rows={t.numbers_dates} dense cols={[
            { key: 'kind', label: 'Kind' }, { key: 'occurrences', label: 'Occurrences', num: true },
            { key: 'examples', label: 'Examples', render: (r) => <span className="font-mono text-[11.5px] text-ink-3">{String(r.examples).slice(0, 90)}</span> },
            ...['nltk', 'spacy', 'custom', 'hybrid'].map((k) => ({ key: `kept_whole_${k}`, label: k, num: true, render: (r: Any) => `${(r[`kept_whole_${k}`] * 100).toFixed(0)}%` })),
          ]} />
        </Card>
        <Why why="Every later stage counts tokens: if 1/2 becomes three tokens, the index cannot find the fraction, the POS tagger sees a SYM between numbers and NER cannot see a money amount."
          where="First step after cleaning and sentence splitting. The hybrid tokenizer runs inside spaCy so POS, lemmas and NER use the same tokens."
          depends="Cleaned text with vulgar fractions expanded (3½ → 3 1/2) and digit-word glue split (1/8pieces → 1/8 pieces)."
          moved={<>Tokenizing before cleaning keeps the glue errors. Protecting spans also changes NER: once <span className="font-mono">$3,650</span> is one token, spaCy's model no longer tags it, so the EntityRuler adds MONEY patterns (see Named entities).</>} />
        <Callout>Custom and hybrid share the same rules, so both are near 1 on the rule-defined types; the gold spans were written with those rules in mind, which favours them. Hybrid adds spaCy's exceptions, which lifts acronyms and expressions over custom alone. Overall: hybrid F1 {fmt(scores.find((s) => s.tokenizer === 'hybrid')?.f1, 3)} vs NLTK {fmt(scores.find((s) => s.tokenizer === 'nltk')?.f1, 3)} and spaCy {fmt(scores.find((s) => s.tokenizer === 'spacy')?.f1, 3)}.</Callout>
      </Section>
    </div>
  )
}
