import { useEffect, useState } from 'react'
import { usePost, useResult, type Any } from '../lib/api'
import { slot } from '../lib/palette'
import Chart from '../components/Chart'
import { bars } from '../components/charts'
import { Button, Callout, Card, DataTable, DownloadLink, ErrorBox, Examples, Grid, Loading, PageHeader, Pill, Section, TextArea, Why } from '../components/ui'
import { Highlight, Metrics, entityColor, type Ent } from '../components/nlp'

const EXAMPLES = [
  'Ms. Priya Raman from Greenfield Public School in Bengaluru bought fraction tiles for Rs 2,500 on 12 August 2026.',
  'We compare BKT, DKT and SQKT on ASSIST2009 and MathDial, reporting AUC and RMSE.',
  'Aarav still adds the numerators and the denominators, the classic add-across misconception about equivalent fractions.',
  'OpenStax and Eedi released the data; GPT-4o graded answers on Google Classroom before the NeurIPS 2025 deadline.',
]
const STATUS_COLOR: Record<string, number> = { correct: 3, incorrect: 4, missed: 8, spurious: 5 }

export default function Ner() {
  const r = useResult('ner')
  const run = usePost('/api/ner')
  const [text, setText] = useState(EXAMPLES[0])
  const go = (s = text) => run.mutate({ text: s })
  useEffect(() => { go() }, []) // eslint-disable-line react-hooks/exhaustive-deps
  if (!r.data) return <Loading />
  const n = r.data
  const types: string[] = [...n.types.standard, ...n.types.domain]
  const m = n.model.overall, u = n.ruler.overall

  return (
    <div>
      <PageHeader eyebrow="Module 2 · Exercise 13" title="Named entities, standard and domain"
        lead={<>spaCy's model finds people, places and dates but knows nothing of <b>concepts</b>, <b>misconceptions</b>, <b>knowledge-tracing models</b>,
          <b> datasets</b> or <b>metrics</b>. An EntityRuler placed before the model adds those five types from a lexicon, plus money and product
          patterns the model misses, and is scored against 161 hand-annotated entities in 61 sentences.</>} />

      <Card title="Try it" subtitle="The same text through the model alone and through the ruler + model">
        <TextArea value={text} onChange={setText} rows={3} />
        <Examples items={EXAMPLES} onPick={(s) => { setText(s); go(s) }} />
        <div className="mt-3 flex justify-end"><Button onClick={() => go()}>Find entities</Button></div>
        {run.error && <ErrorBox error={run.error} />}
        {run.data && (
          <Grid cols={2} className="mt-4">
            {(['model', 'ruler'] as const).map((k) => (
              <div key={k} className="rounded-xl border border-line p-4">
                <div className="mb-2 flex items-center justify-between text-sm">
                  <span className="font-medium text-ink">{k === 'model' ? 'spaCy en_core_web_sm' : 'EntityRuler + spaCy'}</span>
                  <span className="text-ink-3">{run.data[k].length} entities</span>
                </div>
                <Highlight text={run.data.text} ents={run.data[k] as Ent[]} />
              </div>
            ))}
          </Grid>
        )}
        <div className="mt-4 flex flex-wrap gap-3 text-xs text-ink-3">
          {types.map((t) => <span key={t} className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: entityColor(t) }} />{t}</span>)}
        </div>
      </Card>

      <Section title="Scores against the gold annotations" lead="Exact span and type. Incorrect means the right span with the wrong type; spurious means a span with no gold entity.">
        <Metrics items={[
          { label: 'Model F1', value: m.f1 }, { label: 'Ruler + model F1', value: u.f1, note: `+${(u.f1 - m.f1).toFixed(3)}` },
          { label: 'Model recall', value: m.recall }, { label: 'Ruler recall', value: u.recall },
          { label: 'Model precision', value: m.precision }, { label: 'Ruler precision', value: u.precision },
        ]} />
        <Grid cols={2}>
          <Card title="F1 by entity type">
            <Chart height={400} label="NER F1 by type" option={() => bars({
              horizontal: true, categories: types, max: 1, digits: 2,
              series: [
                { name: 'spaCy model', data: types.map((t) => n.model.by_type[t]?.f1 ?? 0), color: slot(1) },
                { name: 'EntityRuler + model', data: types.map((t) => n.ruler.by_type[t]?.f1 ?? 0), color: slot(3) },
              ],
            })} />
          </Card>
          <Card title="What happened to each gold entity" subtitle="EntityRuler + model">
            <Chart height={400} label="NER outcome by type" option={() => bars({
              horizontal: true, stack: true, categories: types,
              series: (['correct', 'incorrect', 'missed', 'spurious'] as const).map((s) => ({ name: s, data: types.map((t) => n.ruler.by_type[t]?.[s] ?? 0), color: slot(STATUS_COLOR[s]) })),
            })} />
          </Card>
        </Grid>
        <Card pad={false} title="ner_scores.csv" actions={<DownloadLink file="ner_scores.csv" />}>
          <DataTable dense rows={types.map((t) => ({ type: t, domain: n.types.domain.includes(t), gold: n.ruler.by_type[t]?.gold, ...Object.fromEntries(['precision', 'recall', 'f1'].flatMap((k) => [[`m_${k}`, n.model.by_type[t]?.[k]], [`r_${k}`, n.ruler.by_type[t]?.[k]]])) }))} cols={[
            { key: 'type', label: 'Type', render: (x) => <span className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-full" style={{ background: entityColor(x.type as string) }} />{x.type as string}{x.domain ? <Pill tone="accent">domain</Pill> : null}</span> },
            { key: 'gold', label: 'Gold', num: true },
            { key: 'm_precision', label: 'Model P', num: true }, { key: 'm_recall', label: 'Model R', num: true }, { key: 'm_f1', label: 'Model F1', num: true },
            { key: 'r_precision', label: 'Ruler P', num: true }, { key: 'r_recall', label: 'Ruler R', num: true }, { key: 'r_f1', label: 'Ruler F1', num: true },
          ]} />
        </Card>
        <Callout tone="warn">EVENT stays at 0: "ACL 2025" is read as a DATE, "Parent Maths Workshop" as a PERSON and "Fraction Fair 2026" matches the concept "fraction", and the lexicon has no event patterns. PERSON is untouched by the ruler: the model misses or mistypes single Indian first names (Aarav as ORG, Rohan as GPE), which a name list would fix.</Callout>
      </Section>

      <Section title="Tokenization changes NER" lead="The hybrid tokenizer keeps $3,650 and Rs 2,500 as single tokens. spaCy's statistical model was trained on its own tokens, so it stops recognising those spans; the ruler's MONEY patterns restore them. This is a dependency between Modules 2 and 13 worth knowing.">
        <Card pad={false} title="ner_tokenization_dependency.csv" actions={<DownloadLink file="ner_tokenization_dependency.csv" />}>
          <DataTable rows={n.dependency} dense cols={[
            { key: 'entity', label: 'Gold entity', render: (x) => <b className="font-mono text-ink">{x.entity as string}</b> },
            { key: 'spacy_default_tokens', label: 'spaCy tokens + model', render: (x) => <span className="font-mono text-[12px]">{x.spacy_default_tokens as string}</span> },
            { key: 'hybrid_tokens_model_only', label: 'Hybrid tokens + model', render: (x) => <span className="font-mono text-[12px]">{x.hybrid_tokens_model_only as string}</span> },
            { key: 'hybrid_tokens_plus_ruler', label: 'Hybrid tokens + ruler', render: (x) => <span className="font-mono text-[12px] text-good-ink">{x.hybrid_tokens_plus_ruler as string}</span> },
          ]} />
        </Card>
      </Section>

      <Section title="Sample sentences">
        {n.samples.map((s: Any, i: number) => (
          <Card key={i}>
            <Grid cols={2}>
              <div><div className="mb-1 text-xs font-semibold uppercase tracking-wide text-ink-3">Model</div><Highlight text={s.text} ents={s.model} /></div>
              <div><div className="mb-1 text-xs font-semibold uppercase tracking-wide text-ink-3">Ruler + model</div><Highlight text={s.text} ents={s.ruler} /></div>
            </Grid>
          </Card>
        ))}
      </Section>

      <Section title="Table E: every gold entity">
        <Card pad={false} title="ner_results.csv" actions={<DownloadLink file="ner_results.csv" />}>
          <DataTable rows={n.table_e} dense cols={[
            { key: 'sentence', label: 'Sentence' }, { key: 'entity', label: 'Entity', render: (x) => <b className="text-ink">{x.entity as string}</b> },
            { key: 'gold_type', label: 'Gold type' },
            { key: 'predicted_type_model', label: 'Model' }, { key: 'status_model', label: 'Model result', render: (x) => <Pill tone={x.status_model === 'correct' ? 'good' : x.status_model === 'spurious' ? 'warn' : 'bad'}>{x.status_model as string}</Pill> },
            { key: 'predicted_type_ruler', label: 'Ruler' }, { key: 'status_ruler', label: 'Ruler result', render: (x) => <Pill tone={x.status_ruler === 'correct' ? 'good' : x.status_ruler === 'spurious' ? 'warn' : 'bad'}>{x.status_ruler as string}</Pill> },
          ]} />
        </Card>
        <Why why="Entities tie a learner's sentence to the concept and misconception documents and are the 'concepts' field of the evidence record."
          where="After tokenization, inside the same spaCy pipeline; the ruler runs before the statistical NER so its spans take priority."
          depends="The domain lexicon (config/domain_lexicon.csv), the hybrid tokenizer, and case: model names and acronyms match on exact case, concepts on lower case with plurals."
          moved="Putting the ruler after the model lets the model claim BKT as ORG first; running NER on lower-cased or lemmatized text removes the capitals the model relies on." />
      </Section>
    </div>
  )
}
