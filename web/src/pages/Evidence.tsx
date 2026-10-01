import { useEffect, useState } from 'react'
import { usePost, type Any } from '../lib/api'
import { slot } from '../lib/palette'
import { Button, Callout, Card, ErrorBox, Examples, Grid, Loading, PageHeader, Pill, Section, Segmented, TextArea } from '../components/ui'
import { Bar, JsonView, TokenChips } from '../components/nlp'

const EXAMPLES = [
  'I did 1/2 + 1/3 = 2/5 because you add the tops and then add the bottoms.',
  '1/8 is bigger than 1/6 because 8 is bigger than 6.',
  '0.25 is bigger than 0.3 because 25 is more than 3.',
  '2/3 is equal to 4/5 because I added 2 to the top and 2 to the bottom.',
  '0.4 × 0.2 = 0.8, you just multiply the numbers.',
  'The common denominator of 1/4 and 1/6 is 12, so I got 5/12.',
]

export default function Evidence() {
  const run = usePost('/api/evidence')
  const [text, setText] = useState(EXAMPLES[0])
  const [view, setView] = useState('card')
  const go = (s = text) => run.mutate({ text: s })
  useEffect(() => { go() }, []) // eslint-disable-line react-hooks/exhaustive-deps
  const e: Any = run.data

  return (
    <div>
      <PageHeader eyebrow="Research bridge · preview only" title="From a learner's sentence to a candidate evidence record"
        lead="What the research needs from this assessment is a structured observation: the learner's words, what the pipeline extracted, and which misconception documents they resemble. This page assembles that record from the modules above. It is a prototype, not a diagnosis." />
      <Callout tone="warn">No learner model, probability or mastery estimate is computed here. Detectors are transparent arithmetic and wording checks, and grounding is tf-idf retrieval over 55 MaE misconceptions and 11 synthetic malrules. Every record is marked for human review.</Callout>

      <Card className="mt-5" title="A learner explanation">
        <TextArea value={text} onChange={setText} rows={2} />
        <Examples items={EXAMPLES} onPick={(s) => { setText(s); go(s) }} />
        <div className="mt-3 flex justify-end"><Button onClick={() => go()}>Build record</Button></div>
      </Card>

      {run.isPending && <Loading />}
      {run.error && <div className="mt-4"><ErrorBox error={run.error} /></div>}
      {e && !run.isPending && (
        <Section title={`Record ${e.evidence_id}`} lead={e.status}>
          <div className="flex justify-end"><Segmented value={view} onChange={setView} options={[{ value: 'card', label: 'Readable' }, { value: 'json', label: 'JSON' }]} /></div>
          {view === 'json' ? <JsonView data={e} /> : (
            <>
              <Grid cols={2}>
                <Card title="1 · Observation" subtitle={`channel: ${e.observation.channel}`}>
                  <p className="text-[15px] text-ink">{e.observation.cleaned_text}</p>
                  <div className="mt-3"><TokenChips tokens={e.extracted.tokens} /></div>
                </Card>
                <Card title="2 · Extracted">
                  <dl className="space-y-3 text-sm">
                    <div><dt className="text-xs font-semibold uppercase tracking-wide text-ink-3">Numbers and expressions</dt>
                      <dd className="mt-1 flex flex-wrap gap-1.5">{e.extracted.numbers_and_expressions.length ? e.extracted.numbers_and_expressions.map((n: Any, i: number) => <Pill key={i}><span className="font-mono">{n.text}</span><span className="text-ink-3">{n.rule}</span></Pill>) : <span className="text-ink-3">none</span>}</dd></div>
                    <div><dt className="text-xs font-semibold uppercase tracking-wide text-ink-3">Concepts (EntityRuler)</dt>
                      <dd className="mt-1 flex flex-wrap gap-1.5">{e.extracted.concepts.length ? e.extracted.concepts.map((n: Any, i: number) => <Pill key={i} tone="accent">{n.text ?? n}</Pill>) : <span className="text-ink-3">none</span>}</dd></div>
                    <div><dt className="text-xs font-semibold uppercase tracking-wide text-ink-3">Verb and object</dt>
                      <dd className="mt-1 flex flex-wrap gap-1.5">{e.extracted.verb_object.length ? e.extracted.verb_object.map((v: Any, i: number) => <Pill key={i} tone="accent"><b>{v.verb}</b> → {v.object}</Pill>) : <span className="text-ink-3">none</span>}</dd></div>
                    {e.extracted.other_entities.length > 0 && <div><dt className="text-xs font-semibold uppercase tracking-wide text-ink-3">Other entities</dt>
                      <dd className="mt-1 flex flex-wrap gap-1.5">{e.extracted.other_entities.map((n: Any, i: number) => <Pill key={i}>{n.text} · {n.label}</Pill>)}</dd></div>}
                  </dl>
                </Card>
              </Grid>
              <Card title="3 · Detectors" subtitle="Each detector checks the arithmetic or the wording and names a hypothesis; strength is a fixed prior per detector, not a probability">
                {e.interpretation.detectors.length === 0 ? <p className="text-sm text-ink-3">No detector fired. The retrieved grounding below is the only signal.</p> : (
                  <div className="space-y-3">
                    {e.interpretation.detectors.map((d: Any, i: number) => (
                      <div key={i} className="grid gap-2 rounded-xl border border-line p-3 sm:grid-cols-[1fr_180px] sm:items-center">
                        <div>
                          <div className="font-medium text-ink">{d.hypothesis}</div>
                          <div className="mt-0.5 text-xs text-ink-3">{d.detector} · <span className="font-mono">{d.span}</span> · {d.check}</div>
                        </div>
                        <div className="flex items-center gap-2"><Bar value={d.strength} color={slot(1)} /><span className="num w-8 text-xs text-ink-2">{d.strength.toFixed(1)}</span></div>
                      </div>
                    ))}
                  </div>
                )}
              </Card>
              <Card title="4 · Retrieved grounding" subtitle="Nearest misconception descriptions by tf-idf cosine">
                <div className="space-y-2.5">
                  {e.interpretation.retrieved_grounding.map((g: Any) => (
                    <div key={g.id} className="grid gap-2 rounded-xl border border-line p-3 sm:grid-cols-[1fr_180px] sm:items-center">
                      <div>
                        <div className="flex flex-wrap items-center gap-2"><span className="font-mono text-xs font-semibold text-accent-ink">{g.id}</span><span className="text-sm font-medium text-ink">{g.name}</span><Pill tone={g.source.includes('real') ? 'good' : 'warn'}>{g.source}</Pill><span className="text-xs text-ink-3">{g.doc_id}</span></div>
                        <div className="mt-1 font-mono text-[12px] text-ink-3">{g.example}</div>
                      </div>
                      <div className="flex items-center gap-2"><Bar value={g.score} max={Math.max(...e.interpretation.retrieved_grounding.map((x: Any) => x.score))} color={slot(3)} /><span className="num w-10 text-xs text-ink-2">{g.score.toFixed(3)}</span></div>
                    </div>
                  ))}
                </div>
              </Card>
              <Card title="5 · Provenance and review">
                <div className="flex flex-wrap gap-1.5 text-xs">
                  {Object.entries(e.provenance as Record<string, string>).map(([k, v]) => <Pill key={k}>{k}: {String(v)}</Pill>)}
                  <Pill tone="warn">review: {String(e.review_status)}</Pill>
                </div>
              </Card>
            </>
          )}
        </Section>
      )}
    </div>
  )
}
