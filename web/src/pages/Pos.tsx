import { useEffect, useState } from 'react'
import clsx from 'clsx'
import { usePost, useResult, type Any } from '../lib/api'
import { slot } from '../lib/palette'
import Chart from '../components/Chart'
import { bars } from '../components/charts'
import { Button, Callout, Card, DataTable, DownloadLink, ErrorBox, Examples, Grid, Loading, PageHeader, Pill, Section, TextArea, Why } from '../components/ui'

const EXAMPLES = [
  'Simplify 12/8 and then add the numerators.',
  'One over two is the same as 1/2 but the top number is smaller.',
  'Meredith thinks 1/8 is bigger than 1/6 because 8 is bigger than 6.',
  'Round 3.456 to the nearest tenth, then subtract 1.2 from it.',
]
const ORDER = ['nltk', 'spacy', 'rules_nltk', 'rules_spacy', 'backoff', 'crf']

export default function Pos() {
  const r = useResult('pos')
  const tag = usePost('/api/pos')
  const syn = usePost('/api/syntax')
  const [text, setText] = useState(EXAMPLES[0])
  const go = (s = text) => { tag.mutate({ text: s }); syn.mutate({ text: s }) }
  useEffect(() => { go() }, []) // eslint-disable-line react-hooks/exhaustive-deps
  if (!r.data) return <Loading />
  const p = r.data
  const dist: Any[] = p.distribution.filter((d: Any) => d.default + d.custom > 0)

  return (
    <div>
      <PageHeader eyebrow="Module 2 · Exercise 5 (and the start of 12)" title="Part-of-speech tagging on classroom language"
        lead={<>Default taggers are trained on newspaper text. In a maths lesson, sentence-initial imperatives (<i>Simplify</i>, <i>Round</i>),
          fraction words (<i>halves</i>), variables (<i>x</i>) and <i>over</i> as a fraction bar are tagged wrongly often enough to matter for lemmas
          and noun phrases.</>} />

      <Card title="Try it" subtitle="Six taggers on the same hybrid tokens. Cells that disagree with the rules-over-NLTK tagger are highlighted.">
        <TextArea value={text} onChange={setText} rows={2} />
        <Examples items={EXAMPLES} onPick={(s) => { setText(s); go(s) }} />
        <div className="mt-3 flex justify-end"><Button onClick={() => go()}>Tag</Button></div>
        {tag.error && <ErrorBox error={tag.error} />}
        {tag.data && tag.data.tokens.length > 0 && (
          <div className="scroll-thin mt-4 overflow-x-auto rounded-xl border border-line">
            <table className="w-full text-left text-[13px]">
              <thead className="bg-surface-2 text-xs text-ink-3">
                <tr><th className="px-3 py-2">Token</th>{ORDER.map((k) => <th key={k} className="px-3 py-2">{tag.data.labels[k]}</th>)}<th className="px-3 py-2">UPOS</th><th className="px-3 py-2">Rule fired</th></tr>
              </thead>
              <tbody>
                {tag.data.tokens.map((t: string, i: number) => {
                  const ref = tag.data.taggers.rules_nltk[i]
                  return (
                    <tr key={i} className="border-t border-line/70">
                      <td className="px-3 py-1.5 font-mono font-medium text-ink">{t}</td>
                      {ORDER.map((k) => {
                        const v = tag.data.taggers[k][i]
                        return <td key={k} className={clsx('px-3 py-1.5 font-mono', v !== ref ? 'bg-accent-soft font-semibold text-accent-ink' : 'text-ink-2')}>{v}</td>
                      })}
                      <td className="px-3 py-1.5 font-mono text-ink-3">{tag.data.upos[i]}</td>
                      <td className="px-3 py-1.5 text-xs text-ink-3">{tag.data.rules_fired[i]}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
        {syn.data && (
          <Grid cols={2} className="mt-5">
            <div>
              <div className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-ink-3">Verb and object (dependency parse)</div>
              <div className="flex flex-wrap gap-1.5">
                {syn.data.verb_object.length === 0 && <span className="text-sm text-ink-3">none found</span>}
                {syn.data.verb_object.map((v: Any, i: number) => <Pill key={i} tone="accent"><b>{v.verb}</b> → {v.object}</Pill>)}
              </div>
            </div>
            <div>
              <div className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-ink-3">Noun phrases</div>
              <div className="flex flex-wrap gap-1.5">{syn.data.noun_phrases.map((n: string, i: number) => <Pill key={i}>{n}</Pill>)}</div>
            </div>
          </Grid>
        )}
      </Card>

      <Section title="Where the default taggers go wrong" lead="Domain terms counted across the corpus with the tags NLTK and spaCy give them; the share of occurrences that are wrong against the expected tag.">
        <Card title="Share of occurrences mistagged">
          <Chart height={420} label="Mistag share per term" option={() => bars({
            horizontal: true, categories: p.identification.map((x: Any) => `${x.term} (${x.expected})`), percent: true, max: 1,
            series: [
              { name: 'NLTK', data: p.identification.map((x: Any) => x.nltk_wrong_share), color: slot(1) },
              { name: 'spaCy', data: p.identification.map((x: Any) => x.spacy_wrong_share), color: slot(2) },
            ],
          })} />
        </Card>
        <Card pad={false} title="pos_domain_terms.csv" actions={<DownloadLink file="pos_domain_terms.csv" />}>
          <DataTable rows={p.identification} dense cols={[
            { key: 'term', label: 'Term', render: (x) => <b className="font-mono text-ink">{x.term as string}</b> }, { key: 'occurrences', label: 'Occurrences', num: true },
            { key: 'expected', label: 'Expected' }, { key: 'nltk_tags', label: 'NLTK tags', render: (x) => <span className="font-mono text-[12px]">{x.nltk_tags as string}</span> },
            { key: 'spacy_tags', label: 'spaCy tags', render: (x) => <span className="font-mono text-[12px]">{x.spacy_tags as string}</span> },
            { key: 'nltk_wrong_share', label: 'NLTK wrong', num: true, render: (x) => `${((x.nltk_wrong_share as number) * 100).toFixed(0)}%` },
            { key: 'spacy_wrong_share', label: 'spaCy wrong', num: true, render: (x) => `${((x.spacy_wrong_share as number) * 100).toFixed(0)}%` },
          ]} />
        </Card>
      </Section>

      <Section title="Tag distribution: default against custom" lead="Counts over a sample of the corpus: the first 40 sentences of each of D01–D12. The visible change is VB: the custom tagger moves sentence-initial imperatives from NN and NNP to VB. NLTK already tags most fractions CD.">
        <Card>
          <Chart height={300} label="Tag distribution" option={() => bars({
            categories: dist.map((d) => d.tag), rotate: 0,
            series: [{ name: 'NLTK default', data: dist.map((d) => d.default), color: slot(1) }, { name: 'Custom (rules over NLTK)', data: dist.map((d) => d.custom), color: slot(3) }],
          })} />
        </Card>
      </Section>

      <Section title="Exercise 5: verbs, objects and noun phrases" lead="The brief asks for the verbs and nouns in instructional and learner sentences. Verb–object pairs come from spaCy's dependency parse over hybrid tokens.">
        <Grid cols={2}>
          {p.ex5.verbs_and_nouns.map((x: Any, i: number) => (
            <Card key={i}>
              <p className="text-[14.5px] text-ink">{x.text}</p>
              <div className="mt-3 flex flex-wrap gap-1.5">{x.verb_object.map((v: Any, j: number) => <Pill key={j} tone="accent"><b>{v.verb}</b> → {v.object}</Pill>)}</div>
              <div className="mt-2 flex flex-wrap gap-1.5">{x.noun_phrases.map((n: string, j: number) => <Pill key={j}>{n}</Pill>)}</div>
            </Card>
          ))}
          {p.ex5.nouns_only.map((x: Any, i: number) => (
            <Card key={`n${i}`}>
              <p className="text-[14.5px] text-ink">{x.text}</p>
              <div className="mt-3 flex flex-wrap gap-1.5">{x.noun_phrases.map((n: string, j: number) => <Pill key={j}>{n}</Pill>)}</div>
            </Card>
          ))}
        </Grid>
        <DownloadLink file="ex5_verbs_nounphrases.csv" />
        <Callout>The verb–object pairs are the most useful output for the research: "add → the tops" and "add → the bottoms" name the procedure a learner used, which the evidence preview turns into a misconception hypothesis.</Callout>
        <Why why="Lemmas need the tag (WordNet), noun phrases give the concepts a sentence is about, and verb–object pairs describe what a learner did."
          where="After tokenization, on the original-case tokens and before stop-word removal, so the tagger has full context."
          depends="The hybrid tokenizer: if 1/2 is split, the tagger sees '1', '/', '2' and tags the slash as a symbol."
          moved="Tagging after stop-word removal or stemming removes the context words and yields non-words ('denomin') the model has never seen." />
      </Section>
    </div>
  )
}
