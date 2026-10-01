import { useEffect, useState } from 'react'
import { usePost, useResult, type Any } from '../lib/api'
import { fmt, slot } from '../lib/palette'
import Chart from '../components/Chart'
import { bars, lines } from '../components/charts'
import { Button, Callout, Card, DataTable, DownloadLink, ErrorBox, Examples, Grid, Loading, PageHeader, Pill, Section, TextArea, Why } from '../components/ui'
import { TokenChips } from '../components/nlp'

const EXAMPLES = [
  'denominators numerators misconception LLM-based scaffolding',
  'The LCD of 1/4 and 1/6 is 12, so 3/12 + 2/12 = 5/12.',
  'idk why u flip the second fraction when dividing',
  'Bayesian knowledge tracing with MathDial and ASSIST2009',
]
const METHODS = [
  { key: 'gpt2', label: 'gpt2' }, { key: 'cl100k_base', label: 'cl100k_base' }, { key: 'o200k_base', label: 'o200k_base' },
  { key: 'ours_500', label: 'ours, 500 merges' }, { key: 'ours_1000', label: 'ours, 1,000' }, { key: 'ours_2000', label: 'ours, 2,000' },
]
const show = (p: string) => p.replace(/^ /, '▁').replace(/ /g, '▁')

export default function Bpe() {
  const r = useResult('bpe')
  const sc = useResult('scale')
  const run = usePost('/api/bpe')
  const [text, setText] = useState(EXAMPLES[0])
  const [merges, setMerges] = useState(1000)
  const go = (s = text, m = merges) => run.mutate({ text: s, merges: m })
  useEffect(() => { go() }, []) // eslint-disable-line react-hooks/exhaustive-deps
  if (!r.data) return <Loading />
  const b = r.data
  const encs = ['gpt2', 'cl100k_base', 'o200k_base']

  return (
    <div>
      <PageHeader eyebrow="Module 2 · Exercise 4" title="Byte-pair encoding: learned subwords"
        lead="BPE starts from characters and repeatedly merges the most frequent adjacent pair. A vocabulary learned on web text (tiktoken) and one learned on this corpus split domain words differently; the number of pieces per word is the cost a language model pays." />

      <Card title="Try it" subtitle="Our BPE is trained on the 30 cleaned documents with the number of merges you choose">
        <TextArea value={text} onChange={setText} rows={2} />
        <Examples items={EXAMPLES} onPick={(s) => { setText(s); go(s) }} />
        <div className="mt-4 flex flex-wrap items-center gap-4">
          <label className="flex flex-1 items-center gap-3 text-sm text-ink-2">
            <span className="whitespace-nowrap">Merges <b className="num text-ink">{fmt(merges)}</b></span>
            <input type="range" min={50} max={4000} step={50} value={merges} onChange={(e) => setMerges(Number(e.target.value))}
              onMouseUp={() => go()} onTouchEnd={() => go()} onKeyUp={() => go()} className="w-full accent-[var(--accent)]" aria-label="Number of merges" />
          </label>
          <Button onClick={() => go()} disabled={run.isPending}>{run.isPending ? 'Training…' : 'Segment'}</Button>
        </div>
        {run.error && <div className="mt-4"><ErrorBox error={run.error} /></div>}
        {run.data && (
          <div className="mt-5 space-y-5">
            <div className="scroll-thin overflow-x-auto rounded-xl border border-line">
              <table className="w-full text-left text-[13px]">
                <thead className="bg-surface-2 text-xs text-ink-3">
                  <tr><th className="px-3 py-2">Word</th><th className="px-3 py-2">Ours ({fmt(run.data.merges)} merges, vocab {fmt(run.data.our_vocab)})</th>{encs.map((e) => <th key={e} className="px-3 py-2">{e}</th>)}</tr>
                </thead>
                <tbody>
                  {run.data.words.map((w: Any, i: number) => (
                    <tr key={i} className="border-t border-line/70 align-top">
                      <td className="px-3 py-2 font-mono font-semibold text-ink">{w.word}</td>
                      <td className="px-3 py-2"><TokenChips tokens={w.ours} /><span className="text-[11px] text-ink-3">{w.ours.length} pieces</span></td>
                      {encs.map((e) => <td key={e} className="px-3 py-2"><TokenChips tokens={w[e].map(show)} /><span className="text-[11px] text-ink-3">{w[e].length} pieces</span></td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="space-y-2">
              <div className="text-xs font-semibold uppercase tracking-wide text-ink-3">Whole text, as each encoding sees it</div>
              {encs.map((e) => (
                <div key={e} className="grid grid-cols-[110px_1fr] items-start gap-3">
                  <span className="pt-0.5 text-xs text-ink-3">{e} · {run.data.sentence[e].length}</span>
                  <TokenChips tokens={run.data.sentence[e].map(show)} />
                </div>
              ))}
            </div>
            <p className="text-xs text-ink-3">▁ marks a leading space (tiktoken keeps the space inside the next piece); &lt;/w&gt; marks the end of a word in our BPE.</p>
          </div>
        )}
      </Card>

      <Section title="Pieces per word, by word class" lead="Ten common, ten rare and ten domain words. All methods keep common words whole. Web-trained encodings handle rare general words far better, but our 2,000-merge vocabulary splits domain words into fewer pieces than any of them.">
        <Card title="Mean pieces per word">
          <Chart height={300} label="Pieces per word by class" option={() => bars({
            categories: b.per_class.map((x: Any) => x.class), digits: 1,
            series: METHODS.map((m, i) => ({ name: m.label, data: b.per_class.map((x: Any) => x[m.key]), color: slot(i + 1) })),
          })} />
        </Card>
      </Section>

      <Section title="Training our BPE" lead="What each merge buys: vocabulary grows by one entry per merge, and the corpus needs fewer tokens.">
        <Grid cols={2}>
          <Card title="Vocabulary size against merges">
            <Chart height={260} label="Vocabulary size curve" option={() => lines({ xs: b.curve.map((x: Any) => x.merges), xValue: true, xName: 'merges', markers: true,
              series: [{ name: 'vocabulary', data: b.curve.map((x: Any) => x.vocabulary_size), color: slot(1) }] })} />
          </Card>
          <Card title="Corpus length against merges" subtitle="tokens needed to encode all 30 documents">
            <Chart height={260} label="Corpus tokens curve" option={() => lines({ xs: b.curve.map((x: Any) => x.merges), xValue: true, xName: 'merges', markers: true,
              series: [{ name: 'corpus tokens', data: b.curve.map((x: Any) => x.corpus_tokens), color: slot(2) }] })} />
          </Card>
        </Grid>
        <Card title="The first 30 merges" subtitle="The most frequent pairs in the corpus become the first vocabulary entries">
          <div className="flex flex-wrap gap-1.5">
            {b.first_merges.map((m: Any) => (
              <span key={m.step} className="inline-flex items-center gap-1.5 rounded-lg border border-line bg-surface-2 px-2 py-1 text-xs">
                <span className="num text-ink-3">{m.step}</span><span className="font-mono text-ink">{m.pair}</span><span className="text-ink-3">→</span><span className="font-mono text-accent-ink">{m.merged}</span><span className="num text-ink-3">{fmt(m.count)}</span>
              </span>
            ))}
          </div>
        </Card>
      </Section>

      <Section title="Table G: BPE on 30 words">
        <Card pad={false} title="bpe_results.csv" actions={<DownloadLink file="bpe_results.csv" />}>
          <DataTable rows={b.words} dense cols={[
            { key: 's_no', label: '#', num: true }, { key: 'word', label: 'Word', render: (x) => <b className="font-mono text-ink">{x.word as string}</b> },
            { key: 'class', label: 'Class', render: (x) => <Pill>{x.class as string}</Pill> },
            ...METHODS.map((m) => ({ key: m.key, label: m.label, render: (x: Any) => <span className="font-mono text-[12px]">{x[m.key]} <span className="text-ink-3">({x[`${m.key}_n`]})</span></span> })),
            { key: 'hybrid', label: 'hybrid word', render: (x) => <span className="font-mono text-[12px] text-ink-3">{x.hybrid as string}</span> },
          ]} />
        </Card>
      </Section>

      <Section title="Every tokenizer, one table" lead="Subword methods never meet an unknown word; a word-level vocabulary (SimpleTokenizerV2) maps unseen words to <|unk|>.">
        <Card pad={false} title="bpe_summary.csv" actions={<DownloadLink file="bpe_summary.csv" />}>
          <DataTable rows={b.summary} dense cols={[
            { key: 'method', label: 'Method' }, { key: 'vocabulary_size', label: 'Vocabulary', num: true }, { key: 'corpus_tokens', label: 'Corpus tokens', num: true },
            { key: 'trained_on', label: 'Trained on' },
            { key: 'oov_rate_unseen_text', label: 'OOV on unseen text', num: true, render: (x) => x.oov_rate_unseen_text === null ? '—' : `${((x.oov_rate_unseen_text as number) * 100).toFixed(1)}%` },
          ]} />
        </Card>
        <Callout>SimpleTokenizerV2 built from the corpus: "{b.v2_example.text}" decodes to "{b.v2_example.decoded}". The word "obviously" never occurs in the corpus, so it is lost; any BPE would spell it from pieces.</Callout>
        {sc.data && (
          <Card title="At scale" subtitle="Tokens per word on MathDial + TalkMoves (see Heavy datasets)">
            <Chart height={240} label="BPE at scale" option={() => bars({
              horizontal: true, categories: sc.data.bpe.map((x: Any) => x.method.replace('our byte-level BPE, ', 'ours · ').replace('tiktoken ', '')), labels: true, digits: 3, categoryWidth: 150,
              series: [{ name: 'tokens per word', data: sc.data.bpe.map((x: Any) => x.tokens_per_word), color: slot(1) }],
            })} />
          </Card>
        )}
        <Why why="BPE is how every current language model reads text. Comparing it with word tokenizers shows what a later LLM-based learner model would see."
          where="A parallel branch: BPE pieces are analysed and compared, and pipeline C3 indexes cl100k pieces to test them for retrieval."
          depends="Cleaned text. Our BPE lower-cases and pre-splits on whitespace and punctuation, as in the lab."
          moved="Used as index terms (C3), BPE pieces hurt retrieval: the same word splits differently with and without a leading space or capital, so query pieces and document pieces often differ (see Pipelines)." />
      </Section>
    </div>
  )
}
