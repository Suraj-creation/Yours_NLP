import type { ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { useResult, type Any } from '../lib/api'
import { fmt, runColor } from '../lib/palette'
import { Callout, Card, Loading, PageHeader, Pill, Section } from '../components/ui'

type Row = { stage: string; to: string; chosen: string; rejected: string; evidence: ReactNode; where: string; depends: string; moved: string }

export default function Justification() {
  const c = useResult('corpus'), t = useResult('tokenization'), pre = useResult('preprocessing'), pos = useResult('pos')
  const ner = useResult('ner'), ng = useResult('ngrams'), pl = useResult('pipelines')
  if (!c.data || !t.data || !pre.data || !pos.data || !ner.data || !ng.data || !pl.data) return <Loading />
  const pdf = c.data.extraction.pdf, html = c.data.extraction.html
  const sc = (n: string) => t.data.scores.find((s: Any) => s.tokenizer === n)
  const sw = (n: string) => pre.data.stopwords.find((s: Any) => s.stopword_list === n)
  const st = (n: string) => pre.data.stemmers.find((s: Any) => s.method === n)
  const lm = (n: string) => pre.data.lemmas.find((s: Any) => s.lemmatizer === n)
  const acc = (n: string) => pos.data.accuracy.find((s: Any) => s.tagger === n)
  const th = (n: string) => pl.data.table_h.find((s: Any) => s.pipeline === n)
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`
  const W = pl.data.winner

  const rows: Row[] = [
    { stage: 'Extraction', to: '/corpus', chosen: 'pdfplumber (PDF), trafilatura (HTML), MathML → a/b', rejected: 'pypdf, BeautifulSoup get_text',
      evidence: <>D13: {pct(pdf.find((x: Any) => x.method === 'pdfplumber').dictionary_word_share)} dictionary words vs {pct(pdf.find((x: Any) => x.method === 'pypdf').dictionary_word_share)} for pypdf. D29: {html.find((x: Any) => x.method === 'trafilatura').boilerplate_phrases_found} boilerplate phrases vs {html.find((x: Any) => x.method !== 'trafilatura').boilerplate_phrases_found}.</>,
      where: 'First, per format', depends: 'Nothing', moved: 'Everything downstream inherits broken words and menu text.' },
    { stage: 'Cleaning', to: '/corpus', chosen: 'Ordered steps: vulgar fractions → NFKC → typography → dehyphenate → references → URLs → … → digit-word glue', rejected: 'NFKC first; no cleaning',
      evidence: <>NFKC before the fraction step turns "3½" into "31/2". The digit-word glue step split {fmt(c.data.cleaning.reduce((a: number, r: Any) => a + r.digit_word_glue, 0))} tokens like "1/8pieces".</>,
      where: 'After extraction, before sentence splitting', depends: 'Extracted text', moved: 'Tokens absorb markup and glue; numbers change value.' },
    { stage: 'Tokenization', to: '/tokenization', chosen: 'Hybrid: spaCy tokenizer + protected spans from 14 regex rules', rejected: 'NLTK, spaCy, custom regex alone',
      evidence: <>Gold F1 {sc('hybrid').f1.toFixed(3)} vs NLTK {sc('nltk').f1.toFixed(3)}, spaCy {sc('spacy').f1.toFixed(3)}, custom {sc('custom').f1.toFixed(3)}; runs inside spaCy so POS and NER share its tokens.</>,
      where: 'After sentence splitting', depends: 'Cleaned text', moved: 'Before cleaning: glue errors survive. It also changes NER (MONEY), fixed with ruler patterns.' },
    { stage: 'Stop words', to: '/preprocessing', chosen: `Custom list (${sw('custom').list_size} words) where removal is used; none in ${W}`, rejected: 'NLTK (198), spaCy (326)',
      evidence: <>NLTK removes {sw('nltk').domain_words_removed.split(', ').length} words that matter in maths (not, more, than, over, same…); custom removes none. Mean F1: none {sw('none').mean_f1}, custom {sw('custom').mean_f1}, spaCy {sw('spacy').mean_f1}.</>,
      where: 'After lower-casing, before stemming', depends: 'Tokenizer and lower-casing', moved: 'After stemming, stems like "thi" and "wa" leak (Exercise 11).' },
    { stage: 'Stemming vs lemmatization', to: '/preprocessing', chosen: 'WordNet lemmas with POS', rejected: 'Porter, Snowball, Lancaster, Regexp, WordNet without POS',
      evidence: <>Over-stemming errors: Porter {st('porter').over_stemming_errors}, Lancaster {st('lancaster').over_stemming_errors}, WordNet {st('wordnet_nopos').over_stemming_errors}. Lemma accuracy with POS {pct(lm('wordnet_pos').accuracy)} vs {pct(lm('wordnet_nopos').accuracy)} without.</>,
      where: 'After POS tagging, before indexing', depends: 'The POS tagger', moved: 'Without POS, verbs stay inflected; stemming merges numerator with numerical.' },
    { stage: 'Order', to: '/preprocessing', chosen: 'Stop words before stemming (A); norm then stop only with lemmas (H)', rejected: 'Stem first (C1)',
      evidence: <>C1 leaks {pre.data.order[1].stop_word_stems_leaked} stop-word stems, {fmt(pre.data.order[1].leaked_occurrences)} extra index tokens, with the same F1 ({pre.data.order[1].mean_f1}).</>,
      where: '—', depends: '—', moved: 'Measured directly in C1.' },
    { stage: 'POS tagging', to: '/custom-pos', chosen: 'Rules + dictionary over NLTK/spaCy', rejected: 'Default taggers, backoff chain, CRF',
      evidence: <>Held-out accuracy: rules {pct(acc('rules_nltk').accuracy_test)}, spaCy {pct(acc('spacy').accuracy_test)}, NLTK {pct(acc('nltk').accuracy_test)}, CRF {pct(acc('crf').accuracy_test)}, backoff {pct(acc('backoff').accuracy_test)} (97 tokens).</>,
      where: 'On original-case tokens, before stop-word removal', depends: 'Hybrid tokens', moved: 'After stop-word removal the tagger loses context.' },
    { stage: 'NER', to: '/ner', chosen: 'EntityRuler (lexicon + patterns) before spaCy NER', rejected: 'spaCy model alone; ruler after the model',
      evidence: <>F1 {ner.data.model.overall.f1.toFixed(3)} → {ner.data.ruler.overall.f1.toFixed(3)}; MONEY {ner.data.model.by_type.MONEY.f1} → {ner.data.ruler.by_type.MONEY.f1}; five domain types from 0 to {['CONCEPT', 'KT_MODEL', 'DATASET', 'METRIC', 'MISCONCEPTION'].map((k) => ner.data.ruler.by_type[k].f1.toFixed(2)).join(', ')}.</>,
      where: 'Inside the spaCy pipeline, on original case', depends: 'Lexicon, hybrid tokens', moved: 'After the model, BKT is claimed as ORG first.' },
    { stage: 'N-grams', to: '/ngrams', chosen: 'Meaningful edge filter on lemma streams, within sentences', rejected: 'Stop-word removal before counting',
      evidence: <>Removal first invents {fmt(ng.data.dictionary[1].invented_by_removal)} bigrams and {fmt(ng.data.dictionary[2].invented_by_removal)} trigrams never seen in the text.</>,
      where: 'On final terms', depends: 'Sentence boundaries, lemmatizer', moved: 'Across sentences, n-grams join unrelated turns.' },
    { stage: 'Index terms', to: '/pipelines', chosen: `Pipeline ${W}: ${th(W).description}`, rejected: 'A, H, C1–C3 (H2 ineligible)',
      evidence: <>Mean F1 {th(W).f1.toFixed(4)} (highest); A {th('A').f1.toFixed(4)}, H {th('H').f1.toFixed(4)}, C3 (BPE) {th('C3').f1.toFixed(4)}; post-hoc H2 {th('H2').f1.toFixed(4)}.</>,
      where: 'Last', depends: 'All earlier steps', moved: 'Different index, different results; see Evaluation.' },
  ]

  return (
    <div>
      <PageHeader eyebrow="Module 3 · Justification" title="Why each step is where it is"
        lead="Every decision below is backed by a number from this run; the numbers update if the experiments are re-run. The brief's four questions for each step — why it is needed, where it sits, what it depends on and what happens if it moves — are answered in the last three columns and in the expandable panel on each page." />

      <Card pad={false}>
        <div className="scroll-thin overflow-x-auto">
          <table className="w-full min-w-[1000px] text-left text-[13px]">
            <thead className="bg-surface-2 text-xs text-ink-3">
              <tr>{['Stage', 'Chosen', 'Rejected', 'Evidence', 'Where', 'Depends on', 'If moved'].map((h) => <th key={h} className="px-3 py-2.5 font-semibold">{h}</th>)}</tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.stage} className="border-t border-line align-top">
                  <td className="px-3 py-3"><Link to={r.to} className="font-semibold text-accent-ink hover:underline">{r.stage}</Link></td>
                  <td className="px-3 py-3 text-ink">{r.chosen}</td>
                  <td className="px-3 py-3 text-ink-3">{r.rejected}</td>
                  <td className="px-3 py-3 leading-relaxed text-ink-2">{r.evidence}</td>
                  <td className="px-3 py-3 text-ink-2">{r.where}</td>
                  <td className="px-3 py-3 text-ink-2">{r.depends}</td>
                  <td className="px-3 py-3 text-ink-2">{r.moved}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Section title="The final pipeline, in order">
        <div className="flex flex-wrap items-center gap-2 text-sm">
          {['Extract (per format)', 'Clean (ordered steps)', 'Split sentences (lines hard)', `Tokenize (${th(W).tokenizer})`, 'Lower-case',
            ...(th(W).order === 'stop_then_norm' ? [`Stop words (${th(W).stopwords})`, `Normalise (${th(W).normalizer})`] : [`Normalise (${th(W).normalizer})`, `Stop words (${th(W).stopwords})`]),
            'Positional index', 'Boolean + phrase query', 'tf-idf ranking'].map((s, i, a) => (
            <span key={s} className="inline-flex items-center gap-2">
              <span className="rounded-xl border border-line bg-surface px-3 py-2 text-ink shadow-[var(--shadow)]">{s}</span>
              {i < a.length - 1 && <span className="text-ink-3">→</span>}
            </span>
          ))}
        </div>
        <p className="text-sm text-ink-2">{th(W).normalizer === 'wordnet_pos' ? 'WordNet needs a POS tag, so each sentence is tagged on its original-case tokens before lower-casing. ' : ''}In parallel, the linguistic branch keeps the original case for POS tagging, the custom tagger and NER, which feed the evidence preview.</p>
      </Section>

      <Section title="Honest reporting">
        <Card>
          <ul className="space-y-3 text-sm leading-relaxed text-ink-2">
            <li className="flex gap-2"><span className="h-2.5 w-2.5 shrink-0 translate-y-1.5 rounded-full" style={{ background: runColor('H') }} /><span><b className="text-ink">The hypothesis lost.</b> H, designed to be best, scored {th('H').f1.toFixed(4)} against {th(W).f1.toFixed(4)}. It is reported as a negative result, with its error analysis on the Pipelines page.</span></li>
            <li className="flex gap-2"><span className="h-2.5 w-2.5 shrink-0 translate-y-1.5 rounded-full" style={{ background: runColor('H2') }} /><span><b className="text-ink">Post-hoc fixes are labelled.</b> H2 was built after reading H's errors and evaluated on the same queries, so its {th('H2').f1.toFixed(4)} is optimistic; it is <Pill tone="warn">not eligible</Pill> for selection.</span></li>
            <li className="flex gap-2"><span className="h-2.5 w-2.5 shrink-0 translate-y-1.5 rounded-full bg-ink-3" /><span><b className="text-ink">Small gold sets.</b> 40 tokenization sentences, 32 POS sentences (12 test), 61 NER sentences and 15 queries. Differences of a point or two are within noise and are described as such.</span></li>
            <li className="flex gap-2"><span className="h-2.5 w-2.5 shrink-0 translate-y-1.5 rounded-full bg-ink-3" /><span><b className="text-ink">Synthetic data is marked.</b> D25–D30 were written for this project to cover missing formats; they are labelled in the catalogue and in every table.</span></li>
          </ul>
        </Card>
        <Callout>The selection rule — {pl.data.selection_rule} — was fixed before any pipeline was run.</Callout>
      </Section>
    </div>
  )
}
