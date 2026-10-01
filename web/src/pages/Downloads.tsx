import { Download } from 'lucide-react'
import { fileUrl, useGet, type Any } from '../lib/api'
import { fmt } from '../lib/palette'
import { Card, ErrorBox, Loading, PageHeader, Pill, Section } from '../components/ui'

/* The brief's results/ folder: these 14 files are required; the rest are supporting evidence. */
const REQUIRED: Record<string, string> = {
  'preprocessing_results.csv': 'Table A · before/after preprocessing (Ex 1, 7–11)',
  'tokenization_comparison.csv': 'Table B · four tokenizers on domain problem sentences (Ex 2, 3)',
  'stemming_lemmatization.csv': 'Table C · stems and lemmas of 30 domain words (Ex 8–10)',
  'pos_tagging_results.csv': 'Table D · default vs custom POS (Ex 5, 12)',
  'ner_results.csv': 'Table E · every gold entity, model and ruler (Ex 13)',
  'unigram_results.csv': 'Table F · top unigrams (Ex 14)',
  'bigram_results.csv': 'Table F · top bigrams (Ex 14)',
  'trigram_results.csv': 'Table F · top trigrams (Ex 14)',
  'ngram_results.csv': 'Table F · n = 1–5 summary and 4-/5-grams (Ex 14)',
  'bpe_results.csv': 'Table G · BPE segmentations (Ex 4)',
  'pipeline_comparison.csv': 'Table H · pipeline comparison (Module 3)',
  'retrieval_results.csv': 'Table I · retrieved documents per query (Module 4, 5)',
  'evaluation_results.csv': 'Table I · P, R, F1 per query (Module 6)',
  'inverted_index.json': 'Positional inverted index of the selected pipeline (Module 4)',
}
const EXTRA: Record<string, string> = {
  'overall_performance.csv': 'Table J · overall performance', 'table_h_brief_format.csv': 'Table H in the brief\'s exact columns',
  'table_j_brief_format.csv': 'Table J in the brief\'s exact columns', 'document_catalog.csv': 'Corpus catalogue with sources and licences',
  'document_statistics.csv': 'Per-document statistics (Module 1)', 'cleaning_log.csv': 'Cleaning steps fired per document',
  'extraction_comparison.csv': 'PDF and HTML extractor comparison', 'tokenizer_scores.csv': 'Tokenizer P/R/F1 on the gold set',
  'tokenizer_corpus_counts.csv': 'Tokens and dictionary per tokenizer', 'tokenizer_audit.csv': 'Tokenizer failure audit',
  'tokenizer_rule_ablation.csv': 'Custom-rule ablation', 'numbers_dates.csv': 'Numbers and dates kept whole (Ex 6)',
  'bpe_summary.csv': 'Every tokenizer: vocabulary, corpus tokens, OOV', 'preprocessing_stages.csv': 'Stage-by-stage counts',
  'stopword_comparison.csv': 'Stop-list comparison (Ex 7)', 'stemmer_scores.csv': 'Stemmer over/under errors (Ex 8)',
  'stemmer_errors.csv': 'Every stemming error', 'order_experiment.csv': 'Stop/stem order (Ex 11)', 'lemma_scores.csv': 'Lemmatizer accuracy (Ex 10)',
  'pos_domain_terms.csv': 'Domain terms and their default tags', 'pos_tagger_accuracy.csv': 'POS tagger accuracy', 'ex5_verbs_nounphrases.csv': 'Exercise 5 output',
  'ner_scores.csv': 'NER P/R/F1 per type', 'ner_tokenization_dependency.csv': 'How tokenization changes NER', 'collocations.csv': 'PMI and log-likelihood collocations',
  'pipeline_stages.csv': 'Vocabulary per stage per pipeline', 'scale/scale_summary.csv': 'Heavy datasets: sizes', 'scale/scale_pipelines.csv': 'Heavy datasets: index build',
  'scale/scale_queries.csv': 'Heavy datasets: queries', 'scale/scale_bpe.csv': 'Heavy datasets: BPE',
}

function Row({ f, note, required }: { f: Any; note?: string; required?: boolean }) {
  return (
    <a href={fileUrl(f.name)} className="group flex items-center gap-3 border-b border-line/70 px-4 py-2.5 last:border-0 hover:bg-surface-2">
      <Download size={15} className="shrink-0 text-ink-3 group-hover:text-accent-ink" />
      <span className="font-mono text-[13px] text-ink">{f.name}</span>
      {required && <Pill tone="accent">required</Pill>}
      <span className="hidden flex-1 truncate text-[13px] text-ink-3 sm:block">{note}</span>
      <span className="num ml-auto text-xs text-ink-3">{f.bytes > 1e6 ? `${(f.bytes / 1e6).toFixed(1)} MB` : `${fmt(f.bytes / 1024, 1)} KB`}</span>
    </a>
  )
}

export default function Downloads() {
  const files = useGet('/api/files')
  if (files.error) return <ErrorBox error={files.error} />
  if (!files.data) return <Loading />
  const all: Any[] = files.data
  const req = Object.keys(REQUIRED).map((n) => all.find((f) => f.name === n) ?? { name: n, bytes: 0, missing: true })
  const rest = all.filter((f) => !(f.name in REQUIRED))
  const missing = req.filter((f) => f.missing)

  return (
    <div>
      <PageHeader eyebrow="Submission" title="Results files"
        lead="Everything the notebook writes to results/. The fourteen files the brief lists are marked; the others are the evidence behind each page of this site." />
      <Card pad={false} title={`Required by the brief · ${req.length - missing.length} of ${req.length} present`}>
        {req.map((f) => <Row key={f.name} f={f} note={REQUIRED[f.name]} required />)}
      </Card>
      <Section title="Supporting files">
        <Card pad={false}>{rest.map((f) => <Row key={f.name} f={f} note={EXTRA[f.name]} />)}</Card>
      </Section>
    </div>
  )
}
