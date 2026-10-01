import { useMemo, useRef, useState } from 'react'
import { Upload } from 'lucide-react'
import { api, useGet, useResult, type Any } from '../lib/api'
import { fmt } from '../lib/palette'
import { Button, Callout, Card, DataTable, DownloadLink, ErrorBox, Grid, Loading, PageHeader, Pill, Section, Segmented, Select, Stat, Why } from '../components/ui'
import { LayerDot, Metrics } from '../components/nlp'

const LAYER_TEXT: Record<string, string> = {
  concept: 'Concept knowledge (textbook sections)',
  research: 'Research literature and dataset documentation',
  learner: 'Real learner and classroom language (MathDial, TalkMoves, MaE)',
  synthetic: 'Synthetic learner artefacts written for this project, labelled',
}

export default function Corpus() {
  const corpus = useResult('corpus')
  const [doc, setDoc] = useState('D25')
  const [view, setView] = useState<'clean' | 'raw'>('clean')
  const d = useGet(`/api/documents/${doc}?chars=8000`)
  const [up, setUp] = useState<Any>(null)
  const [upErr, setUpErr] = useState<unknown>(null)
  const [busy, setBusy] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  const formats = useMemo(() => {
    if (!corpus.data) return []
    const m: Record<string, number> = {}
    corpus.data.catalog.forEach((r: Any) => { m[r.format] = (m[r.format] ?? 0) + 1 })
    return Object.entries(m).sort((a, b) => b[1] - a[1])
  }, [corpus.data])
  const families = useMemo(() => new Set((corpus.data?.catalog ?? []).map((r: Any) => String(r.format).split(' ')[0])).size, [corpus.data])

  if (!corpus.data) return <Loading />
  const cat: Any[] = corpus.data.catalog
  const layers = ['concept', 'research', 'learner', 'synthetic'].map((l) => ({ l, n: cat.filter((r) => r.layer === l).length }))
  const synthetic = cat.filter((r) => r.synthetic === 'True').length

  const onUpload = async (f: File) => {
    setBusy(true); setUpErr(null)
    try { setUp(await api.upload(f)) } catch (e) { setUpErr(e); setUp(null) } finally { setBusy(false) }
  }

  const cleaningSteps = d.data ? Object.entries(d.data.cleaning as Record<string, number>).filter(([, v]) => v > 0) : []

  return (
    <div>
      <PageHeader eyebrow="Module 1 · Corpus construction and cleaning" title="Thirty documents in ten formats"
        lead={<>The assessment corpus has four layers so that queries can connect a learner's words to concept explanations and to the research
          that names misconceptions. Real, openly licensed sources come first; the six synthetic learner artefacts fill formats the public data
          does not offer (DOCX, chat JSON, a PDF worksheet) and are labelled as synthetic everywhere.</>} />

      <Grid cols={4}>
        <Stat label="Documents" value={cat.length} note={`${families} file formats`} />
        <Stat label="Real sources" value={cat.length - synthetic} note="OpenStax, ACL Anthology, MathDial, TalkMoves, MaE" />
        <Stat label="Synthetic, labelled" value={synthetic} note="D25–D30, with gold labels" />
        <Stat label="Characters after cleaning" value={fmt(corpus.data.summary.characters)} note={`${fmt(corpus.data.summary.sentences)} sentences`} />
      </Grid>

      <Grid cols={4} className="mt-5">
        {layers.map(({ l, n }) => (
          <div key={l} className="rounded-2xl border border-line bg-surface p-4">
            <div className="flex items-center justify-between"><LayerDot layer={l} /><span className="num text-sm font-semibold text-ink">{n}</span></div>
            <p className="mt-2 text-[13px] leading-snug text-ink-3">{LAYER_TEXT[l]}</p>
          </div>
        ))}
      </Grid>

      <Section title="Document catalogue" lead="Every file in data/, with its source and licence. Click a column to sort.">
        <Card pad={false} actions={<DownloadLink file="document_catalog.csv" />} title="data/ — D01 to D30">
          <DataTable rows={cat} dense cols={[
            { key: 'doc_id', label: 'ID', width: '56px', render: (r) => <button className="font-mono text-accent-ink hover:underline" onClick={() => { setDoc(r.doc_id); document.getElementById('viewer')?.scrollIntoView({ behavior: 'smooth' }) }}>{r.doc_id}</button> },
            { key: 'title', label: 'Title' },
            { key: 'format', label: 'Format' },
            { key: 'layer', label: 'Layer', render: (r) => <LayerDot layer={r.layer} /> },
            { key: 'source', label: 'Source' },
            { key: 'licence', label: 'Licence' },
            { key: 'synthetic', label: 'Synthetic', render: (r) => r.synthetic === 'True' ? <Pill tone="warn">synthetic</Pill> : <span className="text-ink-3">real</span> },
          ]} />
        </Card>
      </Section>

      <Section title="Raw and cleaned text" lead="The loader turns each format into plain text; the cleaner then applies an ordered list of steps and logs how often each one fired.">
        <div id="viewer" />
        <Card title={d.data ? `${d.data.doc_id} · ${d.data.title}` : 'Document'} subtitle={d.data ? `${d.data.filename} · ${d.data.format}` : undefined}
          actions={<>
            <Select value={doc} onChange={setDoc} options={cat.map((r) => ({ value: r.doc_id, label: `${r.doc_id} · ${r.format}` }))} />
            <Segmented value={view} onChange={(v) => setView(v as 'clean' | 'raw')} options={[{ value: 'clean', label: 'Cleaned' }, { value: 'raw', label: 'Raw extract' }]} />
          </>}>
          {d.isLoading && <Loading />}
          {d.error && <ErrorBox error={d.error} />}
          {d.data && (
            <>
              <Metrics items={[
                { label: 'Raw characters', value: d.data.raw_length },
                { label: 'Clean characters', value: d.data.clean_length },
                { label: 'Change', value: `${(((d.data.clean_length - d.data.raw_length) / Math.max(1, d.data.raw_length)) * 100).toFixed(1)}%` },
                { label: 'Loader', value: d.data.meta.loader ?? '—' },
                { label: 'Layer', value: d.data.meta.layer ?? '—' },
                { label: 'Steps fired', value: cleaningSteps.length },
              ]} />
              <div className="mt-3 flex flex-wrap gap-1.5">
                {cleaningSteps.length === 0 && <span className="text-xs text-ink-3">No cleaning step changed this document.</span>}
                {cleaningSteps.map(([k, v]) => <Pill key={k}>{k.replace(/_/g, ' ')} · {fmt(v)}</Pill>)}
              </div>
              <pre className="scroll-thin mt-4 max-h-[420px] overflow-auto whitespace-pre-wrap rounded-xl border border-line bg-surface-2 p-4 font-mono text-[12.5px] leading-relaxed text-ink-2">
                {view === 'clean' ? d.data.clean : d.data.raw}
              </pre>
              {(view === 'clean' ? d.data.clean_length : d.data.raw_length) > 8000 && <p className="mt-2 text-xs text-ink-3">Showing the first 8,000 characters.</p>}
            </>
          )}
        </Card>
        <Why why="Tokenizers and indexes see whatever the loader hands them. Markup, page numbers, hyphenated line breaks, vulgar fractions (½) and URLs would each become tokens or break real ones."
          where="Straight after extraction and before sentence splitting, so every later module reads the same cleaned text (one cleaned copy is cached)."
          depends="The right extractor per format: pdfplumber for PDFs, trafilatura for HTML, MathML rendered to a/b before cleaning."
          moved={<>Cleaning after tokenization would leave "3½" and "1/8pieces" as single wrong tokens; running NFKC before the vulgar-fraction step turns "3½" into "31/2", a different number. The step order is fixed in <code className="font-mono">cleaning.py</code> for that reason.</>} />
      </Section>

      <Section title="Choosing an extractor" lead="Two extractors were compared on the formats where extraction quality varies most.">
        <Grid cols={2}>
          {(['pdf', 'html'] as const).map((k) => (
            <Card key={k} title={k === 'pdf' ? 'PDF · D13 TalkMoves coding manual' : 'HTML · D29 class web page'}
              subtitle={k === 'pdf' ? 'Share of words found in a dictionary; higher means fewer broken words' : 'Navigation and cookie phrases that leaked into the text'}>
              <div className="space-y-4">
                {corpus.data.extraction[k].map((r: Any) => (
                  <div key={r.method}>
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-medium text-ink">{r.method}</span>
                      <span className="num text-ink-2">{k === 'pdf' ? `${(r.dictionary_word_share * 100).toFixed(1)}% dictionary words` : `${r.boilerplate_phrases_found} boilerplate phrases · ${fmt(r.words)} words`}</span>
                    </div>
                    <p className="mt-1.5 line-clamp-3 rounded-lg bg-surface-2 p-2.5 font-mono text-[11.5px] leading-relaxed text-ink-3">{r.sample}</p>
                  </div>
                ))}
              </div>
            </Card>
          ))}
        </Grid>
        <Callout>pdfplumber keeps the words of D13 whole ("Talk Moves"), while pypdf breaks them at kerning gaps ("T alk Mo ves"); trafilatura removes the page's menu, cookie banner and footer, which BeautifulSoup's plain text keeps. Both winners are the loaders' defaults.</Callout>
        <div className="flex flex-wrap gap-2"><DownloadLink file="extraction_comparison.csv" /><DownloadLink file="cleaning_log.csv" /></div>
      </Section>

      <Section title="Try your own file" lead="Upload any TXT, PDF, DOCX, HTML, JSON, JSONL, CSV, XLSX, XML or Markdown file (15 MB max). It goes through the same loader, cleaner and hybrid tokenizer.">
        <Card>
          <input ref={fileRef} type="file" className="hidden" onChange={(e) => { const f = e.target.files?.[0]; if (f) onUpload(f); e.target.value = '' }} />
          <Button onClick={() => fileRef.current?.click()} disabled={busy}><Upload size={16} /> {busy ? 'Processing…' : 'Choose a file'}</Button>
          {upErr != null && <div className="mt-4"><ErrorBox error={upErr} /></div>}
          {up && (
            <div className="mt-5 space-y-4">
              <div className="text-sm text-ink-2"><b className="text-ink">{up.filename}</b> · detected {up.format}</div>
              <Metrics items={[
                { label: 'Sentences', value: up.stats.sentences }, { label: 'Tokens', value: up.stats.tokens },
                { label: 'Word tokens', value: up.stats.word_tokens }, { label: 'Vocabulary', value: up.stats.vocabulary },
                { label: 'Type-token ratio', value: up.stats.type_token_ratio, digits: 3 }, { label: 'Characters', value: up.stats.characters },
              ]} />
              <div className="flex flex-wrap gap-1.5">
                {Object.entries(up.cleaning as Record<string, number>).filter(([, v]) => v > 0).map(([k, v]) => <Pill key={k}>{k.replace(/_/g, ' ')} · {v}</Pill>)}
              </div>
              {up.entities?.length > 0 && (
                <div>
                  <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-ink-3">Entities in the first 5,000 characters</div>
                  <div className="flex flex-wrap gap-1.5">{up.entities.map((e: Any, i: number) => <Pill key={i}>{e.text} · {e.label}</Pill>)}</div>
                </div>
              )}
              <pre className="scroll-thin max-h-80 overflow-auto whitespace-pre-wrap rounded-xl border border-line bg-surface-2 p-4 font-mono text-[12.5px] text-ink-2">{up.preview}</pre>
            </div>
          )}
        </Card>
      </Section>

      <Section title="Formats in the corpus">
        <div className="flex flex-wrap gap-2">
          {formats.map(([f, n]) => <Pill key={f}>{f} · {n}</Pill>)}
        </div>
        <Card title="What the loaders keep aside as metadata">
          <p className="text-sm leading-relaxed text-ink-2">MathDial dialogue acts such as <code className="font-mono">(focus)</code> and <code className="font-mono">(probing)</code> and the TalkMoves teacher and student codes are moved to metadata, so the index holds only what was said. OpenStax image alt texts are counted and excluded by default, because they describe pictures rather than teach.</p>
        </Card>
      </Section>
    </div>
  )
}
