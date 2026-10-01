import { Component, StrictMode, Suspense, lazy, type ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import './index.css'
import Layout, { PipelineProvider } from './components/Layout'
import { Loading } from './components/ui'

const Home = lazy(() => import('./pages/Home'))
const Corpus = lazy(() => import('./pages/Corpus'))
const Statistics = lazy(() => import('./pages/Statistics'))
const Scale = lazy(() => import('./pages/Scale'))
const Tokenization = lazy(() => import('./pages/Tokenization'))
const Preprocessing = lazy(() => import('./pages/Preprocessing'))
const Bpe = lazy(() => import('./pages/Bpe'))
const Pos = lazy(() => import('./pages/Pos'))
const CustomPos = lazy(() => import('./pages/CustomPos'))
const Ner = lazy(() => import('./pages/Ner'))
const Ngrams = lazy(() => import('./pages/Ngrams'))
const IndexPage = lazy(() => import('./pages/IndexPage'))
const SearchPage = lazy(() => import('./pages/SearchPage'))
const Pipelines = lazy(() => import('./pages/Pipelines'))
const Evaluation = lazy(() => import('./pages/Evaluation'))
const Justification = lazy(() => import('./pages/Justification'))
const Evidence = lazy(() => import('./pages/Evidence'))
const Downloads = lazy(() => import('./pages/Downloads'))

/** One broken page must not blank the whole site: show the error in place of the page. */
class Boundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null }
  static getDerivedStateFromError(error: Error) { return { error } }
  render() {
    if (!this.state.error) return this.props.children
    return (
      <div className="rounded-2xl border border-bad/30 bg-bad/5 p-6 text-sm text-ink-2">
        <div className="font-semibold text-bad">This page failed to render.</div>
        <pre className="mt-2 whitespace-pre-wrap font-mono text-xs">{this.state.error.message}</pre>
      </div>
    )
  }
}

const qc = new QueryClient({ defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } } })

const routes: [string, React.ComponentType][] = [
  ['/', Home], ['/corpus', Corpus], ['/statistics', Statistics], ['/scale', Scale], ['/tokenization', Tokenization],
  ['/preprocessing', Preprocessing], ['/bpe', Bpe], ['/pos', Pos], ['/custom-pos', CustomPos], ['/ner', Ner],
  ['/ngrams', Ngrams], ['/index', IndexPage], ['/search', SearchPage], ['/pipelines', Pipelines],
  ['/evaluation', Evaluation], ['/justification', Justification], ['/evidence', Evidence], ['/downloads', Downloads],
]

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={qc}>
      <PipelineProvider>
        <BrowserRouter>
          <Routes>
            <Route element={<Layout />}>
              {routes.map(([path, C]) => (
                <Route key={path} path={path} element={<Boundary key={path}><Suspense fallback={<Loading />}><C /></Suspense></Boundary>} />
              ))}
              <Route path="*" element={<div className="py-20 text-ink-3">Page not found.</div>} />
            </Route>
          </Routes>
        </BrowserRouter>
      </PipelineProvider>
    </QueryClientProvider>
  </StrictMode>,
)
