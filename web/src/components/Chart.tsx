/* ECharts wrapper: one instance per chart, resized with its box, redrawn on theme change.
   The option is built by a function so it can read the live CSS tokens (light or dark). */
import { useEffect, useRef } from 'react'
import * as echarts from 'echarts/core'
import { BarChart, GraphChart, HeatmapChart, LineChart, ScatterChart } from 'echarts/charts'
import { GridComponent, LegendComponent, TooltipComponent, VisualMapComponent } from 'echarts/components'
import { LabelLayout } from 'echarts/features'
import { CanvasRenderer } from 'echarts/renderers'
import type { EChartsOption } from 'echarts'

// Register only what the site draws: keeps the bundle to a fraction of full ECharts.
echarts.use([BarChart, LineChart, ScatterChart, HeatmapChart, GraphChart, GridComponent, TooltipComponent, LegendComponent,
  VisualMapComponent, LabelLayout, CanvasRenderer])
import { chrome } from '../lib/palette'
import { useThemeVersion } from '../lib/theme'

export type OptionFn = () => EChartsOption

export function baseAxis(extra: Record<string, unknown> = {}) {
  const c = chrome()
  return {
    axisLine: { lineStyle: { color: c.lineStrong } },
    axisTick: { show: false },
    axisLabel: { color: c.ink3, fontSize: 11, fontFamily: 'Inter Variable, system-ui' },
    splitLine: { lineStyle: { color: c.line, type: 'solid' as const } },
    nameTextStyle: { color: c.ink3, fontSize: 11 },
    ...extra,
  }
}

export function tooltip(extra: Record<string, unknown> = {}) {
  const c = chrome()
  return {
    backgroundColor: c.surface, borderColor: c.line, borderWidth: 1, padding: [8, 10],
    textStyle: { color: c.ink, fontSize: 12, fontFamily: 'Inter Variable, system-ui' },
    extraCssText: 'box-shadow: 0 4px 16px rgba(0,0,0,.08); border-radius: 8px;',
    ...extra,
  }
}

export function legend(extra: Record<string, unknown> = {}) {
  const c = chrome()
  return { top: 0, left: 0, icon: 'roundRect', itemWidth: 10, itemHeight: 10, itemGap: 14,
    textStyle: { color: c.ink2, fontSize: 12 }, ...extra }
}

export default function Chart({ option, height = 300, className = '', label, onPoint }: {
  option: OptionFn; height?: number; className?: string; label?: string; onPoint?: (p: Record<string, unknown>) => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  const inst = useRef<echarts.ECharts | null>(null)
  const themeV = useThemeVersion()

  useEffect(() => {
    if (!ref.current) return
    inst.current = echarts.init(ref.current, undefined, { renderer: 'canvas' })
    const ro = new ResizeObserver(() => inst.current?.resize())
    ro.observe(ref.current)
    return () => { ro.disconnect(); inst.current?.dispose(); inst.current = null }
  }, [])

  // Build the option every render but only push it to ECharts when its serialisable
  // content (or the theme) changes, so typing elsewhere on a page never re-animates charts.
  const built = option()
  const key = JSON.stringify(built)

  useEffect(() => {
    if (!inst.current) return
    const c = chrome()
    const o = built
    inst.current.setOption({
      backgroundColor: 'transparent',
      textStyle: { fontFamily: 'Inter Variable, system-ui', color: c.ink2 },
      animationDuration: 400,
      ...o,
    }, true)
    const onClick = onPoint
    inst.current.off('click')
    if (onClick) inst.current.on('click', (p) => onClick(p as unknown as Record<string, unknown>))
  }, [key, themeV]) // eslint-disable-line react-hooks/exhaustive-deps

  return <div ref={ref} role="img" aria-label={label} className={`w-full ${className}`} style={{ height }} />
}
