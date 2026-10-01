/* Reusable ECharts option builders following the dataviz mark specs:
   bars <= 24px with 4px rounded data-ends, 2px lines, >= 8px markers, hairline solid grids,
   one axis per chart, legend for >= 2 series, tooltips on every mark. */
import type { EChartsOption } from 'echarts'
import { baseAxis, legend, tooltip } from './Chart'
import { chrome, fmt } from '../lib/palette'

export type Series = { name: string; data: (number | null)[]; color: string }

export function bars(o: {
  categories: string[]; series: Series[]; horizontal?: boolean; valueName?: string; stack?: boolean
  percent?: boolean; max?: number; digits?: number; grid?: Record<string, number>; labels?: boolean; categoryWidth?: number
  itemColors?: string[]; logValue?: boolean; rotate?: number
}): EChartsOption {
  const c = chrome()
  const vfmt = (v: number) => (o.percent ? `${(v * 100).toFixed(o.digits ?? 0)}%` : fmt(v, o.digits ?? 0))
  const valueAxis = {
    type: (o.logValue ? 'log' : 'value') as 'value', name: o.valueName, nameLocation: 'end' as const, max: o.max,
    ...baseAxis({ axisLabel: { color: c.ink3, fontSize: 11, formatter: (v: number) => vfmt(v) } }),
    nameTextStyle: { color: c.ink3, fontSize: 11, align: (o.horizontal ? 'right' : 'left') as 'left' }, nameGap: o.horizontal ? 4 : 10,
  }
  const catAxis = {
    type: 'category' as const, data: o.categories,
    ...baseAxis({ splitLine: { show: false },
      axisLabel: { color: c.ink2, fontSize: 11, interval: 0, width: o.categoryWidth ?? 120, overflow: 'truncate', hideOverlap: false, rotate: o.rotate ?? 0 } }),
  }
  const n = o.series.length
  const radius = o.horizontal ? [0, 4, 4, 0] : [4, 4, 0, 0]
  return {
    // leave a row for the legend and, on vertical bars, another for the unit name above the axis
    grid: { left: 8, right: 24, top: (n > 1 ? 36 : 16) + (o.valueName && !o.horizontal ? 18 : 0), bottom: 8, containLabel: true, ...o.grid },
    legend: n > 1 ? legend() : undefined,
    tooltip: tooltip({ trigger: 'axis', axisPointer: { type: 'shadow', shadowStyle: { color: c.line, opacity: 0.35 } },
      valueFormatter: (v: unknown) => vfmt(Number(v)) }),
    xAxis: (o.horizontal ? valueAxis : catAxis) as never,
    yAxis: (o.horizontal ? { ...catAxis, inverse: true } : valueAxis) as never,
    series: o.series.map((s) => ({
      name: s.name, type: 'bar', stack: o.stack ? 'total' : undefined,
      data: o.itemColors && n === 1 ? s.data.map((v, i) => ({ value: v, itemStyle: { color: o.itemColors![i] } })) : s.data,
      barMaxWidth: 24, barGap: '12%', barCategoryGap: n > 2 ? '28%' : '40%',
      itemStyle: { color: s.color, borderRadius: o.stack ? 0 : radius, borderColor: c.surface, borderWidth: o.stack ? 1 : 0 },
      emphasis: { focus: 'series' },
      label: o.labels && n === 1 ? { show: true, position: o.horizontal ? 'right' : 'top', color: c.ink2, fontSize: 11,
        formatter: (p: { value: number }) => vfmt(p.value) } : undefined,
    })) as never,
  }
}

export function lines(o: {
  xs: (number | string)[]; series: Series[]; xName?: string; yName?: string; logX?: boolean; logY?: boolean
  yPercent?: boolean; area?: boolean; digits?: number; xValue?: boolean; markers?: boolean
}): EChartsOption {
  const c = chrome()
  const n = o.series.length
  const yf = (v: number) => (o.yPercent ? `${Math.round(v * 100)}%` : fmt(v, o.digits ?? 0))
  return {
    grid: { left: 8, right: 28, top: n > 1 ? 40 : 20, bottom: 8, containLabel: true },
    legend: n > 1 ? legend() : undefined,
    tooltip: tooltip({ trigger: 'axis', axisPointer: { type: 'line', lineStyle: { color: c.lineStrong } }, valueFormatter: (v: unknown) => yf(Number(v)) }),
    xAxis: {
      type: o.logX ? 'log' : o.xValue ? 'value' : 'category', data: o.logX || o.xValue ? undefined : o.xs, name: o.xName,
      nameLocation: 'middle', nameGap: 28, boundaryGap: false as never, ...baseAxis({ splitLine: { show: false } }),
    } as never,
    yAxis: { type: o.logY ? 'log' : 'value', name: o.yName, ...baseAxis({ axisLabel: { color: c.ink3, fontSize: 11, formatter: yf } }) } as never,
    series: o.series.map((s) => ({
      name: s.name, type: 'line', showSymbol: o.markers ?? false, symbolSize: 8, smooth: false,
      data: o.logX || o.xValue ? s.data.map((y, i) => [o.xs[i], y]) : s.data,
      lineStyle: { width: 2, color: s.color, cap: 'round', join: 'round' }, itemStyle: { color: s.color, borderColor: c.surface, borderWidth: 2 },
      areaStyle: o.area ? { color: s.color, opacity: 0.1 } : undefined, emphasis: { focus: 'series' },
    })) as never,
  }
}

export function heatmap(o: {
  xLabels: string[]; yLabels: string[]; data: [number, number, number][]; max?: number; min?: number; digits?: number; showValues?: boolean
  xRotate?: number; visual?: boolean; showZero?: boolean
}): EChartsOption {
  const c = chrome()
  const max = o.max ?? Math.max(1, ...o.data.map((d) => d[2]))
  const min = o.min ?? 0
  // Text on a cell must stay readable: light text on the dark end of the ramp, ink on the light end.
  const dark = document.documentElement.dataset.theme === 'dark' ||
    (!document.documentElement.dataset.theme && window.matchMedia?.('(prefers-color-scheme: dark)').matches)
  const labelColor = (v: number) => {
    const t = (v - min) / Math.max(1e-9, max - min)
    return dark ? (t > 0.7 ? '#10233f' : c.ink) : (t > 0.55 ? '#ffffff' : c.ink)
  }
  return {
    grid: { left: 8, right: 16, top: 8, bottom: o.visual ? 48 : 8, containLabel: true },
    tooltip: tooltip({ formatter: (p: { data: { value: [number, number, number] } }) =>
      `<b>${fmt(p.data.value[2], o.digits ?? 0)}</b><br/><span style="color:${c.ink3}">${o.yLabels[p.data.value[1]]} × ${o.xLabels[p.data.value[0]]}</span>` }),
    xAxis: { type: 'category', data: o.xLabels, splitArea: { show: false },
      ...baseAxis({ splitLine: { show: false }, axisLabel: { color: c.ink3, fontSize: 10, rotate: o.xRotate ?? 0, interval: 0 } }) } as never,
    yAxis: { type: 'category', data: o.yLabels, inverse: true, ...baseAxis({ splitLine: { show: false }, axisLabel: { color: c.ink2, fontSize: 11, interval: 0 } }) } as never,
    visualMap: { min, max, show: !!o.visual, orient: 'horizontal', left: 'center', bottom: 0, itemHeight: 140, itemWidth: 10, calculable: false,
      text: [fmt(max, o.digits ?? 0), fmt(min, o.digits ?? 0)],
      textStyle: { color: c.ink3, fontSize: 10 }, inRange: { color: c.seq.slice(0, 8) } } as never,
    series: [{
      type: 'heatmap', itemStyle: { borderColor: c.surface, borderWidth: 2, borderRadius: 3 },
      data: o.data.map((d) => ({ value: d, label: { color: labelColor(d[2]) } })),
      label: { show: o.showValues ?? false, fontSize: 10,
        formatter: (p: { data: { value: [number, number, number] } }) => (p.data.value[2] || o.showZero ? fmt(p.data.value[2], o.digits ?? 0) : '') },
      emphasis: { itemStyle: { borderColor: c.ink, borderWidth: 1 } },
    }] as never,
  }
}

export function scatter(o: {
  series: { name: string; data: [number, number, string?][]; color: string; size?: number }[]; xName?: string; yName?: string
  logX?: boolean; logY?: boolean; digits?: number
}): EChartsOption {
  const c = chrome()
  const n = o.series.length
  return {
    grid: { left: 8, right: 28, top: n > 1 ? 40 : 20, bottom: 24, containLabel: true },
    legend: n > 1 ? legend() : undefined,
    tooltip: tooltip({ trigger: 'item', formatter: (p: { seriesName: string; data: [number, number, string?] }) =>
      `<b>${p.data[2] ?? p.seriesName}</b><br/>${o.xName ?? 'x'}: ${fmt(p.data[0], o.digits ?? 0)}<br/>${o.yName ?? 'y'}: ${fmt(p.data[1], o.digits ?? 0)}` }),
    xAxis: { type: o.logX ? 'log' : 'value', name: o.xName, nameLocation: 'middle', nameGap: 28, ...baseAxis({ splitLine: { show: false } }) } as never,
    yAxis: { type: o.logY ? 'log' : 'value', name: o.yName, ...baseAxis() } as never,
    series: o.series.map((s) => ({
      name: s.name, type: 'scatter', data: s.data, symbolSize: s.size ?? 8,
      itemStyle: { color: s.color, borderColor: c.surface, borderWidth: 1.5, opacity: 0.9 }, emphasis: { focus: 'series' },
    })) as never,
  }
}

/** Least-squares fit of y = a * x^b in log-log space. */
export function powerFit(pts: [number, number][]) {
  const L = pts.filter(([x, y]) => x > 0 && y > 0).map(([x, y]) => [Math.log(x), Math.log(y)])
  const n = L.length
  const mx = L.reduce((a, p) => a + p[0], 0) / n, my = L.reduce((a, p) => a + p[1], 0) / n
  const b = L.reduce((a, p) => a + (p[0] - mx) * (p[1] - my), 0) / L.reduce((a, p) => a + (p[0] - mx) ** 2, 0)
  return { a: Math.exp(my - b * mx), b }
}
