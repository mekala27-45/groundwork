// Hand built rather than a charting library: every chart on this page is
// a small, fixed shape (four metrics, four configs; three categories),
// and this series has followed its own specific dataviz rules since day
// 2 (categorical hues in fixed order never cycled, thin marks, recessive
// gridlines, a table view alongside every chart) closely enough that a
// general purpose library would mostly be fought rather than used. No
// new dependency, no new peer range to keep compatible with the
// TypeScript pin this app already had to fix once.

const CHART_WIDTH = 640;
const CHART_HEIGHT = 300;
const PADDING_LEFT = 40;
const PADDING_BOTTOM = 28;
const PADDING_TOP = 20;
const PLOT_RIGHT = CHART_WIDTH - 12;
const PLOT_HEIGHT = CHART_HEIGHT - PADDING_TOP - PADDING_BOTTOM;

function gridlineTicks(maxValue: number): number[] {
  return [0, 0.25, 0.5, 0.75, 1].map((fraction) => fraction * maxValue);
}

function Gridlines({
  maxValue,
  formatValue,
}: {
  maxValue: number;
  formatValue: (v: number) => string;
}): React.JSX.Element {
  return (
    <>
      {gridlineTicks(maxValue).map((tick) => {
        const y = PADDING_TOP + PLOT_HEIGHT * (1 - tick / maxValue);
        return (
          <g key={tick}>
            <line
              x1={PADDING_LEFT}
              y1={y}
              x2={PLOT_RIGHT}
              y2={y}
              stroke="var(--border-subtle)"
              strokeWidth={1}
              vectorEffect="non-scaling-stroke"
            />
            <text
              x={PADDING_LEFT - 6}
              y={y}
              textAnchor="end"
              dominantBaseline="middle"
              className="fill-[var(--text-muted)] text-[9px]"
            >
              {formatValue(tick)}
            </text>
          </g>
        );
      })}
    </>
  );
}

export interface BarSeries {
  key: string;
  label: string;
  color: string;
}

// Grouped bars: one group per x-axis category (a metric), one bar per
// series within it (a chunking-and-rerank configuration). Direct value
// labels above every bar, since four series is exactly this series' own
// "direct labels at four or fewer" threshold, plus a compact legend so
// the color to config mapping does not have to be memorized.
export function GroupedBarChart({
  groups,
  series,
  values,
  maxValue,
  formatValue = (v) => v.toFixed(2),
  ariaLabel,
}: {
  groups: string[];
  series: BarSeries[];
  values: number[][];
  maxValue: number;
  formatValue?: (v: number) => string;
  ariaLabel: string;
}): React.JSX.Element {
  const plotWidth = PLOT_RIGHT - PADDING_LEFT;
  const groupWidth = plotWidth / groups.length;
  const groupGap = groupWidth * 0.18;
  const barGap = 3;
  const barsAreaWidth = groupWidth - groupGap;
  const barWidth = (barsAreaWidth - barGap * (series.length - 1)) / series.length;

  return (
    <div>
      <svg
        viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
        role="img"
        aria-label={ariaLabel}
        className="w-full"
      >
        <Gridlines maxValue={maxValue} formatValue={formatValue} />
        {groups.map((group, groupIndex) => {
          const groupX = PADDING_LEFT + groupIndex * groupWidth + groupGap / 2;
          return (
            <g key={group}>
              {series.map((s, seriesIndex) => {
                const value = values[groupIndex]?.[seriesIndex] ?? 0;
                const barHeight = maxValue > 0 ? (value / maxValue) * PLOT_HEIGHT : 0;
                const x = groupX + seriesIndex * (barWidth + barGap);
                const y = PADDING_TOP + PLOT_HEIGHT - barHeight;
                return (
                  <g key={s.key}>
                    <rect
                      x={x}
                      y={y}
                      width={barWidth}
                      height={Math.max(barHeight, 0)}
                      rx={2}
                      style={{ fill: s.color }}
                    />
                    {barHeight > 14 && (
                      <text
                        x={x + barWidth / 2}
                        y={y - 4}
                        textAnchor="middle"
                        className="fill-[var(--text-muted)] text-[8px]"
                      >
                        {formatValue(value)}
                      </text>
                    )}
                  </g>
                );
              })}
              <text
                x={groupX + barsAreaWidth / 2}
                y={CHART_HEIGHT - PADDING_BOTTOM + 16}
                textAnchor="middle"
                className="fill-[var(--text-secondary)] text-[10px]"
              >
                {group}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
        {series.map((s) => (
          <div key={s.key} className="flex items-center gap-1.5 text-xs text-[var(--text-secondary)]">
            <span
              className="inline-block h-2 w-2 rounded-sm"
              style={{ backgroundColor: s.color }}
              aria-hidden="true"
            />
            {s.label}
          </div>
        ))}
      </div>
    </div>
  );
}

// A single bar per category, direct labels under and value labels above,
// no legend needed at this count (three categories, well under the four
// series direct label threshold).
export function CategoryBarChart({
  bars,
  maxValue,
  formatValue = (v) => v.toFixed(2),
  ariaLabel,
}: {
  bars: { label: string; value: number; color: string }[];
  maxValue: number;
  formatValue?: (v: number) => string;
  ariaLabel: string;
}): React.JSX.Element {
  const plotWidth = PLOT_RIGHT - PADDING_LEFT;
  const slotWidth = plotWidth / bars.length;
  const barWidth = slotWidth * 0.45;

  return (
    <svg
      viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`}
      role="img"
      aria-label={ariaLabel}
      className="w-full"
    >
      <Gridlines maxValue={maxValue} formatValue={formatValue} />
      {bars.map((bar, index) => {
        const value = bar.value;
        const barHeight = maxValue > 0 ? (value / maxValue) * PLOT_HEIGHT : 0;
        const slotX = PADDING_LEFT + index * slotWidth;
        const x = slotX + (slotWidth - barWidth) / 2;
        const y = PADDING_TOP + PLOT_HEIGHT - barHeight;
        return (
          <g key={bar.label}>
            <rect
              x={x}
              y={y}
              width={barWidth}
              height={Math.max(barHeight, 0)}
              rx={3}
              style={{ fill: bar.color }}
            />
            <text
              x={x + barWidth / 2}
              y={y - 6}
              textAnchor="middle"
              className="fill-[var(--text-primary)] text-[10px] font-medium"
            >
              {formatValue(value)}
            </text>
            <text
              x={slotX + slotWidth / 2}
              y={CHART_HEIGHT - PADDING_BOTTOM + 16}
              textAnchor="middle"
              className="fill-[var(--text-secondary)] text-[10px]"
            >
              {bar.label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
