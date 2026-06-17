import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {
  Presentation,
  PresentationFile,
} from 'file:///C:/Users/alpbu/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';

const ROOT = 'E:/PROJECTS/GitHub/CSE4078S26_Grp6';
const FINAL_PPTX = path.join(ROOT, 'outputs', 'CSE4078S26_Grp6_FinalPresentation_Qwen3_LoRA.pptx');
const WORK = path.join(os.tmpdir(), 'codex-presentations', 'manual-cse4078-final-qwen3-light');
const TMP = path.join(WORK, 'tmp');
const PREVIEW = path.join(TMP, 'preview');
const LAYOUT = path.join(TMP, 'layout');
const QA = path.join(TMP, 'qa');

await fs.mkdir(PREVIEW, { recursive: true });
await fs.mkdir(LAYOUT, { recursive: true });
await fs.mkdir(QA, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });

const C = {
  bg: '#ffffff',
  ink: '#0f172a',
  muted: '#475569',
  faint: '#64748b',
  line: '#d7dee8',
  line2: '#e7ecf2',
  panel: '#f7f9fc',
  panel2: '#eef3f8',
  navy: '#1e3a5f',
  blue: '#2f6f9f',
  blueSoft: '#e9f2fb',
  green: '#16825d',
  greenSoft: '#e8f6ef',
  red: '#b42338',
  redSoft: '#fdebed',
  amber: '#9a6700',
  amberSoft: '#fff6df',
};

const deck = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const FONT_H = 'Aptos Display';
const FONT_B = 'Aptos';
const PAGE = { left: 56, top: 38, width: 1168, height: 622 };

function shape(slide, geometry, position, fill = 'none', line = { style: 'solid', fill: 'none', width: 0 }, extra = {}) {
  return slide.shapes.add({ geometry, position, fill, line, ...extra });
}

function text(slide, value, x, y, w, h, style = {}) {
  const t = shape(slide, 'textbox', { left: x, top: y, width: w, height: h });
  t.text = value;
  t.text.style = {
    typeface: style.typeface ?? FONT_B,
    fontSize: style.fontSize ?? 18,
    bold: style.bold ?? false,
    color: style.color ?? C.ink,
    alignment: style.alignment ?? 'left',
    verticalAlignment: style.verticalAlignment ?? 'top',
    lineSpacing: style.lineSpacing ?? 1.05,
    autoFit: style.autoFit ?? 'shrinkText',
    insets: style.insets ?? { top: 2, right: 2, bottom: 2, left: 2 },
  };
  return t;
}

function line(slide, x, y, w, color = C.line) {
  return shape(slide, 'line', { left: x, top: y, width: w, height: 0 }, 'none', {
    style: 'solid',
    fill: color,
    width: 1,
  });
}

function card(slide, x, y, w, h, fill = C.panel, stroke = C.line, radius = 8) {
  return shape(slide, 'roundRect', { left: x, top: y, width: w, height: h }, fill, {
    style: 'solid',
    fill: stroke,
    width: 1,
  }, { borderRadius: radius });
}

function header(slide, kicker, title, subtitle, n) {
  slide.background.fill = C.bg;
  shape(slide, 'rect', { left: 0, top: 0, width: 1280, height: 12 }, C.navy, {
    style: 'solid',
    fill: C.navy,
    width: 0,
  });
  text(slide, kicker.toUpperCase(), PAGE.left, 34, 640, 22, {
    fontSize: 11,
    bold: true,
    color: C.blue,
  });
  text(slide, title, PAGE.left, 62, 900, 50, {
    fontSize: 34,
    bold: true,
    color: C.ink,
    typeface: FONT_H,
    lineSpacing: 0.95,
  });
  if (subtitle) {
    text(slide, subtitle, PAGE.left, 118, 930, 30, {
      fontSize: 15,
      color: C.muted,
    });
  }
  footer(slide, n);
}

function footer(slide, n) {
  line(slide, PAGE.left, 674, 1020, C.line2);
  text(slide, 'CSE4078S26 Group 6 | Turkish Legal QA with Small LLMs', PAGE.left, 686, 620, 20, {
    fontSize: 10,
    color: C.faint,
  });
  text(slide, String(n), 1190, 686, 32, 20, {
    fontSize: 10,
    color: C.faint,
    alignment: 'right',
  });
}

function badge(slide, value, x, y, w, tone = 'blue') {
  const fill = tone === 'green' ? C.greenSoft : tone === 'red' ? C.redSoft : tone === 'amber' ? C.amberSoft : C.blueSoft;
  const color = tone === 'green' ? C.green : tone === 'red' ? C.red : tone === 'amber' ? C.amber : C.blue;
  card(slide, x, y, w, 28, fill, C.line, 14);
  text(slide, value, x + 10, y + 6, w - 20, 16, {
    fontSize: 11,
    bold: true,
    color,
    alignment: 'center',
    verticalAlignment: 'middle',
  });
}

function kpi(slide, x, y, w, h, value, label, tone = 'blue') {
  const fill = tone === 'green' ? C.greenSoft : tone === 'red' ? C.redSoft : tone === 'amber' ? C.amberSoft : C.panel;
  const color = tone === 'green' ? C.green : tone === 'red' ? C.red : tone === 'amber' ? C.amber : C.blue;
  card(slide, x, y, w, h, fill, C.line, 8);
  text(slide, value, x + 14, y + 16, w - 28, 46, {
    fontSize: 34,
    bold: true,
    color,
    alignment: 'center',
    typeface: FONT_H,
  });
  text(slide, label, x + 16, y + 66, w - 32, h - 74, {
    fontSize: 12,
    color: C.muted,
    alignment: 'center',
    lineSpacing: 1.05,
  });
}

function table(slide, x, y, w, rowH, colWs, rows, opts = {}) {
  let yy = y;
  for (let r = 0; r < rows.length; r += 1) {
    let xx = x;
    const isHeader = r === 0;
    for (let c = 0; c < colWs.length; c += 1) {
      const cw = colWs[c] * w;
      const fill = isHeader ? C.navy : (r % 2 ? C.bg : C.panel);
      shape(slide, 'rect', { left: xx, top: yy, width: cw, height: rowH }, fill, {
        style: 'solid',
        fill: C.line,
        width: 1,
      });
      text(slide, String(rows[r][c] ?? ''), xx + 8, yy + 7, cw - 16, rowH - 12, {
        fontSize: isHeader ? (opts.headerFont ?? 11) : (opts.font ?? 12),
        bold: isHeader || (opts.boldFirstCol && c === 0),
        color: isHeader ? '#ffffff' : C.ink,
        alignment: opts.align?.[c] ?? 'left',
        lineSpacing: 1.0,
      });
      xx += cw;
    }
    yy += rowH;
  }
}

function miniBar(slide, x, y, label, value, maxValue, color = C.blue) {
  text(slide, label, x, y, 132, 18, { fontSize: 12, color: C.muted });
  shape(slide, 'rect', { left: x + 140, top: y + 4, width: 260, height: 10 }, C.panel2, {
    style: 'solid',
    fill: C.panel2,
    width: 0,
  });
  shape(slide, 'rect', { left: x + 140, top: y + 4, width: 260 * (value / maxValue), height: 10 }, color, {
    style: 'solid',
    fill: color,
    width: 0,
  });
  text(slide, value.toFixed(3), x + 414, y - 1, 56, 18, { fontSize: 12, bold: true, color: C.ink, alignment: 'right' });
}

// 1
{
  const s = deck.slides.add();
  s.background.fill = C.bg;
  shape(s, 'rect', { left: 0, top: 0, width: 1280, height: 12 }, C.navy, { style: 'solid', fill: C.navy, width: 0 });
  text(s, 'CSE4078 SPRING 2026 | FINAL PRESENTATION', 58, 46, 620, 24, { fontSize: 12, bold: true, color: C.blue });
  text(s, 'Fine-Tuning Small LLMs for Turkish Legal Question Answering', 58, 110, 740, 132, {
    fontSize: 48,
    bold: true,
    typeface: FONT_H,
    lineSpacing: 0.94,
  });
  text(s, 'Group 6', 60, 270, 160, 34, { fontSize: 24, bold: true, color: C.navy, typeface: FONT_H });
  text(s, 'Dilan DILEN | Ceren EREN | Zorbey Onur AK | Alp BUYUKKOSE | Kerem Hakki KOC', 60, 310, 850, 26, {
    fontSize: 14,
    color: C.muted,
  });
  text(s, 'Department of Computer Engineering, Marmara University', 60, 338, 680, 24, { fontSize: 13, color: C.faint });
  kpi(s, 58, 414, 250, 124, '+178%', 'ROUGE-1 gain over Qwen3 base', 'green');
  kpi(s, 324, 414, 250, 124, '+384%', 'ROUGE-2 gain over Qwen3 base', 'green');
  kpi(s, 590, 414, 250, 124, '+27.7%', 'BERTScore F1 relative gain', 'green');
  kpi(s, 856, 414, 250, 124, '1,500', 'official unseen test examples', 'blue');
  card(s, 58, 570, 1066, 54, C.panel, C.line, 8);
  text(s, 'Selected model: Qwen/Qwen3-4B-Instruct-2507 with bf16 LoRA. Main message: fine-tuning improves corpus alignment substantially, but legal hallucination remains visible.', 78, 588, 1026, 24, {
    fontSize: 15,
    color: C.ink,
  });
  footer(s, 1);
}

// 2
{
  const s = deck.slides.add();
  header(s, 'Project setup', 'Problem, data, and split isolation', 'The official test split is used only once, after training and model selection are complete.', 2);
  table(s, 82, 184, 1048, 44, [0.30, 0.18, 0.52], [
    ['Split or artifact', 'Examples', 'Role in the project'],
    ['Official training split', '13,354', 'input to preprocessing and SFT only'],
    ['After preprocessing', '11,898', 'cleaned train/validation pool'],
    ['SFT train / validation', '10,709 / 1,189', 'deterministic 90/10 split with seed 42'],
    ['Official test split', '1,500', 'final evaluation only; never used for cleaning or tuning'],
  ], { font: 13, boldFirstCol: true });
  kpi(s, 82, 462, 320, 118, 'No leakage', 'test examples are not used in preprocessing, validation, or checkpoint selection', 'blue');
  kpi(s, 456, 462, 320, 118, 'Seed 42', 'split, training, and reproducible analysis use fixed seeds where applicable', 'blue');
  kpi(s, 830, 462, 300, 118, 'One test set', 'all final before/after scores use the same 1,500 examples', 'blue');
}

// 3
{
  const s = deck.slides.add();
  header(s, 'Baseline selection', 'Why Qwen3-4B-Instruct was selected', 'The model screen used the same 200-example subset and the same BERTScore model.', 3);
  table(s, 54, 182, 748, 50, [0.27, 0.085, 0.085, 0.085, 0.105, 0.105, 0.105, 0.16], [
    ['Model', 'R-1', 'R-2', 'R-L', 'BERT P', 'BERT R', 'BERT F1', 'Decision'],
    ['Instella-3B', '0.022', '0.000', '0.020', '0.508', '0.543', '0.523', 'drop'],
    ['Qwen2.5-3B', '0.180', '0.077', '0.146', '0.615', '0.658', '0.635', 'older'],
    ['Qwen3-4B', '0.230', '0.108', '0.183', '0.653', '0.699', '0.674', 'select'],
    ['Qwen3.5-4B', '0.235', '0.097', '0.168', '0.634', '0.705', '0.667', 'context'],
  ], { font: 10.5, headerFont: 10, boldFirstCol: true, align: ['left', 'center', 'center', 'center', 'center', 'center', 'center', 'center'] });
  card(s, 812, 184, 354, 214, C.panel, C.line, 8);
  text(s, 'Selection logic', 840, 208, 260, 28, { fontSize: 21, bold: true, color: C.navy, typeface: FONT_H });
  text(s, 'Qwen3-4B wins R-2, R-L, BERT precision, and BERT F1. Qwen3.5 is close and slightly higher on R-1 and BERT recall, but it was not the safer LoRA target for this setup.', 840, 250, 292, 86, {
    fontSize: 14,
    color: C.muted,
    lineSpacing: 1.08,
  });
  badge(s, '200-example screen', 930, 350, 150, 'blue');
  card(s, 812, 430, 354, 210, C.blueSoft, C.line, 8);
  text(s, 'Qwen3 vs Qwen3.5 metric histogram', 836, 450, 306, 22, {
    fontSize: 16,
    bold: true,
    color: C.navy,
    alignment: 'center',
  });
  s.charts.add('bar', {
    position: { left: 832, top: 484, width: 314, height: 122 },
    categories: ['R-1', 'R-2', 'R-L', 'BERT P', 'BERT R', 'BERT F1'],
    series: [
      { name: 'Qwen3', values: [0.230, 0.108, 0.183, 0.653, 0.699, 0.674], fill: C.blue },
      { name: 'Qwen3.5', values: [0.235, 0.097, 0.168, 0.634, 0.705, 0.667], fill: '#94a3b8' },
    ],
    barOptions: { direction: 'column', grouping: 'clustered', gapWidth: 45 },
    legend: { position: 'bottom', overlay: false, textStyle: { fill: C.muted, fontSize: 8 } },
    yAxis: {
      min: 0,
      max: 0.75,
      numberFormatCode: '0.0',
      majorGridlines: { style: 'solid', fill: C.line, width: 1 },
      textStyle: { fill: C.muted, fontSize: 8 },
    },
    xAxis: { textStyle: { fill: C.muted, fontSize: 7 } },
    dataLabels: { showValue: false },
    chartFill: C.blueSoft,
    plotAreaFill: C.blueSoft,
    chartLine: { style: 'solid', fill: C.blueSoft, width: 0 },
  });
  text(s, 'Qwen3 leads on 4/6 metrics; Qwen3.5 leads on R-1 and BERT R.', 842, 614, 296, 18, {
    fontSize: 9,
    color: C.muted,
    alignment: 'center',
  });
}

// 4
{
  const s = deck.slides.add();
  header(s, 'Exploratory data analysis', 'Answer lengths justify a 256-token generation cap', 'The final cap reduces truncation without encouraging long, repetitive generations.', 4);
  card(s, 64, 190, 524, 332, C.panel, C.line, 8);
  s.charts.add('bar', {
    position: { left: 94, top: 226, width: 460, height: 236 },
    categories: ['Median', 'Mean', 'p90', 'p95', 'p99', 'Max'],
    series: [{ name: 'Reference answer words', values: [20, 24, 42, 51, 91, 343], fill: C.blue }],
    hasLegend: false,
    barOptions: { direction: 'column', grouping: 'clustered', gapWidth: 55 },
    yAxis: { min: 0, max: 360, majorGridlines: { style: 'solid', fill: C.line, width: 1 }, numberFormatCode: '0', textStyle: { fill: C.muted, fontSize: 11 } },
    xAxis: { textStyle: { fill: C.muted, fontSize: 11 } },
    dataLabels: { showValue: true, position: 'outEnd', textStyle: { fill: C.ink, fontSize: 10, bold: true } },
    chartFill: C.panel,
    plotAreaFill: C.panel,
    chartLine: { style: 'solid', fill: C.panel, width: 0 },
  });
  text(s, 'Reference answer length on the official test set (word proxy)', 94, 476, 420, 22, {
    fontSize: 13,
    bold: true,
    color: C.ink,
  });
  kpi(s, 646, 198, 230, 114, '6.3%', 'references would exceed a 128-token budget', 'red');
  kpi(s, 910, 198, 230, 114, '99.2%', 'references covered by a 256-token budget', 'green');
  kpi(s, 646, 344, 230, 114, '12', 'test references above 256 tokens', 'amber');
  kpi(s, 910, 344, 230, 114, '65', 'mean Qwen-tokenized answer length', 'blue');
  text(s, 'Final evaluation uses max_new_tokens=256 and greedy decoding. The goal is enough room for valid legal answers, not extra verbosity.', 648, 500, 488, 42, {
    fontSize: 15,
    color: C.muted,
    lineSpacing: 1.08,
  });
}

// 5
{
  const s = deck.slides.add();
  header(s, 'Preprocessing', 'Train-only cleaning pipeline', 'The reproduced run uses answer length 6, question cap 2, answer cap 6, semantic threshold 0.955, and cap 4.', 5);
  const stages = [
    ['Raw train', '13,354'],
    ['Rule-based', '12,336'],
    ['Semantic cap', '11,898'],
    ['SFT train', '10,709'],
    ['Validation', '1,189'],
  ];
  stages.forEach(([label, value], i) => {
    const x = 64 + i * 224;
    card(s, x, 208, 168, 82, C.panel, C.line, 8);
    text(s, label, x + 14, 224, 140, 20, { fontSize: 12, color: C.muted, bold: true });
    text(s, value, x + 14, 248, 140, 30, { fontSize: 24, bold: true, color: C.navy, typeface: FONT_H });
    if (i < stages.length - 1) {
      line(s, x + 178, 249, 32, C.line);
      shape(s, 'triangle', { left: x + 208, top: 242, width: 12, height: 14, rotation: 90 }, C.line, { style: 'solid', fill: C.line, width: 0 });
    }
  });
  table(s, 102, 374, 1030, 44, [0.22, 0.27, 0.31, 0.20], [
    ['Stage', 'Action', 'Purpose', 'Parameter'],
    ['Rule-based cleaning', 'normalize, exact dedup, frequency caps', 'remove trivial duplicates and ultra-short answers', 'min answer=6, q<=2, a<=6'],
    ['Embedding', 'embed question + answer pairs', 'find near-duplicate paraphrases', 'Qwen3-Embedding-4B, bf16'],
    ['Semantic cap', 'cluster, split by polarity, cap each group', 'keep opposite conclusions separate', 'threshold=0.955, cap=4'],
  ], { font: 12, boldFirstCol: true });
  badge(s, 'seed=42 split', 1010, 120, 126, 'blue');
}

// 6
{
  const s = deck.slides.add();
  header(s, 'Rule-based cleaning', 'Stage 1 is deterministic cleaning', 'This stage is better described as rule-based cleaning rather than a pure regex step.', 6);
  table(s, 76, 178, 570, 52, [0.58, 0.20, 0.22], [
    ['Step', 'Removed', 'Remaining'],
    ['Raw training split', '-', '13,354'],
    ['Answer shorter than 6 words', '829', '12,525'],
    ['Exact question-answer duplicate', '4', '12,521'],
    ['Same-question cap above 2', '2', '12,519'],
    ['Same-answer cap above 6', '183', '12,336'],
  ], { font: 13, boldFirstCol: true, align: ['left', 'center', 'center'] });
  card(s, 724, 190, 410, 176, C.amberSoft, C.line, 8);
  text(s, 'Important correction', 750, 216, 330, 28, { fontSize: 21, bold: true, color: C.amber, typeface: FONT_H });
  text(s, 'The final model data is reproduced by 6 / 2 / 6. The later script default 7 / 2 / 5 gives 12,084 rows, so it is not the final training run.', 750, 260, 340, 72, {
    fontSize: 14,
    color: C.ink,
    lineSpacing: 1.08,
  });
  badge(s, 'answer length=6', 748, 328, 114, 'amber');
  badge(s, 'question cap=2', 876, 328, 114, 'amber');
  badge(s, 'answer cap=6', 1004, 328, 114, 'amber');
  card(s, 724, 424, 410, 112, C.panel, C.line, 8);
  text(s, 'Normalization', 750, 448, 300, 26, { fontSize: 20, bold: true, color: C.navy, typeface: FONT_H });
  text(s, 'Readable text is preserved, while duplicate keys ignore case, punctuation, and whitespace.', 750, 482, 346, 38, { fontSize: 14, color: C.muted });
}

// 7
{
  const s = deck.slides.add();
  header(s, 'Semantic check', 'Near-duplicate control with polarity awareness', 'Similarity groups are split by conclusion polarity before applying the cap.', 7);
  card(s, 64, 184, 330, 292, C.panel, C.line, 8);
  text(s, 'Embedding configuration', 90, 210, 270, 28, { fontSize: 20, bold: true, color: C.navy, typeface: FONT_H });
  table(s, 90, 252, 252, 34, [0.47, 0.53], [
    ['Parameter', 'Value'],
    ['model', 'Qwen3-Embedding-4B'],
    ['precision', 'bfloat16'],
    ['input', 'question + answer'],
    ['batch size', '32'],
    ['normalize', 'true'],
  ], { font: 10, headerFont: 10, boldFirstCol: true });
  card(s, 462, 184, 330, 292, C.panel, C.line, 8);
  text(s, 'Semantic cap', 488, 210, 220, 28, { fontSize: 20, bold: true, color: C.navy, typeface: FONT_H });
  kpi(s, 494, 260, 126, 102, '0.955', 'cosine similarity threshold', 'blue');
  kpi(s, 642, 260, 126, 102, '4', 'maximum per cluster + polarity', 'blue');
  text(s, '12,336 after rule-based cleaning becomes 11,898 after the semantic cap. This removes 438 near-duplicate rows.', 498, 386, 246, 58, {
    fontSize: 14,
    color: C.muted,
    lineSpacing: 1.08,
  });
  card(s, 860, 184, 330, 292, C.panel, C.line, 8);
  text(s, 'Polarite örnekleri', 886, 210, 230, 28, { fontSize: 20, bold: true, color: C.navy, typeface: FONT_H });
  table(s, 886, 252, 252, 34, [0.50, 0.50], [
    ['Olumlu', 'Olumsuz'],
    ['hukuka uygun', 'hukuka aykırı'],
    ['geçerli', 'geçersiz'],
    ['hakkı vardır', 'hakkı yoktur'],
    ['mümkün', 'mümkün değildir'],
    ['evet', 'hayır'],
  ], { font: 10, headerFont: 10 });
  text(s, 'Why this matters: a sentence that says an action is allowed can be semantically close to a sentence that says the same action is forbidden.', 182, 542, 880, 38, {
    fontSize: 17,
    color: C.ink,
    alignment: 'center',
  });
  badge(s, 'no LLM judge', 1034, 120, 110, 'blue');
}

// 8
{
  const s = deck.slides.add();
  header(s, 'Training', 'bf16 LoRA training and the epoch-3 break point', 'Validation loss reaches its minimum at epoch 3, then rises while training loss keeps falling.', 8);
  table(s, 64, 170, 472, 34, [0.48, 0.52], [
    ['Parameter', 'Value'],
    ['base model', 'Qwen3-4B-Instruct-2507'],
    ['method / precision', 'LoRA SFT, bf16 base'],
    ['train / validation', '10,709 / 1,189'],
    ['epochs trained / selected', '4 / 3'],
    ['batch / accumulation', '1 / 16, effective 16'],
    ['learning rate', '2e-4, cosine, warmup 0.03'],
    ['max sequence length', '512'],
    ['LoRA r / alpha / dropout', '16 / 32 / 0.05'],
    ['target modules', 'q,k,v,o + gate,up,down'],
    ['optimizer', 'paged AdamW 8-bit'],
    ['seed / data seed', '42 / 42'],
  ], { font: 10, headerFont: 10, boldFirstCol: true });
  card(s, 600, 174, 540, 352, C.panel, C.line, 8);
  s.charts.add('line', {
    position: { left: 636, top: 210, width: 466, height: 230 },
    categories: ['epoch 1', 'epoch 2', 'epoch 3', 'epoch 4'],
    series: [
      { name: 'validation loss', values: [0.4699, 0.3962, 0.3833, 0.4058], line: { style: 'solid', fill: C.red, width: 3 }, marker: { symbol: 'circle', size: 7 } },
      { name: 'training loss', values: [0.4653, 0.3695, 0.2280, 0.1500], line: { style: 'solid', fill: C.blue, width: 3 }, marker: { symbol: 'circle', size: 7 } },
    ],
    legend: { position: 'bottom', overlay: false, textStyle: { fill: C.muted, fontSize: 11 } },
    yAxis: { min: 0.1, max: 0.5, numberFormatCode: '0.0', majorGridlines: { style: 'solid', fill: C.line, width: 1 }, textStyle: { fill: C.muted, fontSize: 11 } },
    xAxis: { textStyle: { fill: C.muted, fontSize: 11 } },
    dataLabels: { showValue: false },
    chartFill: C.panel,
    plotAreaFill: C.panel,
    chartLine: { style: 'solid', fill: C.panel, width: 0 },
  });
  text(s, 'Selected checkpoint: epoch 3, validation loss 0.3833. Epoch 4 is the visible overfitting point.', 648, 466, 430, 38, {
    fontSize: 14,
    color: C.ink,
    lineSpacing: 1.08,
  });
  badge(s, 'RTX 4070 Ti 12 GB', 984, 120, 154, 'blue');
}

// 9
{
  const s = deck.slides.add();
  header(s, 'Results', 'Full-test before and after metrics', 'The before/after comparison uses the same 1,500 official test examples and saved generations.', 9);
  table(s, 64, 166, 720, 34, [0.28, 0.15, 0.18, 0.18, 0.21], [
    ['Metric', 'Base', 'Fine-tuned', 'Absolute gain', 'Relative gain'],
    ['ROUGE-1', '0.201', '0.560', '+0.359', '+178%'],
    ['ROUGE-2', '0.094', '0.454', '+0.360', '+384%'],
    ['ROUGE-L', '0.152', '0.527', '+0.375', '+248%'],
    ['R-Lsum', '0.163', '0.527', '+0.364', '+223%'],
    ['BERT F1', '0.649', '0.829', '+0.180', '+27.7%'],
  ], { font: 12, headerFont: 11, boldFirstCol: true, align: ['left', 'center', 'center', 'center', 'center'] });

  card(s, 826, 166, 334, 204, C.greenSoft, C.line, 8);
  text(s, 'Main gains', 852, 190, 250, 24, { fontSize: 20, bold: true, color: C.navy, typeface: FONT_H });
  text(s, '+384%', 852, 224, 116, 26, { fontSize: 24, bold: true, color: C.green, typeface: FONT_H });
  text(s, 'ROUGE-2 relative gain', 978, 230, 150, 18, { fontSize: 12, color: C.muted });
  line(s, 852, 260, 248, C.line);
  text(s, '+178%', 852, 276, 116, 26, { fontSize: 24, bold: true, color: C.green, typeface: FONT_H });
  text(s, 'ROUGE-1 relative gain', 978, 282, 150, 18, { fontSize: 12, color: C.muted });
  line(s, 852, 312, 248, C.line);
  text(s, '+27.7%', 852, 328, 116, 26, { fontSize: 24, bold: true, color: C.green, typeface: FONT_H });
  text(s, 'BERT F1 relative gain', 978, 334, 150, 18, { fontSize: 12, color: C.muted });

  card(s, 64, 406, 780, 206, C.panel, C.line, 8);
  text(s, 'Score comparison on the full test set', 92, 426, 520, 24, { fontSize: 19, bold: true, color: C.navy, typeface: FONT_H });
  s.charts.add('bar', {
    position: { left: 88, top: 462, width: 720, height: 118 },
    categories: ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'BERT F1'],
    series: [
      { name: 'Base', values: [0.201, 0.094, 0.152, 0.649], fill: '#94a3b8' },
      { name: 'Qwen3 + LoRA', values: [0.560, 0.454, 0.527, 0.829], fill: C.green },
    ],
    barOptions: { direction: 'column', grouping: 'clustered', gapWidth: 55 },
    legend: { position: 'bottom', overlay: false, textStyle: { fill: C.muted, fontSize: 9 } },
    yAxis: {
      min: 0,
      max: 0.9,
      numberFormatCode: '0.0',
      majorGridlines: { style: 'solid', fill: C.line, width: 1 },
      textStyle: { fill: C.muted, fontSize: 9 },
    },
    xAxis: { textStyle: { fill: C.muted, fontSize: 10 } },
    dataLabels: { showValue: false },
    chartFill: C.panel,
    plotAreaFill: C.panel,
    chartLine: { style: 'solid', fill: C.panel, width: 0 },
  });

  card(s, 876, 406, 284, 206, C.panel, C.line, 8);
  text(s, 'Metric note', 902, 430, 220, 24, { fontSize: 19, bold: true, color: C.navy, typeface: FONT_H });
  text(s, 'BERTScore model: bert-base-multilingual-cased, lang=tr.', 902, 468, 230, 44, {
    fontSize: 13,
    color: C.ink,
    lineSpacing: 1.1,
  });
  text(s, 'Scores measure similarity to reference answers, not legal correctness.', 902, 528, 230, 44, {
    fontSize: 13,
    color: C.muted,
    lineSpacing: 1.1,
  });
}

// 10
{
  const s = deck.slides.add();
  header(s, 'Error analysis', 'Objective script shows what changed', 'The analysis uses fixed rules, not another language model as a judge.', 10);
  table(s, 78, 184, 748, 58, [0.42, 0.19, 0.19, 0.20], [
    ['Metric', 'Base', 'Fine-tuned', 'Change'],
    ['Mean answer length', '89.9 words', '23.5 words', 'much shorter'],
    ['Article-number agreement', '98.4%', '98.7%', 'stable'],
    ['Yes/no polarity accuracy', '81.6%', '95.8%', 'large gain'],
    ['Repetition / degeneration', '13.6%', '0.5%', 'almost removed'],
  ], { font: 14, boldFirstCol: true, align: ['left', 'center', 'center', 'center'] });
  card(s, 884, 194, 250, 222, C.panel, C.line, 8);
  text(s, 'Interpretation', 910, 224, 198, 28, { fontSize: 22, bold: true, color: C.navy, typeface: FONT_H, alignment: 'center' });
  text(s, 'Fine-tuning mostly fixes format, verbosity, and repetition. It does not verify legal facts, dates, or conclusions.', 912, 274, 196, 74, {
    fontSize: 15,
    color: C.muted,
    alignment: 'center',
    lineSpacing: 1.08,
  });
  badge(s, 'scriptable analysis', 924, 364, 150, 'blue');
  kpi(s, 116, 498, 230, 106, '-66.4', 'generated words per answer', 'green');
  kpi(s, 402, 498, 230, 106, '+14.2 pts', 'yes/no polarity accuracy', 'green');
  kpi(s, 688, 498, 230, 106, '-13.1 pts', 'repetition rate', 'green');
}

// 11
{
  const s = deck.slides.add();
  header(s, 'Hallucination case 1', 'Systematic 2007 date regression after fine-tuning', 'A clear failure mode: the base model often says 1982 correctly, while the fine-tuned model injects 2007.', 11);
  kpi(s, 86, 180, 224, 106, '10', 'fine-tuned 2007 regressions in the date-question scan', 'red');
  kpi(s, 344, 180, 224, 106, '0', 'base outputs with 2007 in the same scan', 'green');
  kpi(s, 602, 180, 224, 106, '1982', 'correct reference year in all shown cases', 'amber');
  badge(s, 'reproducible JSONL scan', 950, 218, 180, 'blue');
  table(s, 66, 334, 1120, 48, [0.09, 0.37, 0.16, 0.18, 0.20], [
    ['ID', 'Question', 'Reference', 'Base', 'Fine-tuned'],
    ['183', 'Constitution Article 176 - accepted year?', '1982', '1982', '2007'],
    ['144', 'Constitution Article 161 - accepted year?', '1982', '1982', '10.08.2007'],
    ['162', 'Constitution Article 133 - accepted year?', '1982', '1982', '12.08.2007'],
    ['237', 'Constitution Article 127 - accepted year?', '1982', '1982', '2007 pattern'],
  ], { font: 12, headerFont: 11, boldFirstCol: true, align: ['center', 'left', 'center', 'center', 'center'] });
  text(s, 'This is the strongest qualitative finding: higher overlap scores can coexist with a new, consistent legal-date hallucination.', 120, 588, 1010, 30, {
    fontSize: 17,
    alignment: 'center',
  });
}

// 12
{
  const s = deck.slides.add();
  header(s, 'Hallucination case 2', 'Wrong conclusions and invented legal details remain', 'Fine-tuning can make an answer shorter and more confident while preserving or creating a legal error.', 12);
  table(s, 58, 176, 1126, 58, [0.08, 0.34, 0.22, 0.16, 0.20], [
    ['ID', 'Case', 'Reference conclusion', 'Base', 'Fine-tuned'],
    ['155', 'Election renewal after a presidential decision', 'No - parliamentary election must be held together', 'Yes', 'Yes, general election is not mandatory'],
    ['381', 'One-fourth accusation of treason in parliament', 'No - threshold is not met', 'off-topic', 'Yes, trial is possible'],
    ['371', 'Free apartment transfer after mother died', 'May be invalid due to sham transaction', 'maybe valid', 'lawful if registered'],
    ['197', 'Penalty for not showing a child', 'enforcement criminal court sanction', 'hedges', 'invents 1-3 years in prison'],
  ], { font: 11, headerFont: 10, boldFirstCol: true, align: ['center', 'left', 'left', 'left', 'left'] });
  card(s, 58, 482, 356, 132, C.redSoft, C.line, 8);
  text(s, 'Wrong conclusion', 84, 506, 304, 30, { fontSize: 21, bold: true, color: C.red, alignment: 'center', typeface: FONT_H });
  text(s, 'Article match is not enough; the final conclusion can still flip.', 92, 550, 288, 46, {
    fontSize: 15,
    color: C.ink,
    alignment: 'center',
    lineSpacing: 1.08,
  });
  card(s, 462, 482, 356, 132, C.amberSoft, C.line, 8);
  text(s, 'Invented legal detail', 488, 506, 304, 30, { fontSize: 21, bold: true, color: C.amber, alignment: 'center', typeface: FONT_H });
  text(s, 'The model may invent a sanction, year, threshold, or extra condition.', 496, 550, 288, 46, {
    fontSize: 15,
    color: C.ink,
    alignment: 'center',
    lineSpacing: 1.08,
  });
  card(s, 866, 482, 356, 132, C.blueSoft, C.line, 8);
  text(s, 'Dataset noise matters', 892, 506, 304, 30, { fontSize: 21, bold: true, color: C.blue, alignment: 'center', typeface: FONT_H });
  text(s, 'Terse references can make automatic scores look acceptable.', 900, 550, 288, 46, {
    fontSize: 15,
    color: C.ink,
    alignment: 'center',
    lineSpacing: 1.08,
  });
}

// 13
{
  const s = deck.slides.add();
  header(s, 'Qualitative wins', 'Where fine-tuning clearly helps', 'The fine-tuned model is often shorter, more direct, and closer to the reference answer style.', 13);
  table(s, 62, 176, 1116, 58, [0.08, 0.34, 0.22, 0.20, 0.16], [
    ['ID', 'Question type', 'Baseline behavior', 'Fine-tuned behavior', 'Takeaway'],
    ['8', 'Parliamentary question unanswered', 'long hedged legal discussion', 'matches reference conclusion', 'format + conclusion'],
    ['21', 'Public enterprise duties', 'wrong yes conclusion', 'correct no conclusion', 'polarity fixed'],
    ['20', 'Private-sector mobbing', 'long generic advice', 'concise labor-court route', 'verbosity fixed'],
    ['16', 'Rent determination lawsuit', 'long definition with drift', 'short definition close to reference', 'answer style fixed'],
  ], { font: 12, headerFont: 10, boldFirstCol: true, align: ['center', 'left', 'left', 'left', 'left'] });
  kpi(s, 118, 512, 248, 92, '13.6% -> 0.5%', 'degeneration nearly disappears', 'green');
  kpi(s, 414, 512, 248, 92, '89.9 -> 23.5', 'mean generated words', 'green');
  kpi(s, 710, 512, 248, 92, '81.6% -> 95.8%', 'yes/no polarity accuracy', 'green');
}

// 14
{
  const s = deck.slides.add();
  header(s, 'Final takeaway', 'What improved, and what still blocks reliability', 'The system is a stronger corpus-aligned QA generator, not a reliable legal authority.', 14);
  table(s, 72, 176, 1076, 58, [0.23, 0.37, 0.40], [
    ['Area', 'Result', 'Meaning'],
    ['Baseline choice', 'Qwen3-4B selected', 'best trainable dense baseline among screened models'],
    ['Preprocessing', '13,354 -> 11,898', 'duplicates reduced without touching the official test set'],
    ['LoRA training', 'epoch 3 best validation loss', 'bf16 LoRA fits a 12 GB consumer GPU and improves metrics'],
    ['Evaluation', 'ROUGE-1 0.201 -> 0.560, BERT F1 0.649 -> 0.829', 'large corpus-similarity gain on the full test set'],
    ['Risk', '2007 date regression and wrong yes/no cases', 'automatic metrics are not legal correctness'],
  ], { font: 12, headerFont: 11, boldFirstCol: true });
  card(s, 142, 538, 280, 68, C.panel, C.line, 8);
  text(s, 'Next: retrieval grounding', 160, 556, 244, 20, { fontSize: 16, bold: true, color: C.navy, alignment: 'center' });
  text(s, 'cite authoritative legal text', 160, 580, 244, 18, { fontSize: 11, color: C.muted, alignment: 'center' });
  card(s, 500, 538, 280, 68, C.panel, C.line, 8);
  text(s, 'Next: citation verification', 518, 556, 244, 20, { fontSize: 16, bold: true, color: C.navy, alignment: 'center' });
  text(s, 'check article, date, and number claims', 518, 580, 244, 18, { fontSize: 11, color: C.muted, alignment: 'center' });
  card(s, 858, 538, 280, 68, C.panel, C.line, 8);
  text(s, 'Next: human legal review', 876, 556, 244, 20, { fontSize: 16, bold: true, color: C.navy, alignment: 'center' });
  text(s, 'expert rubric beyond similarity', 876, 580, 244, 18, { fontSize: 11, color: C.muted, alignment: 'center' });
}

// 15
{
  const s = deck.slides.add();
  s.background.fill = C.bg;
  shape(s, 'rect', { left: 0, top: 0, width: 1280, height: 12 }, C.navy, { style: 'solid', fill: C.navy, width: 0 });
  text(s, 'Thank you', 0, 210, 1280, 56, { fontSize: 46, bold: true, typeface: FONT_H, alignment: 'center' });
  text(s, 'Questions?', 0, 282, 1280, 72, { fontSize: 58, bold: true, color: C.navy, typeface: FONT_H, alignment: 'center' });
  text(s, 'Group 6 | CSE4078 Spring 2026', 0, 398, 1280, 26, { fontSize: 18, color: C.muted, alignment: 'center' });
  text(s, 'Repository: github.com/cereneren/CSE4078S26_Grp6', 0, 438, 1280, 24, { fontSize: 15, color: C.faint, alignment: 'center' });
  text(s, 'Academic prototype only - not legal advice', 0, 478, 1280, 24, { fontSize: 15, bold: true, color: C.red, alignment: 'center' });
  footer(s, 15);
}

await fs.writeFile(path.join(TMP, 'source-notes.txt'), `Sources and provenance

User-provided requested revisions: white theme, restrained colors, one language, full visual QA.
Data and metrics: repository JSONL and metric JSON files under data/, models/fine_tuned_v2/, and outputs/.
Presentation content language: English throughout. Turkish legal examples are translated or described in English to avoid mixed-language slides.
`, 'utf8');

await fs.writeFile(path.join(TMP, 'slide-plan.txt'), `White-theme 15-slide deck.
Palette: white background, slate text, muted navy/blue, green for improvements, red/amber only for risk.
Layout rule: no nested cards; wide tables use fixed row heights and generous margins.
`, 'utf8');

for (const [idx, slide] of deck.slides.items.entries()) {
  const stem = `slide-${String(idx + 1).padStart(2, '0')}`;
  const png = await deck.export({ slide, format: 'png', scale: 1 });
  await fs.writeFile(path.join(PREVIEW, `${stem}.png`), new Uint8Array(await png.arrayBuffer()));
  const layout = await slide.export({ format: 'layout' });
  await fs.writeFile(path.join(LAYOUT, `${stem}.layout.json`), await layout.text(), 'utf8');
}

const inspect = await deck.inspect({ kind: 'slide,textbox,shape,chart,table', maxChars: 12000 });
await fs.writeFile(path.join(TMP, 'inspect.ndjson'), inspect.ndjson, 'utf8');
const pptx = await PresentationFile.exportPptx(deck);
await pptx.save(FINAL_PPTX);
await fs.writeFile(path.join(QA, 'visual-qa.txt'), 'Rendered all slides. Manual inspection pending.\n', 'utf8');

const st = await fs.stat(FINAL_PPTX);
console.log(JSON.stringify({
  out: FINAL_PPTX,
  bytes: st.size,
  slides: deck.slides.items.length,
  workspace: WORK,
  preview: PREVIEW,
}, null, 2));
