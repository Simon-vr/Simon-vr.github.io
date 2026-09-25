const path = require('path');
const { readFile, writeFile } = require('node:fs/promises');
const cheerio = require('cheerio');
const { loadAppConfig } = require('./config');

const METADATA_FIELDS = ['title', 'tag', 'excerpt'];
const BATCH_SIZE = 24;
const PROTECTED_SEGMENT_REGEX = /\$\$[\s\S]*?\$\$|\$[^$\n]+\$|https?:\/\/[^\s<>"')]+/g;
const SKIP_TAGS = new Set(['script', 'style', 'code', 'pre', 'noscript']);

function printUsage() {
  console.log('用法: node ./utils/translate.js <category> <id> [--metadata-only|--html-only] [--dry-run]');
  console.log('示例: node ./utils/translate.js share napoleon_math');
  console.log('示例: node ./utils/translate.js log book_antarctica --metadata-only');
}

function parseArgs(argv) {
  if (argv.includes('--help') || argv.includes('-h')) {
    printUsage();
    process.exit(0);
  }

  if (argv.length < 2) {
    printUsage();
    throw new Error('参数不足：请提供分类和文章 id');
  }

  const [category, id, ...flags] = argv;
  const metadataOnly = flags.includes('--metadata-only');
  const htmlOnly = flags.includes('--html-only');
  const dryRun = flags.includes('--dry-run');

  if (metadataOnly && htmlOnly) {
    throw new Error('参数冲突：--metadata-only 与 --html-only 不能同时使用');
  }

  return {
    category,
    id,
    dryRun,
    translateMetadataPart: !htmlOnly,
    translateHtmlPart: !metadataOnly,
  };
}

function normalizeText(value) {
  return String(value).replace(/\r\n/g, '\n');
}

function chunkArray(items, size) {
  const chunks = [];
  for (let i = 0; i < items.length; i += size) {
    chunks.push(items.slice(i, i + size));
  }
  return chunks;
}

function extractJsonArray(rawText) {
  const candidates = [];
  const trimmed = String(rawText || '').trim();
  if (trimmed) {
    candidates.push(trimmed);
  }

  const fencedMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fencedMatch && fencedMatch[1]) {
    candidates.push(fencedMatch[1].trim());
  }

  const start = trimmed.indexOf('[');
  const end = trimmed.lastIndexOf(']');
  if (start !== -1 && end !== -1 && end > start) {
    candidates.push(trimmed.slice(start, end + 1));
  }

  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate);
      if (Array.isArray(parsed)) {
        return parsed;
      }
    } catch (_error) {
      continue;
    }
  }

  throw new Error(`模型返回内容不是合法 JSON 数组: ${trimmed.slice(0, 240)}`);
}

function ensureTranslationConfig(translationConfig) {
  if (!translationConfig) {
    throw new Error('缺少 translation 配置');
  }

  const requiredFields = ['baseUrl', 'apiKey', 'model'];
  for (const field of requiredFields) {
    if (!translationConfig[field]) {
      throw new Error(`translation.${field} 未配置`);
    }
  }
}

async function callModelTranslate(textItems, translationConfig) {
  const endpoint = `${String(translationConfig.baseUrl).replace(/\/+$/, '')}/chat/completions`;
  const timeoutMs = Number(translationConfig.requestTimeoutMs || 120000);
  const temperature = Number.isFinite(Number(translationConfig.temperature))
    ? Number(translationConfig.temperature)
    : 1;

  const systemPrompt = [
    'You are a professional translator from Simplified Chinese to natural English.',
    'Keep original meaning, tone, and punctuation.',
    'Keep placeholders like __KEEP_0__ exactly unchanged.',
    'Do not add explanations.',
    'Return only a valid JSON array of strings with the same length as input.'
  ].join(' ');

  const userPrompt = [
    'Translate each item in this JSON array to fluent natural English.',
    'Output only JSON array, no code fence, no comments.',
    '',
    JSON.stringify(textItems)
  ].join('\n');

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  let response;
  try {
    response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${translationConfig.apiKey}`,
      },
      body: JSON.stringify({
        model: translationConfig.model,
        temperature,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt },
        ],
      }),
      signal: controller.signal,
    });
  } catch (error) {
    if (error.name === 'AbortError') {
      throw new Error(`翻译请求超时（>${timeoutMs}ms）`);
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }

  const responseText = await response.text();
  let payload = null;
  try {
    payload = JSON.parse(responseText);
  } catch (_error) {
    if (!response.ok) {
      throw new Error(`翻译请求失败(${response.status}): ${responseText.slice(0, 240)}`);
    }
    throw new Error(`翻译接口返回非 JSON: ${responseText.slice(0, 240)}`);
  }

  if (!response.ok) {
    const message = payload && payload.error && payload.error.message
      ? payload.error.message
      : responseText.slice(0, 240);
    throw new Error(`翻译请求失败(${response.status}): ${message}`);
  }

  const content = payload && payload.choices && payload.choices[0]
    && payload.choices[0].message && payload.choices[0].message.content
    ? payload.choices[0].message.content
    : '';

  if (!content) {
    throw new Error('翻译接口返回为空');
  }

  const translated = extractJsonArray(content).map((item) => String(item));
  if (translated.length !== textItems.length) {
    throw new Error(`翻译条目数量不一致: 输入 ${textItems.length}, 输出 ${translated.length}`);
  }

  return translated;
}

function maskProtectedSegments(text) {
  const protectedPairs = [];
  let index = 0;

  const maskedText = normalizeText(text).replace(PROTECTED_SEGMENT_REGEX, (segment) => {
    const token = `__KEEP_${index}__`;
    protectedPairs.push({ token, segment });
    index += 1;
    return token;
  });

  return { maskedText, protectedPairs };
}

function unmaskProtectedSegments(text, protectedPairs) {
  let restored = String(text);
  for (const pair of protectedPairs) {
    restored = restored.split(pair.token).join(pair.segment);
  }
  return restored;
}

function shouldTranslateText(text) {
  const normalized = normalizeText(text).trim();
  if (!normalized) {
    return false;
  }

  return /[\u4e00-\u9fff]/.test(normalized);
}

function collectTranslatableTextNodes($) {
  const nodes = [];
  const rootNode = $('body').length ? $('body').get(0) : $.root().get(0);

  function walk(node) {
    if (!node) {
      return;
    }

    if (node.type === 'text') {
      if (shouldTranslateText(node.data || '')) {
        nodes.push(node);
      }
      return;
    }

    if (node.type === 'tag' && SKIP_TAGS.has(node.name)) {
      return;
    }

    const children = node.children || [];
    for (const child of children) {
      walk(child);
    }
  }

  walk(rootNode);
  return nodes;
}

async function translateMetadata(contentDir, translationConfig, dryRun) {
  const metadataPath = path.join(contentDir, 'metadata.json');
  const metadataRaw = await readFile(metadataPath, 'utf8');
  const metadata = JSON.parse(metadataRaw);

  const entries = [];
  for (const field of METADATA_FIELDS) {
    const value = metadata[field];
    if (typeof value === 'string' && value.trim()) {
      entries.push({ field, value: value.trim() });
    }
  }

  if (entries.length === 0) {
    return { updatedCount: 0, skipped: true };
  }

  const sourceTexts = entries.map((entry) => entry.value);
  const translatedTexts = dryRun
    ? sourceTexts
    : await callModelTranslate(sourceTexts, translationConfig);

  entries.forEach((entry, index) => {
    metadata[`${entry.field}_en`] = translatedTexts[index].trim();
  });

  if (!dryRun) {
    await writeFile(metadataPath, `${JSON.stringify(metadata, null, 2)}\n`, 'utf8');
  }

  return {
    updatedCount: entries.length,
    skipped: false,
  };
}

async function translateHtml(contentDir, translationConfig, dryRun) {
  const cnPath = path.join(contentDir, 'text_CN.html');
  const enPath = path.join(contentDir, 'text_EN.html');
  const html = await readFile(cnPath, 'utf8');

  const $ = cheerio.load(html, { decodeEntities: false });
  if ($('html').length) {
    $('html').attr('lang', 'en');
  }

  const nodes = collectTranslatableTextNodes($);
  if (nodes.length === 0) {
    if (!dryRun) {
      await writeFile(enPath, $.html(), 'utf8');
    }
    return { translatedNodeCount: 0, skipped: true };
  }

  const tasks = nodes.map((node) => {
    const originalText = node.data || '';
    const { maskedText, protectedPairs } = maskProtectedSegments(originalText);
    return {
      node,
      maskedText,
      protectedPairs,
    };
  });

  const batches = chunkArray(tasks, BATCH_SIZE);

  for (const batch of batches) {
    const sourceTexts = batch.map((item) => item.maskedText);
    const translatedTexts = dryRun
      ? sourceTexts
      : await callModelTranslate(sourceTexts, translationConfig);

    batch.forEach((item, index) => {
      const translated = translatedTexts[index];
      item.node.data = unmaskProtectedSegments(translated, item.protectedPairs);
    });
  }

  if (!dryRun) {
    await writeFile(enPath, $.html(), 'utf8');
  }

  return {
    translatedNodeCount: tasks.length,
    skipped: false,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const appConfig = loadAppConfig();
  ensureTranslationConfig(appConfig.translation);

  const contentDir = path.join(path.dirname(__dirname), 'docs', 'content', args.category, args.id);

  const summary = {
    metadata: null,
    html: null,
  };

  if (args.translateMetadataPart) {
    summary.metadata = await translateMetadata(contentDir, appConfig.translation, args.dryRun);
  }

  if (args.translateHtmlPart) {
    summary.html = await translateHtml(contentDir, appConfig.translation, args.dryRun);
  }

  if (args.dryRun) {
    console.log('Dry-run 完成（未写入文件）');
  } else {
    console.log('翻译完成');
  }

  if (summary.metadata) {
    if (summary.metadata.skipped) {
      console.log('metadata: 没有可翻译字段');
    } else {
      console.log(`metadata: 已更新 ${summary.metadata.updatedCount} 个字段`);
    }
  }

  if (summary.html) {
    if (summary.html.skipped) {
      console.log('html: 没有可翻译文本节点');
    } else {
      console.log(`html: 已翻译 ${summary.html.translatedNodeCount} 个文本节点`);
    }
  }
}

main().catch((error) => {
  console.error(`translate 失败: ${error.message}`);
  process.exit(1);
});
