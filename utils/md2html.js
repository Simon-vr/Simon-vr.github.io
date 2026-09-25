const { readFile, writeFile } = require('node:fs/promises');
const path = require('path');
const cheerio = require('cheerio');
const fs = require('fs').promises;

const IMAGE_EXTENSION_REGEX = /\.(png|jpe?g|gif|webp|bmp|svg|avif)(\?.*)?$/i;
const CUSTOM_MEDIA_REGEX = /\[(.*?)\]\((.*?)\)/;

let markedInstance = null;

async function getMarked() {
  if (markedInstance) {
    return markedInstance;
  }

  const { marked } = await import('marked');
  marked.setOptions({
    gfm: true,
    breaks: false,
  });
  markedInstance = marked;
  return markedInstance;
}

function normalizePathForHtml(filepath) {
  return filepath.replace(/\\/g, '/');
}

function isCustomImageTarget(target) {
  return IMAGE_EXTENSION_REGEX.test(target.trim());
}

function isVideoLabel(label) {
  return label.trim().toLowerCase() === 'video';
}

function isMusicLabel(label) {
  return label.trim().toLowerCase() === 'music';
}

function extractMusicId(url) {
  const idMatch = url.match(/[?&]id=(\d+)/i) || url.match(/\/song\/(\d+)/i);
  return idMatch ? idMatch[1] : null;
}

function buildVideoIframe(url) {
  return `<iframe class="md-video-frame" src="${url}"></iframe>`;
}

function buildMusicIframe(songId) {
  return `<iframe class="md-music-frame" src="https://music.163.com/outchain/player?type=2&id=${songId}&auto=0&height=66"></iframe>`;
}

function preprocessMarkdown(md) {
  const lines = md.split(/\r?\n/);
  const transformed = lines.map((line) => {
    const trimmed = line.trim();
    const match = trimmed.match(CUSTOM_MEDIA_REGEX);
    if (!match) {
      return line;
    }

    const label = match[1].trim();
    const target = match[2].trim();

    if (isVideoLabel(label)) {
      return buildVideoIframe(target);
    }

    if (isMusicLabel(label)) {
      const songId = extractMusicId(target);
      if (!songId) {
        console.warn(`未识别到网易云音乐ID，跳过: ${target}`);
        return line;
      }
      return buildMusicIframe(songId);
    }

    if (isCustomImageTarget(target)) {
      return `![${label}](${target})`;
    }

    return line;
  });

  return transformed.join('\n');
}

async function resetFolder(folderPath) {
  await fs.rm(folderPath, { recursive: true, force: true });
  await fs.mkdir(folderPath, { recursive: true });
}

async function copyUsedImages(imageDir, targetDir, usedImages) {
  if (usedImages.size === 0) {
    return;
  }

  for (const filename of usedImages) {
    const sourcePath = path.join(imageDir, filename);
    const targetPath = path.join(targetDir, filename);
    try {
      await fs.copyFile(sourcePath, targetPath);
      console.log(`已复制图像: ${filename}`);
    } catch (error) {
      if (error.code === 'ENOENT') {
        console.warn(`图像不存在，已跳过: ${filename}`);
      } else {
        throw error;
      }
    }
  }
}

function isLocalImageSource(src) {
  if (!src) return false;
  if (/^https?:\/\//i.test(src) || /^data:/i.test(src)) return false;
  return IMAGE_EXTENSION_REGEX.test(src);
}

function isFromHandscriptImageDir(src) {
  const normalized = normalizePathForHtml(src);
  return normalized.startsWith('image/handscript/') || normalized.startsWith('./image/handscript/');
}

async function md2html(className, folderName) {
  const marked = await getMarked();
  const baseDir = path.join(path.dirname(__dirname), 'content', className, folderName);
  const mdPath = path.join(baseDir, 'handscript.md');
  const htmlPath = path.join(baseDir, 'text_CN.html');
  const imageDir = path.join(baseDir, 'image', 'handscript');
  const assetsDir = path.join(baseDir, 'assets');

  await resetFolder(assetsDir);

  const rawMd = await readFile(mdPath, 'utf8');
  const preprocessedMd = preprocessMarkdown(rawMd);
  const html = marked.parse(preprocessedMd);

  const $ = cheerio.load(html);
  const usedImages = new Set();

  $('iframe').each((_, elem) => {
    const $iframe = $(elem);
    const src = $iframe.attr('src') || '';
    if (src.includes('music.163.com/outchain/player') || $iframe.hasClass('md-music-frame')) {
      $iframe.attr('frameborder', 'no');
      $iframe.attr('border', '0');
      $iframe.attr('marginwidth', '0');
      $iframe.attr('marginheight', '0');
      $iframe.attr('width', '100%');
      $iframe.attr('height', '86');
      return;
    }

    $iframe.attr('width', '100%');
    $iframe.attr('height', '500');
    $iframe.attr('scrolling', 'no');
    $iframe.attr('frameborder', '0');
    $iframe.attr('allowfullscreen', 'allowfullscreen');
    $iframe.attr('sandbox', 'allow-same-origin allow-scripts');
  });

  $('img').each((_, elem) => {
    const $img = $(elem);
    const src = $img.attr('src') || '';
    const altText = ($img.attr('alt') || '').trim();

    if (isLocalImageSource(src)) {
      const filename = path.basename(normalizePathForHtml(src));
      if (filename) {
        if (isFromHandscriptImageDir(src)) {
          usedImages.add(filename);
        }
        $img.attr('src', `./assets/${filename}`);
      }
    }

    $img.removeAttr('style');
    $img.addClass('md-image');

    const $figure = $('<figure class="md-figure"></figure>');
    const $caption = altText ? $('<figcaption></figcaption>').text(altText) : null;
    const $parent = $img.parent();
    const isSingleImageParagraph = $parent.is('p') && $parent.children().length === 1 && !$parent.text().trim();

    if (isSingleImageParagraph) {
      $parent.replaceWith($figure);
      $figure.append($img);
      if ($caption) {
        $figure.append($caption);
      }
    } else if (!$parent.is('figure')) {
      $img.replaceWith($figure);
      $figure.append($img);
      if ($caption) {
        $figure.append($caption);
      }
    } else if ($caption && $parent.find('figcaption').length === 0) {
      $parent.append($caption);
    }
  });

  const output = $.html();
  await writeFile(htmlPath, output, 'utf8');
  await copyUsedImages(imageDir, assetsDir, usedImages);

  console.log('HTML已成功转写');
}

// 从命令行参数获取文件夹名称
const className = process.argv[2];
const folderName = process.argv[3];
if (!className || !folderName) {
  console.error('请提供分类, 文件夹id名称');
  process.exit(1);
}
md2html(className, folderName);