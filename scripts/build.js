#!/usr/bin/env node
/**
 * 站点维护脚本（单目录模式）
 *
 * docs/ 既是源、也是 GitHub Pages 的产物，因此这里不做复制，
 * 只负责生成前端读取所需的索引与数据文件：
 *
 *   docs/content/<category>/index.json   每个分类的文章 id 列表
 *   docs/data/map-data.json              地图数据
 *
 * 用法：
 *   node scripts/build.js
 *   npm run build
 */

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const DOCS_DIR = path.join(ROOT, 'docs');
const CONTENT_DIR = path.join(DOCS_DIR, 'content');

// 仅用于写作、不需要发布的文件/目录
const UNPUBLISHED_FILES = new Set(['handscript.md', '.DS_Store', '.gitkeep']);
const UNPUBLISHED_DIRS = new Set(['image']);

function log(message) {
  console.log(`[build] ${message}`);
}

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

/** 清理不应发布的写作中间产物 */
function cleanUnpublished(dir) {
  let removed = 0;
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (UNPUBLISHED_DIRS.has(entry.name)) {
        fs.rmSync(full, { recursive: true, force: true });
        removed += 1;
      } else {
        removed += cleanUnpublished(full);
      }
    } else if (UNPUBLISHED_FILES.has(entry.name)) {
      fs.rmSync(full, { force: true });
      removed += 1;
    }
  }
  return removed;
}

function main() {
  if (!fs.existsSync(DOCS_DIR)) {
    throw new Error(`docs/ 目录不存在: ${DOCS_DIR}`);
  }
  if (!fs.existsSync(CONTENT_DIR)) {
    throw new Error(`docs/content/ 目录不存在: ${CONTENT_DIR}`);
  }

  // 关闭 Jekyll 处理，避免下划线开头的文件被忽略，同时加快部署
  fs.writeFileSync(path.join(DOCS_DIR, '.nojekyll'), '');

  const cleaned = cleanUnpublished(CONTENT_DIR);
  if (cleaned > 0) {
    log(`已清理 ${cleaned} 个不需要发布的写作中间产物`);
  }

  const categories = fs
    .readdirSync(CONTENT_DIR, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .filter((name) => name !== 'map');

  let totalPosts = 0;

  for (const category of categories) {
    const categoryDir = path.join(CONTENT_DIR, category);
    const ids = fs
      .readdirSync(categoryDir, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map((entry) => entry.name)
      .filter((id) => fs.existsSync(path.join(categoryDir, id, 'metadata.json')));

    // 按日期倒序写入索引
    const posts = ids
      .map((id) => {
        try {
          const metadata = JSON.parse(fs.readFileSync(path.join(categoryDir, id, 'metadata.json'), 'utf8'));
          return { id, date: metadata.date || '' };
        } catch (error) {
          console.warn(`[build] 跳过无法解析的 metadata: ${category}/${id} (${error.message})`);
          return null;
        }
      })
      .filter(Boolean)
      .sort((a, b) => new Date(b.date || 0) - new Date(a.date || 0))
      .map((item) => item.id);

    fs.writeFileSync(
      path.join(categoryDir, 'index.json'),
      JSON.stringify(posts, null, 2),
      'utf8'
    );

    totalPosts += posts.length;
    log(`分类 ${category}: ${posts.length} 篇文章`);
  }

  // 地图数据：保持前端读取路径 docs/data/map-data.json
  const mapSrc = path.join(CONTENT_DIR, 'map', 'data', 'asset', 'mapdata.json');
  if (fs.existsSync(mapSrc)) {
    ensureDir(path.join(DOCS_DIR, 'data'));
    fs.copyFileSync(mapSrc, path.join(DOCS_DIR, 'data', 'map-data.json'));
    log('已更新 data/map-data.json');
  } else {
    console.warn('[build] 未找到地图数据 content/map/data/asset/mapdata.json');
  }

  log(`完成：共 ${totalPosts} 篇文章`);
}

main();
