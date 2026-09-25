#!/usr/bin/env node
/**
 * 纯静态构建脚本
 *
 * 将 content/ 中的文章与资源、site/ 中的页面模板合并，输出到 docs/，
 * 由 GitHub Pages 直接托管（无需任何后端）。
 *
 * 用法：
 *   node scripts/build.js
 *   npm run build
 *
 * 输出结构（docs/）：
 *   index.html / blog.html / log.html / share.html / study.html /
 *   map.html / project.html / info.html
 *   css/ · js/
 *   data/blogs/<category>.json   文章列表（含中英字段）
 *   data/map-data.json           地图数据
 *   content/<category>/<id>/text_CN.html · text_EN.html · assets/
 */

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const SITE_DIR = path.join(ROOT, 'site');
const CONTENT_DIR = path.join(ROOT, 'content');
const OUT_DIR = path.join(ROOT, 'docs');

const SKIP_DIR_NAMES = new Set(['node_modules', '.git', '.DS_Store']);

function log(message) {
  console.log(`[build] ${message}`);
}

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

function removeDir(dir) {
  fs.rmSync(dir, { recursive: true, force: true });
}

/** 递归复制目录（跳过系统垃圾文件） */
function copyDir(src, dest) {
  if (!fs.existsSync(src)) return;
  ensureDir(dest);
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    if (SKIP_DIR_NAMES.has(entry.name)) continue;
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      copyDir(srcPath, destPath);
    } else {
      fs.copyFileSync(srcPath, destPath);
    }
  }
}

function copyFileIfExists(src, dest) {
  if (!fs.existsSync(src)) return false;
  ensureDir(path.dirname(dest));
  fs.copyFileSync(src, dest);
  return true;
}

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, 'utf8'));
}

/** 收集某个分类下的文章元数据 */
function collectCategory(category) {
  const categoryDir = path.join(CONTENT_DIR, category);
  const entries = fs.readdirSync(categoryDir, { withFileTypes: true });
  const blogs = [];

  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const postDir = path.join(categoryDir, entry.name);
    const metadataPath = path.join(postDir, 'metadata.json');
    if (!fs.existsSync(metadataPath)) continue;

    let metadata;
    try {
      metadata = readJson(metadataPath);
    } catch (error) {
      console.warn(`[build] 跳过无法解析的 metadata: ${metadataPath} (${error.message})`);
      continue;
    }

    blogs.push({ id: entry.name, ...metadata });
  }

  blogs.sort((a, b) => new Date(b.date || 0) - new Date(a.date || 0));
  return blogs;
}

/** 复制单篇文章的正文与资源 */
function copyPost(category, id) {
  const srcDir = path.join(CONTENT_DIR, category, id);
  const destDir = path.join(OUT_DIR, 'content', category, id);

  copyFileIfExists(path.join(srcDir, 'text_CN.html'), path.join(destDir, 'text_CN.html'));
  copyFileIfExists(path.join(srcDir, 'text_EN.html'), path.join(destDir, 'text_EN.html'));

  const assetsSrc = path.join(srcDir, 'assets');
  if (fs.existsSync(assetsSrc)) {
    copyDir(assetsSrc, path.join(destDir, 'assets'));
  }
}

function main() {
  if (!fs.existsSync(SITE_DIR)) {
    throw new Error(`site/ 目录不存在: ${SITE_DIR}`);
  }

  log('清理旧的 docs/ ...');
  removeDir(OUT_DIR);
  ensureDir(OUT_DIR);

  log('复制页面模板 site/ -> docs/ ...');
  copyDir(SITE_DIR, OUT_DIR);

  // 关闭 Jekyll 处理，避免下划线开头的文件被忽略，同时加快部署
  fs.writeFileSync(path.join(OUT_DIR, '.nojekyll'), '');

  const categories = fs
    .readdirSync(CONTENT_DIR, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => entry.name)
    .filter((name) => name !== 'map');

  ensureDir(path.join(OUT_DIR, 'data', 'blogs'));

  let totalPosts = 0;
  for (const category of categories) {
    const blogs = collectCategory(category);
    const outFile = path.join(OUT_DIR, 'data', 'blogs', `${category}.json`);
    fs.writeFileSync(outFile, JSON.stringify(blogs, null, 2), 'utf8');
    for (const blog of blogs) {
      copyPost(category, blog.id);
      totalPosts += 1;
    }
    log(`分类 ${category}: ${blogs.length} 篇文章`);
  }

  const mapSrc = path.join(CONTENT_DIR, 'map', 'data', 'asset', 'mapdata.json');
  if (copyFileIfExists(mapSrc, path.join(OUT_DIR, 'data', 'map-data.json'))) {
    log('已写入 data/map-data.json');
  } else {
    console.warn('[build] 未找到地图数据 mapdata.json');
  }

  log(`完成：共 ${totalPosts} 篇文章，输出目录 docs/`);
}

main();
