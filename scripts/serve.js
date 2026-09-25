#!/usr/bin/env node
/**
 * 本地静态预览服务器（零依赖）
 *
 * 用法：
 *   node scripts/serve.js [port]
 *   npm run serve
 *
 * 打开 http://localhost:8080 预览 docs/ 下的最终效果。
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const DOCS_DIR = path.join(ROOT, 'docs');
const PORT = Number(process.argv[2] || process.env.PORT || 8080);

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.bmp': 'image/bmp',
  '.avif': 'image/avif',
  '.ico': 'image/x-icon',
  '.mp4': 'video/mp4',
  '.webm': 'video/webm',
  '.mp3': 'audio/mpeg',
  '.wav': 'audio/wav',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ttf': 'font/ttf',
  '.pdf': 'application/pdf',
  '.txt': 'text/plain; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8'
};

function send(res, status, body, contentType) {
  res.writeHead(status, { 'Content-Type': contentType || 'text/plain; charset=utf-8' });
  res.end(body);
}

const server = http.createServer((req, res) => {
  const urlPath = decodeURIComponent((req.url || '/').split('?')[0].split('#')[0]);
  let filePath = path.join(DOCS_DIR, urlPath);

  // 防止路径穿越
  if (!filePath.startsWith(DOCS_DIR)) {
    return send(res, 403, 'Forbidden');
  }

  // 目录 -> index.html
  if (fs.existsSync(filePath) && fs.statSync(filePath).isDirectory()) {
    filePath = path.join(filePath, 'index.html');
  }

  // GitHub Pages 风格：无扩展名时尝试补 .html
  if (!fs.existsSync(filePath) && !path.extname(filePath)) {
    const withHtml = `${filePath}.html`;
    if (fs.existsSync(withHtml)) {
      filePath = withHtml;
    }
  }

  if (!fs.existsSync(filePath)) {
    const notFound = path.join(DOCS_DIR, '404.html');
    if (fs.existsSync(notFound)) {
      res.writeHead(404, { 'Content-Type': 'text/html; charset=utf-8' });
      return res.end(fs.readFileSync(notFound));
    }
    return send(res, 404, '404 Not Found');
  }

  const ext = path.extname(filePath).toLowerCase();
  res.writeHead(200, {
    'Content-Type': MIME_TYPES[ext] || 'application/octet-stream',
    'Cache-Control': 'no-cache'
  });
  fs.createReadStream(filePath).pipe(res);
});

server.listen(PORT, () => {
  console.log(`[serve] 预览地址: http://localhost:${PORT}`);
  console.log(`[serve] 根目录: ${DOCS_DIR}`);
});
