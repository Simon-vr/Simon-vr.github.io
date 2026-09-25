#!/usr/bin/env node
/**
 * 一键发布脚本
 *
 * 用法：
 *   node scripts/publish.js                 # 更新索引并 git 提交推送
 *   node scripts/publish.js <category> <id> # 先把 handscript.md 转成 HTML，再更新索引推送
 *
 * 示例：
 *   npm run publish
 *   npm run publish -- log my_new_post
 *
 * 说明：GitHub Pages 的部署由 git push 自动触发，因此“发布”= 构建 + 推送。
 */

const { execSync } = require('child_process');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');

function run(command, options = {}) {
  console.log(`\n$ ${command}`);
  execSync(command, { cwd: ROOT, stdio: 'inherit', ...options });
}

function runSafe(command) {
  try {
    return execSync(command, { cwd: ROOT, encoding: 'utf8' }).trim();
  } catch (error) {
    return '';
  }
}

function main() {
  const args = process.argv.slice(2);
  const [category, id] = args;

  // 1. 可选：把 Markdown 草稿转换为 HTML（并同步图片到 assets/）
  if (category && id) {
    run(`node ./utils/md2html.js ${category} ${id}`);
  } else if (category || id) {
    console.error('用法: node scripts/publish.js [<category> <id>]');
    process.exit(1);
  }

  // 2. 更新索引与数据
  run('node ./scripts/build.js');

  // 3. git 提交与推送
  const inGitRepo = runSafe('git rev-parse --is-inside-work-tree') === 'true';
  if (!inGitRepo) {
    console.log('\n[发布] 当前目录不是 git 仓库，已只完成索引更新。');
    console.log('[发布] 如需自动推送，请先执行 git init 并添加远程仓库。');
    return;
  }

  const changes = runSafe('git status --porcelain');
  if (!changes) {
    console.log('\n[发布] 没有检测到改动，跳过提交。');
    return;
  }

  const message = category && id
    ? `publish: ${category}/${id}`
    : `publish: ${new Date().toISOString().slice(0, 16).replace('T', ' ')}`;

  run('git add -A');
  run(`git commit -m "${message}"`);

  const remotes = runSafe('git remote');
  if (remotes.split('\n').map((item) => item.trim()).includes('origin')) {
    run('git push origin HEAD');
  } else {
    console.log('\n[发布] 未配置 origin 远程仓库，已只完成本地提交。');
  }
}

main();
