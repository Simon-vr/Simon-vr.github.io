#!/usr/bin/env node
/**
 * 一键新建文章
 *
 * 用法：
 *   node scripts/new-post.js <category> <id>
 *   npm run new -- <category> <id>
 *
 * 示例：
 *   npm run new -- log my_new_post
 *
 * 该命令会：
 *   1. 在 content/<category>/<id>/ 下生成 metadata.json、text_CN.html、text_EN.html、handscript.md
 *   2. 重新构建 docs/
 *
 * 后续写作流程：
 *   在 handscript.md 里写 Markdown，然后执行：
 *   npm run publish -- <category> <id>
 */

const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const ROOT = path.resolve(__dirname, '..');

function run(command) {
  execSync(command, { cwd: ROOT, stdio: 'inherit' });
}

function main() {
  const [category, id] = process.argv.slice(2);
  if (!category || !id) {
    console.error('用法: node scripts/new-post.js <category> <id>');
    console.error('示例: node scripts/new-post.js log my_new_post');
    process.exit(1);
  }

  run(`node ./utils/crtblog.js ${category} ${id}`);

  const postDir = path.join(ROOT, 'content', category, id);
  if (!fs.existsSync(postDir)) {
    console.error(`创建失败：${postDir} 不存在`);
    process.exit(1);
  }

  run('node ./scripts/build.js');

  console.log('\n下一步：');
  console.log(`  1. 编辑 content/${category}/${id}/handscript.md 写 Markdown 正文`);
  console.log(`  2. 编辑 content/${category}/${id}/metadata.json 填写标题/日期/标签`);
  console.log(`  3. 执行 npm run publish -- ${category} ${id} 生成正文、构建并推送`);
}

main();
