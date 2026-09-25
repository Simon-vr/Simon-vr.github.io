const fs = require('fs');
const path = require('path');

// 创建文件夹并初始化文件
async function createFolder(className, folderName) {
  // 检查className文件夹是否存在
  const classPath = path.join(path.dirname(__dirname), 'docs', 'content', className);
  if (!fs.existsSync(classPath)) {
    console.error(`className ${className} 不存在`);
    return;
  }

  // 检查folderName文件夹是否已存在
  const targetPath = path.join(classPath, folderName);
  if (fs.existsSync(targetPath)) {
    console.error(`folderName ${folderName} 已存在`);
    return;
  }

  // 创建文件夹
  fs.mkdirSync(targetPath);

  // 初始化 metadata.json
  const metadataPath = path.join(targetPath, 'metadata.json');
  const now = new Date();
  const formattedDate = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
  const metadict = {
    title: 'unknown',
    title_en: 'unknown',
    date: formattedDate,
    tag: 'unknown',
    tag_en: 'unknown',
    excerpt: 'unknown',
    excerpt_en: 'unknown',
    likes: 0,
  };
  const metadataContent = JSON.stringify(metadict, null, 2);
  fs.writeFileSync(metadataPath, metadataContent);

  // 初始化 text_CN.html
  htmlPath = path.join(targetPath, 'text_CN.html');
  htmlContent = `
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Document</title>
    </head>
    <body>
      <h1>欢迎来到 simon-vr, 内容为空</h1>
    </body>
    </html>
  `;
  fs.writeFileSync(htmlPath, htmlContent);
  htmlPath = path.join(targetPath, 'text_EN.html');
  htmlContent = `
    <!DOCTYPE html>
    <html lang="en-EN">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Document</title>
    </head>
    <body>
      <h1>English version is not available</h1>
    </body>
    </html>
  `;
  fs.writeFileSync(htmlPath, htmlContent);
  fs.writeFileSync(path.join(targetPath, 'handscript.md'), "data");
  console.log(`文件夹 ${folderName} 已成功创建`);
}

// 从命令行参数获取文件夹名称
const className = process.argv[2];
const folderName = process.argv[3];
if (!className || !folderName) {
  console.error('请提供分类, 文件夹id名称');
  process.exit(1);
}

createFolder(className, folderName);