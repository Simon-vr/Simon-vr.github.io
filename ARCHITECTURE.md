# simon-vr 架构说明

版本：v3.1.0（纯静态 · 单目录）

## 1. 项目定位

`simon-vr` 是一个以「结构化思考 + 生命力脉冲」为设计主题的个人写作网站。
当前版本为**纯静态站点**：`docs/` 既是源、也是 GitHub Pages 的产物，运行时没有任何后端。

## 2. 单目录架构

```text
docs/                    ← 唯一的站点目录（源 = 产物）
  ├── *.html             index / blog / log / share / study / map / project / info
  ├── icon.png           favicon
  ├── css/style.css
  ├── js/core|modules/
  ├── data/map-data.json
  └── content/
      ├── <category>/
      │   ├── index.json            该分类的文章 id 列表（由脚本生成）
      │   └── <id>/
      │       ├── metadata.json     标题/日期/标签/摘要（中英字段）
      │       ├── text_CN.html      正文
      │       ├── text_EN.html
      │       └── assets/           图片、音视频等
      └── map/data/asset/mapdata.json
```

没有构建期的“源 → 产物”复制：改 `docs/` 下的 HTML/CSS/JS 立即生效。

## 3. 数据访问（无后端）

前端不再依赖 Express 接口，改为按需读取静态文件：

| 用途 | 路径 |
| --- | --- |
| 文章列表 | `content/<category>/index.json` + 各篇 `metadata.json` |
| 文章详情 | `content/<category>/<id>/metadata.json` |
| 正文 | `content/<category>/<id>/text_CN.html` / `text_EN.html` |
| 资源 | `content/<category>/<id>/assets/<fn>` |
| 地图 | `data/map-data.json` |

双语处理：`metadata.json` 同时包含 `title/title_en`、`tag/tag_en`、`excerpt/excerpt_en`，
前端 `BlogAPI.localize()` 按当前语言选择，英文缺失时回退中文。

## 4. 关键模块

```text
docs/js/core/
  api.js           静态数据访问（index.json + metadata.json）
  i18n.js          语言状态与文案
  state.js         详情页状态
  site-config.js   品牌 / siteUrl / giscus 配置
  utils.js         日期、转义、正文资源路径处理
docs/js/modules/
  header.js        导航注入（相对 .html 链接）
  list.js          列表页（读取 index.json）
  detail.js        详情页（语言切换 + 正文加载）
  comments.js      giscus 挂载（按 category/id 建讨论）
  theme.js         明暗主题
```

## 5. scripts/ 的职责

`docs/` 是源，所以脚本只做「维护」，不做复制：

- `build.js`：扫描 `docs/content/<category>/<id>/metadata.json`，按日期倒序写出
  `index.json`；同步 `data/map-data.json`；清理 `handscript.md`、`image/` 等
  不需要发布的写作中间产物；写入 `.nojekyll`。
- `new-post.js`：建目录 → 生成索引。
- `publish.js`：`md2html`（可选）→ `build` → `git commit/push`。
- `serve.js`：本地预览 `docs/`。

内容工具在 `utils/`：`crtblog.js`（新建）、`md2html.js`（Markdown → `text_CN.html`）、
`translate.js`（生成 `text_EN.html` 与 `_en` 字段），三者均直接操作 `docs/content/`。

## 6. 发布流程

GitHub Pages 监听 `main` 分支 `/docs` 目录，push 后自动部署：

```bash
npm run publish -- <category> <id>   # 一键：md2html + 索引 + 提交推送
```

## 7. 视觉系统

风格主轴为 *The Grid & The Spark*：扁平直角的毛玻璃卡片、固定网格背景、
心跳脉冲与轻微动态。主题变量集中在 `docs/css/style.css` 的
`:root`（light 羊皮纸）与 `[data-theme="dark"]`（深青）。

## 8. 评论

giscus 配置在 `docs/js/core/site-config.js`，使用 `specific` 映射 +
`category/id` 作为 term，使每篇文章拥有独立讨论。
