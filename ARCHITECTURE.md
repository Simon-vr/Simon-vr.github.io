# simon-vr 架构说明

版本：v3.0.0（纯静态版）

## 1. 项目定位

`simon-vr` 是一个以「结构化思考 + 生命力脉冲」为设计主题的个人写作网站。
当前版本为**纯静态站点**：所有数据在构建期生成，运行时不再依赖任何后端。

## 2. 架构总览

```text
Source (content/)                     Templates (site/)
        │                                    │
        └──────────── scripts/build.js ──────┘
                            │
                            ▼
                         docs/  ──►  GitHub Pages（https://simon-vr.github.io/）
                            ├── *.html / css / js
                            ├── data/blogs/<category>.json
                            ├── data/map-data.json
                            └── content/<category>/<id>/{text_CN,text_EN}.html + assets/
```

## 3. 从前端到数据的映射

| 旧接口 | 静态替代 |
| --- | --- |
| `GET /api/blogs/:category` | `data/blogs/<category>.json` |
| `GET /api/blog/:category/:id` | 从 `data/blogs/<category>.json` 中按 `id` 查找 |
| `GET /api/blog-content/:category/:id/:lang` | `content/<category>/<id>/text_CN.html` 或 `text_EN.html` |
| `GET /api/blog-resource/:category/:id/:fn` | `content/<category>/<id>/assets/<fn>` |
| `GET /api/map-data` | `data/map-data.json` |

双语处理：列表 JSON 同时包含 `title/title_en`、`tag/tag_en`、`excerpt/excerpt_en`，
前端 `BlogAPI.localize()` 根据当前语言选择，英文缺失时回退中文。

## 4. 关键模块

```text
site/js/core/
  api.js           数据访问（全部改为静态文件 + 相对路径）
  i18n.js          语言状态与文案
  state.js         详情页状态
  site-config.js   品牌 / siteUrl / giscus 配置
  utils.js         日期、转义、正文资源路径处理
site/js/modules/
  header.js        导航注入（相对 .html 链接）
  list.js          列表页
  detail.js        详情页（语言切换 + 正文加载）
  comments.js      giscus 挂载（未配置时展示提示）
  theme.js         明暗主题
```

## 5. 构建流程

`scripts/build.js`：

1. 清空并重建 `docs/`。
2. 复制 `site/` 全部模板到 `docs/`，写入 `.nojekyll`。
3. 遍历 `content/<category>/<id>/metadata.json`，生成
   `docs/data/blogs/<category>.json`（按日期倒序）。
4. 复制每篇文章的 `text_CN.html` / `text_EN.html` 与 `assets/`。
5. 复制 `content/map/data/asset/mapdata.json` 到 `docs/data/map-data.json`。

## 6. 内容数据结构（保持不变）

```text
content/[category]/[id]/
  metadata.json
  text_CN.html
  text_EN.html
  handscript.md
  image/handscript/   # md2html 的图片来源
  assets/             # 线上引用的资源
```

## 7. 发布流程

- 内容工具：`crtblog.js`（新建）→ `md2html.js`（Markdown 转 HTML）→ `translate.js`（翻译）。
- 一键发布：`scripts/publish.js` = `md2html`（可选）→ `build` → `git commit/push`。
- GitHub Pages 监听 `main` 分支 `/docs` 目录，push 后自动部署。

## 8. 视觉系统

风格主轴保持为 *The Grid & The Spark*：网格、硬边框、信息分层，配合心跳脉冲与
轻微动态。主题变量集中在 `site/css/style.css` 的 `:root` 与 `[data-theme="dark"]`。
