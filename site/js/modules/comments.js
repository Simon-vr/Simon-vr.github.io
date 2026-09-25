function getGiscusTerm() {
  // 从 URL 提取 category/id（如 /blog.html?category=log&id=book_antarctica）
  // 返回 'log/book_antarctica' 作为 giscus 讨论标题匹配术语
  const urlParams = new URLSearchParams(window.location.search);
  const category = urlParams.get('category');
  const id = urlParams.get('id');
  if (category && id) {
    return `${category}/${id}`;
  }
  return null;
}

function isGiscusConfigured(giscusConfig) {
  if (!giscusConfig || !giscusConfig.enabled) return false;
  const hasRepoId = typeof giscusConfig.repoId === 'string' && giscusConfig.repoId.trim() !== '';
  const hasCategoryId = typeof giscusConfig.categoryId === 'string' && giscusConfig.categoryId.trim() !== '';
  return hasRepoId && hasCategoryId;
}

function getCommentsHint() {
  if (window.SiteI18n && typeof window.SiteI18n.t === 'function') {
    return window.SiteI18n.t('comments.disabledHint');
  }
  return '评论系统暂未启用。请在 site/js/core/site-config.js 中填写 giscus 配置。';
}

function renderCommentsHint() {
  const commentsRoot = document.getElementById('commentsRoot');
  if (!commentsRoot) return;
  commentsRoot.innerHTML = `<p class="comments-hint">${escapeHtml(getCommentsHint())}</p>`;
}

function initGiscusComments() {
  const commentsRoot = document.getElementById('commentsRoot');
  if (!commentsRoot) return;

  const giscusConfig = window.SiteConfig && window.SiteConfig.giscus;
  if (!isGiscusConfigured(giscusConfig)) {
    renderCommentsHint();
    return;
  }

  const siteLang = (window.SiteI18n && typeof window.SiteI18n.getSiteLang === 'function')
    ? window.SiteI18n.getSiteLang()
    : 'cn';
  const preferredLang = giscusConfig.lang || (siteLang === 'en' ? 'en' : 'zh-CN');

  const script = document.createElement('script');
  script.src = 'https://giscus.app/client.js';
  script.async = true;
  script.crossOrigin = 'anonymous';

  script.setAttribute('data-repo', giscusConfig.repo);
  script.setAttribute('data-repo-id', giscusConfig.repoId);
  script.setAttribute('data-category', giscusConfig.category);
  script.setAttribute('data-category-id', giscusConfig.categoryId);
  // 动态 term：从 URL 提取 category/id，避免所有文章共用 /blog.html 一条讨论
  // 有 term 时用 specific 模式 + 严格匹配；无参数页面回退到配置的 mapping
  const term = getGiscusTerm();
  const mapping = term ? 'specific' : (giscusConfig.mapping || 'pathname');
  script.setAttribute('data-mapping', mapping);
  if (term) {
    script.setAttribute('data-term', term);
  }
  script.setAttribute('data-strict', term ? '1' : (giscusConfig.strict || '0'));
  script.setAttribute('data-reactions-enabled', giscusConfig.reactionsEnabled || '1');
  script.setAttribute('data-emit-metadata', giscusConfig.emitMetadata || '0');
  script.setAttribute('data-input-position', giscusConfig.inputPosition || 'top');
  if (giscusConfig.loading) {
    script.setAttribute('data-loading', giscusConfig.loading);
  }
  script.setAttribute('data-theme', document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark_dimmed' : 'light');
  script.setAttribute('data-lang', preferredLang);

  commentsRoot.innerHTML = '';
  commentsRoot.appendChild(script);
}

document.addEventListener('site-lang-change', () => {
  const giscusConfig = window.SiteConfig && window.SiteConfig.giscus;
  if (!isGiscusConfigured(giscusConfig)) {
    renderCommentsHint();
  }
});
