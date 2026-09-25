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
  script.setAttribute('data-mapping', giscusConfig.mapping || 'pathname');
  script.setAttribute('data-strict', giscusConfig.strict || '0');
  script.setAttribute('data-reactions-enabled', giscusConfig.reactionsEnabled || '1');
  script.setAttribute('data-emit-metadata', giscusConfig.emitMetadata || '0');
  script.setAttribute('data-input-position', giscusConfig.inputPosition || 'top');
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
