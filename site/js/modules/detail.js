/**
 * 博客详情页面模块
 * 负责加载和显示博客详情，处理语言切换与评论区
 */

let commentsInitialized = false;
let requestSerial = 0;
let langButtonsBound = false;

function getSiteLang() {
  if (window.SiteI18n && typeof window.SiteI18n.getSiteLang === 'function') {
    return window.SiteI18n.getSiteLang();
  }
  return 'cn';
}

function t(key, fallback = '') {
  if (window.SiteI18n && typeof window.SiteI18n.t === 'function') {
    const value = window.SiteI18n.t(key);
    return value || fallback;
  }
  return fallback;
}

function showBlogError(message) {
  const blogContent = document.getElementById('blogContent');
  if (blogContent) {
    blogContent.innerHTML = `<div class="error">${escapeHtml(message)}</div>`;
  }
}

function updateLanguageButtons() {
  document.querySelectorAll('.lang-btn').forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.lang === BlogState.currentLang);
  });

  const cnButton = document.querySelector('.lang-btn[data-lang="cn"]');
  const enButton = document.querySelector('.lang-btn[data-lang="en"]');
  if (cnButton) {
    cnButton.textContent = t('detail.langCn', '中文');
  }
  if (enButton) {
    enButton.textContent = t('detail.langEn', 'English');
  }
}

function setupLanguageButtons() {
  const langSwitcher = document.getElementById('langSwitcher');
  if (langSwitcher) {
    langSwitcher.style.display = 'flex';
  }

  updateLanguageButtons();

  if (!langButtonsBound) {
    document.querySelectorAll('.lang-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        switchLanguage(btn.dataset.lang, { syncSiteLang: true });
      });
    });
    langButtonsBound = true;
  }
}

function renderBlogMeta(blog, lang) {
  document.getElementById('blogTitle').textContent = blog.title;
  document.getElementById('blogDate').textContent = formatDate(blog.date, lang);
  document.getElementById('blogTag').textContent = '# ' + blog.tag;
}

async function loadHTMLContent(lang, serial) {
  const blogContent = document.getElementById('blogContent');
  blogContent.innerHTML = `<div class="loading">${escapeHtml(t('detail.contentLoading', '加载中...'))}</div>`;

  try {
    const html = await BlogAPI.getBlogContent(BlogState.currentCategory, BlogState.currentId, lang);
    if (serial !== requestSerial) return;

    const processedHTML = processHTMLContent(html, BlogState.currentCategory, BlogState.currentId);
    blogContent.innerHTML = processedHTML;
    await renderMathJaxInBlogContent();
  } catch (error) {
    if (serial !== requestSerial) return;
    showBlogError(t('detail.contentError', '加载内容失败，请稍后再试'));
    console.error('Error loading HTML content:', error);
  }
}

async function loadBlogByLanguage(lang) {
  const serial = ++requestSerial;

  try {
    const blog = await BlogAPI.getBlogDetail(BlogState.currentCategory, BlogState.currentId, lang);
    if (serial !== requestSerial) return;

    BlogState.blogData = blog;
    renderBlogMeta(blog, lang);
    setupLanguageButtons();
    await loadHTMLContent(lang, serial);

    if (!commentsInitialized && typeof initGiscusComments === 'function') {
      initGiscusComments();
      commentsInitialized = true;
    }
  } catch (error) {
    if (serial !== requestSerial) return;
    showBlogError(t('detail.blogLoadError', '加载失败，请稍后再试'));
    console.error('Error loading blog:', error);
  }
}

/**
 * 初始化博客详情页面
 */
function initBlogDetail() {
  const urlParams = new URLSearchParams(window.location.search);
  const category = urlParams.get('category');
  const id = urlParams.get('id');
  
  if (!category || !id) {
    showBlogError(t('detail.paramError', '参数错误'));
    return;
  }
  
  // 更新全局状态
  BlogState.update({
    currentCategory: category,
    currentId: id,
    currentLang: getSiteLang(),
  });

  switchLanguage(BlogState.currentLang, { syncSiteLang: false });

  document.addEventListener('site-lang-change', (event) => {
    const siteLang = event && event.detail && event.detail.lang
      ? event.detail.lang
      : getSiteLang();

    updateLanguageButtons();
    if (siteLang !== BlogState.currentLang) {
      switchLanguage(siteLang, { syncSiteLang: false });
    }
  });
}

// ========== 新增：专门的 MathJax 渲染函数 ==========
async function renderMathJaxInBlogContent() {
  if (!window.MathJax || !MathJax.startup || !MathJax.typeset) {
    return;
  }

  try {
    // 1. 等待 MathJax 完全初始化（必须等，否则渲染会失效）
    await MathJax.startup.promise;
    
    // 2. 获取博客内容容器，只渲染这个区域（性能最优）
    const blogContent = document.getElementById('blogContent');
    if (!blogContent) return;
    
    // 3. 先清除旧的渲染（防止重复渲染导致的样式问题），再重新渲染
    MathJax.typesetClear([blogContent]);
    MathJax.typeset([blogContent]);
    
    console.log('✅ MathJax 公式渲染完成');
  } catch (error) {
    console.error('❌ MathJax 渲染失败:', error);
  }
}

/**
 * 切换语言
 * @param {string} lang - 目标语言（cn 或 en）
 */
function switchLanguage(lang, options = {}) {
  const normalized = lang === 'en' ? 'en' : 'cn';
  const syncSiteLang = options.syncSiteLang !== false;

  if (normalized === BlogState.currentLang && BlogState.blogData) {
    updateLanguageButtons();
    return;
  }

  BlogState.currentLang = normalized;
  updateLanguageButtons();

  if (syncSiteLang && window.SiteI18n && typeof window.SiteI18n.setSiteLang === 'function') {
    const currentSiteLang = window.SiteI18n.getSiteLang();
    if (currentSiteLang !== normalized) {
      window.SiteI18n.setSiteLang(normalized);
    }
  }

  loadBlogByLanguage(normalized);
}

// 在 DOM 加载完成时初始化博客详情
document.addEventListener('DOMContentLoaded', initBlogDetail);
