/**
 * 博客列表页面模块
 * 负责加载和显示博客列表，处理列表项的点击事件
 */

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

function getCategoryFromPath(pathname) {
  if (pathname.includes('/info')) return 'info';
  if (pathname.includes('/study')) return 'study';
  if (pathname.includes('/log')) return 'log';
  if (pathname.includes('/share')) return 'share';
  return '';
}

function setListLoading(blogList) {
  blogList.innerHTML = `<div class="loading">${escapeHtml(t('list.loading', '加载中...'))}</div>`;
}

function setListEmpty(blogList) {
  blogList.innerHTML = `<div class="loading">${escapeHtml(t('list.empty', '暂无内容'))}</div>`;
}

function setListError(blogList) {
  blogList.innerHTML = `<div class="error">${escapeHtml(t('list.error', '加载失败，请稍后再试'))}</div>`;
}

async function loadBlogList(category) {
  const blogList = document.getElementById('blogList');
  if (!blogList) return;

  const lang = getSiteLang();
  setListLoading(blogList);

  try {
    const blogs = await BlogAPI.getBlogList(category, lang);
    if (blogs.length === 0) {
      setListEmpty(blogList);
      return;
    }

    blogList.innerHTML = '';
    blogs.forEach((blog) => {
      const blogItem = createBlogItem(blog, category, lang);
      blogList.appendChild(blogItem);
    });
  } catch (error) {
    setListError(blogList);
    console.error('Error loading blogs:', error);
  }
}

/**
 * 初始化博客列表
 * 根据当前页面 URL 确定分类，然后加载相应的博客列表
 */
function initBlogList() {
  const currentPath = window.location.pathname;
  const category = getCategoryFromPath(currentPath);
  
  if (!category) return; // 如果不是列表页面，直接返回

  loadBlogList(category);

  document.addEventListener('site-lang-change', () => {
    loadBlogList(category);
  });
}

/**
 * 创建单个博客列表项
 * @param {Object} blog - 博客数据对象
 * @param {string} category - 博客分类
 * @param {string} lang - 当前站点语言
 * @returns {HTMLElement} 博客列表项 DOM 元素
 */
function createBlogItem(blog, category, lang) {
  const item = document.createElement('div');
  item.className = 'blog-item';
  const noExcerpt = t('list.noExcerpt', '暂无摘要');
  const excerptText = blog.excerpt ? `${blog.excerpt}...` : noExcerpt;
  
  item.innerHTML = `
    <h2>${escapeHtml(blog.title)}</h2>
    <div class="meta-row">
      <div class="date">${formatDate(blog.date, lang)}</div>
      <div class="tag">${escapeHtml('# ' + blog.tag)}</div>
    </div>
    <div class="excerpt">${escapeHtml(excerptText)}</div>
  `;
  
  // 添加点击事件，导航到详情页
  item.addEventListener('click', () => {
    window.location.href = `blog.html?category=${category}&id=${blog.id}`;
  });
  
  return item;
}

// 在 DOM 加载完成时初始化列表
document.addEventListener('DOMContentLoaded', initBlogList);
