/**
 * Header component for dynamic navigation injection
 * Injects navigation bar into all pages and sets active link based on current page
 *
 * 纯静态版本：所有链接使用相对 .html 路径，兼容根目录与子路径部署。
 */

const NAV_ITEMS = [
  { href: 'index.html', page: 'index', key: 'nav.index', fallback: 'Origin' },
  { href: 'log.html', page: 'log', key: 'nav.log', fallback: 'Journal' },
  { href: 'map.html', page: 'map', key: 'nav.map', fallback: 'Map' },
  { href: 'share.html', page: 'share', key: 'nav.share', fallback: 'Curated' },
  { href: 'study.html', page: 'study', key: 'nav.study', fallback: 'Study' },
  { href: 'project.html', page: 'project', key: 'nav.project', fallback: 'Build' },
  // { href: 'info.html', page: 'info', key: 'nav.info', fallback: 'Connect' },
];

function getSiteLang() {
  if (window.SiteI18n && typeof window.SiteI18n.getSiteLang === 'function') {
    return window.SiteI18n.getSiteLang();
  }
  return 'cn';
}

function t(key, fallback = '') {
  if (window.SiteI18n && typeof window.SiteI18n.t === 'function') {
    return window.SiteI18n.t(key) || fallback;
  }
  return fallback;
}

function getLangToggleLabel(currentLang) {
  return currentLang === 'en' ? '中文' : 'EN';
}

function getLangToggleTitle(currentLang) {
  if (currentLang === 'en') {
    return t('site.switchToCn', '切换到中文');
  }
  return t('site.switchToEn', 'Switch to English');
}

function createNavLinksHTML() {
  return NAV_ITEMS.map((item) => `
    <li>
      <a href="${item.href}" class="nav-link" data-page="${item.page}" data-nav-key="${item.key}" data-nav-fallback="${item.fallback}">
        ${item.fallback}
      </a>
    </li>
  `).join('');
}

function updateHeaderLanguage() {
  NAV_ITEMS.forEach((item) => {
    const link = document.querySelector(`.nav-link[data-nav-key="${item.key}"]`);
    if (link) {
      link.textContent = t(item.key, item.fallback);
    }
  });

  const currentLang = getSiteLang();
  const langText = document.getElementById('siteLangText');
  const langToggle = document.getElementById('siteLangToggle');
  if (langText) {
    langText.textContent = getLangToggleLabel(currentLang);
  }
  if (langToggle) {
    langToggle.setAttribute('title', getLangToggleTitle(currentLang));
  }
}

function getCurrentPageName() {
  const pathname = window.location.pathname;
  const fileName = pathname.split('/').pop();
  if (!fileName || fileName === '' || pathname.endsWith('/')) {
    return 'index';
  }
  return fileName.replace(/\.html$/, '');
}

function markActiveLink() {
  const currentPage = getCurrentPageName();
  const navLinks = document.querySelectorAll('.nav-links a');

  navLinks.forEach((link) => {
    const page = link.getAttribute('data-page');
    link.classList.toggle('active', page === currentPage);
  });
}

function renderHeader() {
  if (document.querySelector('nav')) {
    updateHeaderLanguage();
    markActiveLink();
    return;
  }

  const currentLang = getSiteLang();

  // Create navigation HTML
  const navHTML = `
    <nav>
      <div class="container">
        <a href="index.html" class="logo">simon-vr</a>
        <div class="nav-right">
          <ul class="nav-links">
            ${createNavLinksHTML()}
          </ul>
          <button id="siteLangToggle" class="site-lang-toggle" title="${getLangToggleTitle(currentLang)}">
            <span id="siteLangText">${getLangToggleLabel(currentLang)}</span>
          </button>
          <button id="themeToggle" class="theme-toggle" title="切换主题"><span id="themeIcon">🌙</span></button>
        </div>
      </div>
    </nav>
  `;

  // Insert navigation at the beginning of body
  document.body.insertAdjacentHTML('afterbegin', navHTML);

  markActiveLink();
  updateHeaderLanguage();

  const langToggle = document.getElementById('siteLangToggle');
  if (langToggle) {
    langToggle.addEventListener('click', () => {
      if (window.SiteI18n && typeof window.SiteI18n.toggleSiteLang === 'function') {
        window.SiteI18n.toggleSiteLang();
      }
    });
  }

  document.addEventListener('site-lang-change', () => {
    updateHeaderLanguage();
  });

  // 如果页面已经由 i18n 脚本初始化过，这里再次应用，确保动态注入的 header 文案也同步
  if (window.SiteI18n && typeof window.SiteI18n.applyI18n === 'function') {
    window.SiteI18n.applyI18n(getSiteLang());
  }
}

// Initialize header when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', renderHeader);
} else {
  renderHeader();
}
