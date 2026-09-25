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
  return currentLang === 'en' ? '中' : 'EN';
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
  document.querySelectorAll('.nav-links a').forEach((link) => {
    link.classList.toggle('active', link.getAttribute('data-page') === currentPage);
  });
}

function getThemeIconSVG(theme) {
  // Sun icon for dark mode (click to go light), Moon icon for light mode (click to go dark)
  if (theme === 'dark') {
    // Sun
    return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>`;
  }
  // Moon
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>`;
}

function getGlobeIconSVG() {
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="2" y1="12" x2="22" y2="12"></line><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path></svg>`;
}

function renderHeader() {
  if (document.querySelector('nav')) {
    updateHeaderLanguage();
    markActiveLink();
    return;
  }

  const currentLang = getSiteLang();
  const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';

  const navHTML = `
    <nav>
      <div class="container">
        <a href="index.html" class="logo">simon-vr</a>
        <div class="nav-right">
          <ul class="nav-links">
            ${createNavLinksHTML()}
          </ul>
          <button id="siteLangToggle" class="icon-btn" title="${getLangToggleTitle(currentLang)}">
            ${getGlobeIconSVG()}
            <span id="siteLangText" style="font-size:0.6rem;margin-left:0.1rem;font-family:var(--mono-font);letter-spacing:0.02em">${getLangToggleLabel(currentLang)}</span>
          </button>
          <button id="themeToggle" class="icon-btn" title="切换主题">
            ${getThemeIconSVG(currentTheme)}
          </button>
        </div>
      </div>
    </nav>
  `;

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

  if (window.SiteI18n && typeof window.SiteI18n.applyI18n === 'function') {
    window.SiteI18n.applyI18n(getSiteLang());
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', renderHeader);
} else {
  renderHeader();
}
