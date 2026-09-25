/**
 * 主题管理模块
 * 负责深色/浅色主题的切换和存储
 */

function initTheme() {
  const themeToggle = document.getElementById('themeToggle');
  const getI18nText = (key, fallback) => {
    if (window.SiteI18n && typeof window.SiteI18n.t === 'function') {
      return window.SiteI18n.t(key) || fallback;
    }
    return fallback;
  };

  const getThemeToggleTitle = (theme) => {
    if (theme === 'dark') {
      return getI18nText('theme.toLight', '切换到日间模式');
    }
    return getI18nText('theme.toDark', '切换到夜间模式');
  };

  function getThemeIconSVG(theme) {
    if (theme === 'dark') {
      return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>`;
    }
    return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>`;
  }

  function syncGiscusTheme(theme) {
    const giscusFrame = document.querySelector('iframe.giscus-frame');
    if (!giscusFrame) return;

    const giscusTheme = theme === 'dark' ? 'dark_dimmed' : 'light';
    giscusFrame.contentWindow.postMessage(
      { giscus: { setConfig: { theme: giscusTheme } } },
      'https://giscus.app'
    );
  }

  if (!themeToggle) return;

  function getInitialTheme() {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
      return savedTheme;
    }
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark';
    }
    return 'dark';
  }

  function applyTheme(theme) {
    const root = document.documentElement;

    if (theme === 'dark') {
      root.setAttribute('data-theme', 'dark');
      themeToggle.setAttribute('title', getThemeToggleTitle('dark'));
    } else {
      root.setAttribute('data-theme', 'light');
      themeToggle.setAttribute('title', getThemeToggleTitle('light'));
    }

    themeToggle.innerHTML = getThemeIconSVG(theme);
    localStorage.setItem('theme', theme);
    syncGiscusTheme(theme);
  }

  function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    applyTheme(newTheme);
  }

  const initialTheme = getInitialTheme();
  applyTheme(initialTheme);

  themeToggle.addEventListener('click', () => {
    toggleTheme();
    localStorage.setItem('theme-manual', 'true');
  });

  if (window.matchMedia) {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');

    if (!localStorage.getItem('theme-manual')) {
      mediaQuery.addEventListener('change', (e) => {
        if (!localStorage.getItem('theme-manual')) {
          applyTheme(e.matches ? 'dark' : 'light');
        }
      });
    }
  }

  document.addEventListener('site-lang-change', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme') === 'dark'
      ? 'dark'
      : 'light';
    themeToggle.setAttribute('title', getThemeToggleTitle(currentTheme));
  });
}

document.addEventListener('DOMContentLoaded', initTheme);
