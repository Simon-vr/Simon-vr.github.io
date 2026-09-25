/**
 * 主题管理模块
 * 负责深色/浅色主题的切换和存储
 */

function initTheme() {
  const themeToggle = document.getElementById('themeToggle');
  const themeIcon = document.getElementById('themeIcon');
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

    function syncGiscusTheme(theme) {
      const giscusFrame = document.querySelector('iframe.giscus-frame');
      if (!giscusFrame) return;

      const giscusTheme = theme === 'dark' ? 'dark_dimmed' : 'light';
      giscusFrame.contentWindow.postMessage(
        { giscus: { setConfig: { theme: giscusTheme } } },
        'https://giscus.app'
      );
    }

  
  if (!themeToggle || !themeIcon) return;
  
  /**
   * 获取初始主题
   * 优先级：localStorage > 系统偏好 > 默认浅色
   * @returns {string} 主题名称（light 或 dark）
   */
  function getInitialTheme() {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
      return savedTheme;
    }
    
    // 检测系统主题偏好
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark';
    }

    // 默认站点首访使用夜间主题
    return 'dark';
  }
  
  /**
   * 应用主题到页面
   * @param {string} theme - 主题名称（light 或 dark）
   */
  function applyTheme(theme) {
    const root = document.documentElement;
    
    if (theme === 'dark') {
      root.setAttribute('data-theme', 'dark');
      themeIcon.textContent = '☀️';
      themeToggle.setAttribute('title', getThemeToggleTitle('dark'));
    } else {
      root.setAttribute('data-theme', 'light');
      themeIcon.textContent = '🌙';
      themeToggle.setAttribute('title', getThemeToggleTitle('light'));
    }
    
    localStorage.setItem('theme', theme);
    syncGiscusTheme(theme);
  }
  
  /**
   * 切换主题
   */
  function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    applyTheme(newTheme);
  }
  
  // 初始化主题
  const initialTheme = getInitialTheme();
  applyTheme(initialTheme);
  
  // 绑定切换事件
  themeToggle.addEventListener('click', () => {
    toggleTheme();
    localStorage.setItem('theme-manual', 'true');
  });
  
  // 监听系统主题变化（仅在用户未手动设置时）
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

// 页面加载时初始化主题（在 HTML 脚本前执行以防止闪烁）
document.addEventListener('DOMContentLoaded', initTheme);
