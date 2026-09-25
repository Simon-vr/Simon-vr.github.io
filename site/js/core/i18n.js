/**
 * 站点多语言模块
 * 负责全站语言状态、文案映射和页面内文案替换
 */

(function initSiteI18n() {
  const STORAGE_KEY = 'site-lang';
  const SUPPORTED_LANGS = new Set(['cn', 'en']);

  const messages = {
    nav: {
      index: { cn: '起点', en: 'Index' },
      log: { cn: '日志', en: 'Log' },
      map: { cn: '旅行', en: 'Map' },
      share: { cn: '分享', en: 'Share' },
      study: { cn: '资料', en: 'Study' },
      project: { cn: '项目', en: 'Project' },
      info: { cn: '关于', en: 'Connect' },
    },
    list: {
      loading: { cn: '加载中...', en: 'Loading...' },
      empty: { cn: '暂无内容', en: 'No entries yet.' },
      error: { cn: '加载失败，请稍后再试', en: 'Failed to load. Please try again later.' },
      noExcerpt: { cn: '暂无摘要', en: 'No excerpt yet' },
    },
    detail: {
      back: { cn: '← 返回', en: '< Back' },
      paramError: { cn: '参数错误', en: 'Invalid parameters' },
      blogLoadError: { cn: '加载失败，请稍后再试', en: 'Failed to load blog. Please try again later.' },
      contentLoading: { cn: '加载中...', en: 'Loading...' },
      contentError: { cn: '加载内容失败，请稍后再试', en: 'Failed to load content. Please try again later.' },
      claimerCn: { cn: '以上内容仅代表个人阶段性观点。转载内容会尽量保留出处，并仅用于学习与分享。', en: 'The views above are personal and time-bound. Reposted materials keep attribution when possible and are shared for learning only.' },
      claimerEn: { cn: 'Writing is an open loop. Feel free to challenge, supplement, or correct.', en: 'Writing is an open loop. Feel free to challenge, supplement, or correct.' },
      discussion: { cn: '留言', en: 'Discussion' },
      langCn: { cn: '中文', en: 'Chinese' },
      langEn: { cn: 'English', en: 'English' },
    },
    page: {
      log: {
        title: { cn: '日志', en: 'Log' },
        subtitle: { cn: '此处与功名进取毫不相关也', en: 'What follows won\'t help you climb any ladder.   --Tiangong Kaiwu' },
      },
      map: {
        title: { cn: '足迹', en: 'Map' },
        subtitle: { cn: '我所经过的地方', en: 'A good traveler has no fixed plans and is not intent on arriving.' },
      },
      share: {
        title: { cn: '分享', en: 'Share' },
        subtitle: { cn: '好东西给好人看', en: 'Доброе дело — доброму человеку' },
      },
      study: {
        title: { cn: '资料', en: 'Study' },
        subtitle: { cn: '学习资料 [普通|不可丢弃]: 在进入必要的竞争性场合前使用，以证明自身在社会中仍存在某种被利用价值。冷却时间：下一次焦虑发作之前被动触发。', en: 'Study Materials [Common Item | Soulbound]: Consume before boss fights (exams, interviews) to avoid the "You Have No Value" debuff. Cooldown: until the next anxiety spiral.' },
      },
      project: {
        title: { cn: '项目', en: 'Project' },
        subtitle: { cn: '从零开始的一些工程实践。', en: 'Some engineering projects built from scratch.' },
      },
      info: {
        title: { cn: '联系', en: 'Connect' },
        subtitle: { cn: '欢迎来信交流：simony@tutamail.com', en: 'Reach out at: simony@tutamail.com' },
        subtitle2: { cn: '也欢迎在文章下方评论区留下观点与补充。', en: 'You are also welcome to leave your views in the comment section.' },
      },
    },
    comments: {
      disabledHint: {
        cn: '评论系统暂未启用',
        en: 'Comments are disabled.',
      },
    },
    theme: {
      toLight: { cn: '切换到日间模式', en: 'Switch to light mode' },
      toDark: { cn: '切换到夜间模式', en: 'Switch to dark mode' },
    },
    site: {
      switchToCn: { cn: '切换到中文', en: 'Switch to Chinese' },
      switchToEn: { cn: 'Switch to English', en: 'Switch to English' },
    },
  };

  let currentLang = null;

  function normalizeLang(lang) {
    return SUPPORTED_LANGS.has(lang) ? lang : 'en';
  }

  function getByPath(obj, keyPath) {
    return String(keyPath || '')
      .split('.')
      .reduce((acc, key) => (acc && acc[key] !== undefined ? acc[key] : undefined), obj);
  }

  function getSiteLang() {
    if (currentLang) {
      return currentLang;
    }

    const saved = localStorage.getItem(STORAGE_KEY);
    currentLang = normalizeLang(saved);
    return currentLang;
  }

  function t(key, lang = getSiteLang()) {
    const entry = getByPath(messages, key);
    if (!entry || typeof entry !== 'object') {
      return '';
    }

    return entry[normalizeLang(lang)] || entry.cn || '';
  }

  function applyI18n(lang = getSiteLang()) {
    const normalized = normalizeLang(lang);
    const htmlLang = normalized === 'en' ? 'en' : 'zh-CN';
    document.documentElement.setAttribute('lang', htmlLang);

    document.querySelectorAll('[data-i18n]').forEach((element) => {
      const key = element.getAttribute('data-i18n');
      const value = t(key, normalized);
      if (value) {
        element.textContent = value;
      }
    });

    document.querySelectorAll('[data-i18n-title]').forEach((element) => {
      const key = element.getAttribute('data-i18n-title');
      const value = t(key, normalized);
      if (value) {
        element.setAttribute('title', value);
      }
    });

    document.querySelectorAll('[data-lang-block]').forEach((element) => {
      const blockLang = normalizeLang(element.getAttribute('data-lang-block'));
      element.hidden = blockLang !== normalized;
    });
  }

  function setSiteLang(lang, options = {}) {
    const normalized = normalizeLang(lang);
    const persist = options.persist !== false;
    const emit = options.emit !== false;
    const previous = getSiteLang();

    currentLang = normalized;
    if (persist) {
      localStorage.setItem(STORAGE_KEY, normalized);
    }

    applyI18n(normalized);

    if (emit && previous !== normalized) {
      document.dispatchEvent(new CustomEvent('site-lang-change', {
        detail: { lang: normalized }
      }));
    }

    return normalized;
  }

  function toggleSiteLang() {
    const nextLang = getSiteLang() === 'en' ? 'cn' : 'en';
    return setSiteLang(nextLang);
  }

  function initI18n() {
    applyI18n(getSiteLang());
  }

  window.SiteI18n = {
    t,
    messages,
    normalizeLang,
    getSiteLang,
    setSiteLang,
    toggleSiteLang,
    applyI18n,
    initI18n,
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initI18n);
  } else {
    initI18n();
  }
})();
