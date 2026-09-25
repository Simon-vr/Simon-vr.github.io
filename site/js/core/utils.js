/**
 * 工具函数模块 - 提供通用的辅助函数
 */

/**
 * HTML转义 - 防止 XSS 攻击
 * @param {string} text - 需要转义的文本
 * @returns {string} 转义后的HTML
 */
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

/**
 * 格式化日期显示
 * @param {string} dateString - ISO格式的日期字符串
 * @param {string} lang - 语言（cn 或 en）
 * @returns {string} 格式化后的日期（如 "March 7, 2026"）
 */
function formatDate(dateString, lang = 'cn') {
  const date = new Date(dateString);
  const locale = lang === 'en' ? 'en-US' : 'zh-CN';
  return date.toLocaleDateString(locale, {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  });
}

/**
 * 处理 HTML 内容中的资源路径
 * 将相对路径转换为 API 路径
 * @param {string} html - 原始HTML内容
 * @param {string} category - 博客分类
 * @param {string} id - 博客ID
 * @returns {string} 处理后的HTML
 */
function processHTMLContent(html, category, id) {
  const tempDiv = document.createElement('div');
  tempDiv.innerHTML = html;
  
  // 处理图片、视频、音频的 src 属性（静态构建后资源位于 content/<category>/<id>/assets/）
  const basePath = `content/${category}/${id}/assets`;
  
  tempDiv.querySelectorAll('img, video, audio').forEach(element => {
    const src = element.getAttribute('src');
    if (src && /^\.?\/?assets[\/]/i.test(src)) {
      // 提取文件名
      const filename = src.match(/[\/\\]([^\/\\]+)$/)?.[1] || src;
      element.src = basePath + '/' + filename;
    }
    
    // 处理 video 和 audio 中的 source 标签
    element.querySelectorAll('source').forEach(source => {
      const sourceSrc = source.getAttribute('src');
      if (sourceSrc && /^\.?\/?assets[\/]/i.test(sourceSrc)) {
        const filename = sourceSrc.match(/[\/\\]([^\/\\]+)$/)?.[1] || sourceSrc;
        source.src = basePath + '/' + filename;
      }
    });
  });
  
  return tempDiv.innerHTML;
}
