/**
 * API 模块（纯静态版） - 统一封装所有数据访问
 *
 * GitHub Pages 没有后端，所有数据都来自构建时生成的静态文件：
 *   data/blogs/<category>.json      文章列表（含中英双语字段）
 *   content/<category>/<id>/*.html  文章正文
 *   content/<category>/<id>/assets/ 文章资源（图片/音视频）
 *   data/map-data.json              旅行地图数据
 *
 * 所有路径均为相对路径，因此站点部署在根目录或子路径下都能正常工作。
 */

const BlogAPI = {
  /**
   * 读取并解析 JSON 文件
   * @param {string} path - 相对路径
   * @returns {Promise<any>}
   */
  async fetchJson(path) {
    const response = await fetch(path);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status} (${path})`);
    }
    return await response.json();
  },

  /**
   * 根据站点语言选择元数据字段（缺失英文时回退中文）
   */
  localize(item, lang) {
    const pick = (key) => {
      if (lang === 'en') {
        const en = item[`${key}_en`];
        if (typeof en === 'string' && en.trim()) {
          return en.trim();
        }
      }
      return typeof item[key] === 'string' ? item[key] : '';
    };

    return {
      id: item.id,
      date: item.date,
      title: pick('title'),
      tag: pick('tag'),
      excerpt: pick('excerpt'),
    };
  },

  /**
   * 获取指定分类的博客列表
   * @param {string} category - 分类（study, log, share 等）
   * @param {string} lang - 语言（cn 或 en）
   * @returns {Promise<Array>} 博客列表数组
   */
  async getBlogList(category, lang = 'cn') {
    try {
      const list = await this.fetchJson(`data/blogs/${category}.json`);
      return list.map((item) => this.localize(item, lang));
    } catch (error) {
      console.error(`Error fetching blog list for ${category}:`, error);
      return [];
    }
  },

  /**
   * 获取单个博客的元数据和信息
   * @param {string} category - 分类
   * @param {string} id - 博客ID
   * @param {string} lang - 语言（cn 或 en）
   * @returns {Promise<Object>} 博客元数据
   */
  async getBlogDetail(category, id, lang = 'cn') {
    try {
      const list = await this.fetchJson(`data/blogs/${category}.json`);
      const item = list.find((entry) => entry.id === id);
      if (!item) {
        throw new Error('Blog not found');
      }
      return this.localize(item, lang);
    } catch (error) {
      console.error(`Error fetching blog detail ${category}/${id}:`, error);
      throw error;
    }
  },

  /**
   * 获取博客的 HTML 内容
   * @param {string} category - 分类
   * @param {string} id - 博客ID
   * @param {string} lang - 语言（cn 或 en）
   * @returns {Promise<string>} HTML 内容
   */
  async getBlogContent(category, id, lang = 'cn') {
    const localizedLang = lang === 'en' ? 'en' : 'cn';
    const fileName = localizedLang === 'en' ? 'text_EN.html' : 'text_CN.html';
    const primary = `content/${category}/${id}/${fileName}`;

    try {
      let response = await fetch(primary);
      // 英文缺失时回退中文，保证详情页可用
      if (!response.ok && localizedLang === 'en') {
        response = await fetch(`content/${category}/${id}/text_CN.html`);
      }
      if (!response.ok) {
        throw new Error('Content not found');
      }
      return await response.text();
    } catch (error) {
      console.error(`Error fetching blog content ${category}/${id}/${localizedLang}:`, error);
      throw error;
    }
  },

  /**
   * 获取博客资源文件的 URL
   * @param {string} category - 分类
   * @param {string} id - 博客ID
   * @param {string} filename - 文件名
   * @returns {string} 资源URL
   */
  getResourceUrl(category, id, filename) {
    return `content/${category}/${id}/assets/${filename}`;
  }
};

// 导出 API 对象以供全局使用
if (typeof module !== 'undefined' && module.exports) {
  module.exports = BlogAPI;
}
