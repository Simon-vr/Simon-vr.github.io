/**
 * API 模块（单目录静态版）
 *
 * docs/ 既是源也是产物，没有后端、也没有构建期生成的列表 JSON。
 * 前端直接按需读取 content/ 目录下的文件：
 *   content/<category>/index.json        该分类的文章 id 列表
 *   content/<category>/<id>/metadata.json 单篇文章元数据
 *   content/<category>/<id>/text_CN.html  正文
 *   content/<category>/<id>/assets/       资源
 *   content/map/data/asset/mapdata.json   地图数据
 *
 * 所有路径均为相对路径，根目录或子路径部署都能正常工作。
 * index.json 由 scripts/build.js 生成（也可以手动维护）。
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
   * 读取某个分类下所有文章的元数据（按日期倒序）
   * @param {string} category - 分类（study, log, share 等）
   * @returns {Promise<Array>} 元数据数组，每项含 id
   */
  async getCategoryPosts(category) {
    const index = await this.fetchJson(`content/${category}/index.json`);
    const ids = Array.isArray(index) ? index : (index.posts || []);

    const results = await Promise.all(
      ids.map(async (id) => {
        try {
          const metadata = await this.fetchJson(`content/${category}/${id}/metadata.json`);
          return { id, ...metadata };
        } catch (error) {
          console.warn(`读取元数据失败: content/${category}/${id}/metadata.json`, error);
          return null;
        }
      })
    );

    return results
      .filter(Boolean)
      .sort((a, b) => new Date(b.date || 0) - new Date(a.date || 0));
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
      const posts = await this.getCategoryPosts(category);
      return posts.map((item) => this.localize(item, lang));
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
      const metadata = await this.fetchJson(`content/${category}/${id}/metadata.json`);
      return this.localize({ id, ...metadata }, lang);
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
