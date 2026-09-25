/**
 * 状态管理模块 - 统一管理应用全局状态
 * 用于在不同模块之间共享状态，避免过度使用全局变量
 */

const BlogState = {
  // 博客详情页面状态
  currentCategory: '',  // 当前浏览的博客分类
  currentId: '',        // 当前浏览的博客ID
  currentLang: 'cn',    // 当前使用的语言（cn 或 en）
  blogData: null,       // 当前博客的元数据
  
  /**
   * 更新博客状态
   * @param {Object} updates - 要更新的状态对象
   */
  update(updates) {
    Object.assign(this, updates);
  },
  
  /**
   * 重置所有状态
   */
  reset() {
    this.currentCategory = '';
    this.currentId = '';
    this.currentLang = 'cn';
    this.blogData = null;
  }
};
