/**
 * 站点配置
 *
 * giscus 评论系统已启用，仓库为 Simon-vr/Simon-vr。
 * 评论按文章维度（category/id）建立讨论，避免所有文章共用一条讨论。
 * 如需更换仓库：在 https://github.com/apps/giscus 安装 App，
 * 到 https://giscus.app/ 生成新的 repoId / categoryId 填入即可。
 */
window.SiteConfig = {
  brandName: 'simon-vr',
  siteUrl: 'https://simon-vr.github.io/',
  giscus: {
    enabled: true,
    repo: 'Simon-vr/Simon-vr',
    repoId: 'R_kgDOUqpqOw',
    category: 'Announcements',
    categoryId: 'DIC_kwDOUqpqO84DGWeL',
    mapping: 'specific',
    strict: '1',
    reactionsEnabled: '1',
    emitMetadata: '1',
    inputPosition: 'top',
    loading: 'lazy',
    lang: 'zh-CN'
  }
};
