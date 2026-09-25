const path = require('path');
const { existsSync, readFileSync } = require('fs');

function getConfigPath() {
  return path.join(path.dirname(__dirname), 'config.json');
}

function loadAppConfig() {
  const configPath = getConfigPath();
  if (!existsSync(configPath)) {
    throw new Error(`配置文件不存在: ${configPath}`);
  }

  let content = '';
  try {
    content = readFileSync(configPath, 'utf8');
  } catch (error) {
    throw new Error(`读取配置文件失败: ${error.message}`);
  }

  try {
    return JSON.parse(content);
  } catch (error) {
    throw new Error(`配置文件 JSON 格式错误: ${error.message}`);
  }
}

module.exports = {
  getConfigPath,
  loadAppConfig,
};
