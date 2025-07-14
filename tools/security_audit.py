"""
COT-DIR 项目安全审计和加固脚本

执行全面的安全检查，包括依赖漏洞扫描、代码安全分析、配置安全检查等。
"""

import subprocess
import sys
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SecurityAuditor:
    """安全审计器"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.security_report = {
            "dependency_vulnerabilities": [],
            "code_security_issues": [],
            "configuration_issues": [],
            "file_permission_issues": [],
            "secrets_found": [],
            "summary": {}
        }
    
    def run_full_audit(self) -> Dict[str, Any]:
        """运行完整安全审计"""
        logger.info("🔒 开始 COT-DIR 项目安全审计...")
        
        # 1. 依赖漏洞扫描
        self.scan_dependency_vulnerabilities()
        
        # 2. 代码安全分析
        self.analyze_code_security()
        
        # 3. 配置安全检查
        self.check_configuration_security()
        
        # 4. 密钥泄露检测
        self.detect_secrets()
        
        # 5. 文件权限检查
        self.check_file_permissions()
        
        # 6. 生成安全报告
        self.generate_security_summary()
        
        return self.security_report
    
    def scan_dependency_vulnerabilities(self):
        """扫描依赖漏洞"""
        logger.info("🔍 扫描依赖漏洞...")
        
        try:
            # 使用 safety 扫描已知漏洞
            result = subprocess.run(
                [sys.executable, "-m", "safety", "check", "--json"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("✅ 未发现已知依赖漏洞")
                self.security_report["dependency_vulnerabilities"] = []
            else:
                try:
                    vulnerabilities = json.loads(result.stdout)
                    self.security_report["dependency_vulnerabilities"] = vulnerabilities
                    logger.warning(f"⚠️ 发现 {len(vulnerabilities)} 个依赖漏洞")
                except json.JSONDecodeError:
                    logger.error("安全扫描输出格式错误")
                    
        except FileNotFoundError:
            logger.warning("Safety 工具未安装，跳过依赖漏洞扫描")
            
        # 检查过时的依赖版本
        self.check_outdated_dependencies()
    
    def check_outdated_dependencies(self):
        """检查过时的依赖"""
        logger.info("📦 检查过时依赖...")
        
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout:
                outdated = json.loads(result.stdout)
                if outdated:
                    logger.warning(f"⚠️ 发现 {len(outdated)} 个过时依赖")
                    self.security_report["outdated_dependencies"] = outdated
                else:
                    logger.info("✅ 所有依赖均为最新版本")
            
        except Exception as e:
            logger.error(f"检查过时依赖失败: {e}")
    
    def analyze_code_security(self):
        """分析代码安全问题"""
        logger.info("🔍 分析代码安全问题...")
        
        try:
            # 使用 bandit 进行安全扫描
            result = subprocess.run(
                [sys.executable, "-m", "bandit", "-r", "src/", "-f", "json"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                try:
                    bandit_report = json.loads(result.stdout)
                    results = bandit_report.get("results", [])
                    
                    if results:
                        logger.warning(f"⚠️ Bandit 发现 {len(results)} 个安全问题")
                        self.security_report["code_security_issues"] = results
                    else:
                        logger.info("✅ Bandit 未发现安全问题")
                        
                except json.JSONDecodeError:
                    logger.error("Bandit 输出格式错误")
                    
        except FileNotFoundError:
            logger.warning("Bandit 工具未安装，跳过代码安全分析")
        
        # 手动检查常见安全问题
        self.manual_security_checks()
    
    def manual_security_checks(self):
        """手动安全检查"""
        logger.info("🔍 执行手动安全检查...")
        
        security_issues = []
        
        # 检查 pickle 使用
        pickle_files = self.find_files_with_pattern(r"import\s+pickle|pickle\.")
        if pickle_files:
            security_issues.append({
                "type": "unsafe_pickle_usage",
                "severity": "HIGH",
                "message": "发现 pickle 使用，可能存在反序列化安全风险",
                "files": pickle_files
            })
        
        # 检查 eval 使用
        eval_files = self.find_files_with_pattern(r"eval\s*\(")
        if eval_files:
            security_issues.append({
                "type": "unsafe_eval_usage", 
                "severity": "HIGH",
                "message": "发现 eval() 使用，存在代码注入风险",
                "files": eval_files
            })
        
        # 检查 shell=True 使用
        shell_files = self.find_files_with_pattern(r"subprocess.*shell\s*=\s*True")
        if shell_files:
            security_issues.append({
                "type": "unsafe_shell_usage",
                "severity": "MEDIUM", 
                "message": "发现 shell=True 使用，可能存在命令注入风险",
                "files": shell_files
            })
        
        # 检查硬编码密码
        password_files = self.find_files_with_pattern(r"password\s*=\s*[\"'][^\"']+[\"']")
        if password_files:
            security_issues.append({
                "type": "hardcoded_password",
                "severity": "HIGH",
                "message": "发现疑似硬编码密码",
                "files": password_files
            })
        
        if security_issues:
            self.security_report["manual_security_issues"] = security_issues
            logger.warning(f"⚠️ 手动检查发现 {len(security_issues)} 个安全问题")
        else:
            logger.info("✅ 手动检查未发现安全问题")
    
    def find_files_with_pattern(self, pattern: str) -> List[str]:
        """查找包含特定模式的文件"""
        matching_files = []
        pattern_re = re.compile(pattern, re.IGNORECASE)
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if pattern_re.search(content):
                        matching_files.append(str(py_file.relative_to(self.project_root)))
            except Exception:
                continue
        
        return matching_files
    
    def check_configuration_security(self):
        """检查配置安全"""
        logger.info("⚙️ 检查配置安全...")
        
        config_issues = []
        
        # 检查配置文件权限
        config_files = [
            "config.yaml", "config.yml", ".env", "settings.py",
            "config/", "configs/"
        ]
        
        for config_file in config_files:
            config_path = self.project_root / config_file
            if config_path.exists():
                if config_path.is_file():
                    # 检查文件权限
                    stat = config_path.stat()
                    mode = oct(stat.st_mode)[-3:]
                    if mode != "600":  # 应该只有所有者可读写
                        config_issues.append({
                            "type": "file_permission",
                            "file": str(config_path),
                            "current_permission": mode,
                            "recommended_permission": "600",
                            "severity": "MEDIUM"
                        })
        
        # 检查环境变量使用
        env_var_files = self.find_files_with_pattern(r"os\.environ|getenv")
        if env_var_files:
            logger.info(f"✅ 发现 {len(env_var_files)} 个文件使用环境变量（推荐做法）")
        
        # 检查敏感信息泄露
        sensitive_patterns = [
            r"api[_-]?key\s*=\s*[\"'][^\"']+[\"']",
            r"secret[_-]?key\s*=\s*[\"'][^\"']+[\"']",
            r"token\s*=\s*[\"'][^\"']+[\"']",
            r"password\s*=\s*[\"'][^\"']+[\"']"
        ]
        
        for pattern in sensitive_patterns:
            files = self.find_files_with_pattern(pattern)
            if files:
                config_issues.append({
                    "type": "sensitive_info_exposure",
                    "pattern": pattern,
                    "files": files,
                    "severity": "HIGH",
                    "message": "发现疑似敏感信息硬编码"
                })
        
        if config_issues:
            self.security_report["configuration_issues"] = config_issues
            logger.warning(f"⚠️ 发现 {len(config_issues)} 个配置安全问题")
        else:
            logger.info("✅ 配置安全检查通过")
    
    def detect_secrets(self):
        """检测密钥泄露"""
        logger.info("🔐 检测密钥泄露...")
        
        secrets_found = []
        
        # 常见密钥模式
        secret_patterns = [
            (r"sk-[a-zA-Z0-9]{48}", "OpenAI API Key"),
            (r"xoxb-[0-9]{11}-[0-9]{11}-[a-zA-Z0-9]{24}", "Slack Bot Token"),
            (r"ghp_[a-zA-Z0-9]{36}", "GitHub Personal Access Token"),
            (r"AKIA[0-9A-Z]{16}", "AWS Access Key"),
            (r"[0-9a-f]{32}", "Potential MD5 Hash/Token"),
            (r"[0-9a-f]{40}", "Potential SHA1 Hash/Token"),
            (r"-----BEGIN [A-Z ]+-----", "PEM Certificate/Key")
        ]
        
        for py_file in self.project_root.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    for pattern, secret_type in secret_patterns:
                        matches = re.finditer(pattern, content)
                        for match in matches:
                            secrets_found.append({
                                "type": secret_type,
                                "file": str(py_file.relative_to(self.project_root)),
                                "line": content[:match.start()].count('\n') + 1,
                                "pattern": pattern,
                                "severity": "HIGH"
                            })
            except Exception:
                continue
        
        # 检查配置文件中的密钥
        config_extensions = [".yaml", ".yml", ".json", ".ini", ".env"]
        for config_file in self.project_root.rglob("*"):
            if config_file.suffix in config_extensions:
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        for pattern, secret_type in secret_patterns:
                            if re.search(pattern, content):
                                secrets_found.append({
                                    "type": secret_type,
                                    "file": str(config_file.relative_to(self.project_root)),
                                    "severity": "CRITICAL"
                                })
                except Exception:
                    continue
        
        if secrets_found:
            self.security_report["secrets_found"] = secrets_found
            logger.error(f"🚨 发现 {len(secrets_found)} 个疑似密钥泄露")
        else:
            logger.info("✅ 未发现密钥泄露")
    
    def check_file_permissions(self):
        """检查文件权限"""
        logger.info("📂 检查文件权限...")
        
        permission_issues = []
        
        # 检查Python文件权限
        for py_file in self.project_root.rglob("*.py"):
            try:
                stat = py_file.stat()
                mode = oct(stat.st_mode)[-3:]
                
                # Python文件不应该有执行权限（除非是脚本）
                if int(mode[2]) % 2 == 1:  # 检查其他用户执行权限
                    permission_issues.append({
                        "type": "excessive_permission",
                        "file": str(py_file.relative_to(self.project_root)),
                        "current_permission": mode,
                        "issue": "其他用户有执行权限",
                        "severity": "LOW"
                    })
            except Exception:
                continue
        
        if permission_issues:
            self.security_report["file_permission_issues"] = permission_issues
            logger.warning(f"⚠️ 发现 {len(permission_issues)} 个文件权限问题")
        else:
            logger.info("✅ 文件权限检查通过")
    
    def generate_security_summary(self):
        """生成安全总结"""
        logger.info("📊 生成安全审计总结...")
        
        summary = {
            "total_issues": 0,
            "critical_issues": 0,
            "high_issues": 0,
            "medium_issues": 0,
            "low_issues": 0,
            "categories": {}
        }
        
        # 统计各类问题
        for category, issues in self.security_report.items():
            if category == "summary":
                continue
                
            if isinstance(issues, list):
                summary["categories"][category] = len(issues)
                summary["total_issues"] += len(issues)
                
                # 按严重程度统计
                for issue in issues:
                    severity = issue.get("severity", "LOW").upper()
                    if severity == "CRITICAL":
                        summary["critical_issues"] += 1
                    elif severity == "HIGH":
                        summary["high_issues"] += 1
                    elif severity == "MEDIUM":
                        summary["medium_issues"] += 1
                    else:
                        summary["low_issues"] += 1
        
        # 生成安全评级
        if summary["critical_issues"] > 0:
            summary["security_rating"] = "CRITICAL"
        elif summary["high_issues"] > 5:
            summary["security_rating"] = "HIGH_RISK"
        elif summary["high_issues"] > 0 or summary["medium_issues"] > 10:
            summary["security_rating"] = "MEDIUM_RISK"
        elif summary["medium_issues"] > 0 or summary["low_issues"] > 5:
            summary["security_rating"] = "LOW_RISK"
        else:
            summary["security_rating"] = "SECURE"
        
        self.security_report["summary"] = summary
        
        # 打印总结
        logger.info(f"🏁 安全审计完成")
        logger.info(f"   总问题数: {summary['total_issues']}")
        logger.info(f"   严重问题: {summary['critical_issues']}")
        logger.info(f"   高危问题: {summary['high_issues']}")  
        logger.info(f"   中危问题: {summary['medium_issues']}")
        logger.info(f"   低危问题: {summary['low_issues']}")
        logger.info(f"   安全评级: {summary['security_rating']}")
    
    def save_report(self, output_file: str = "security_audit_report.json"):
        """保存安全报告"""
        report_path = self.project_root / output_file
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.security_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 安全报告已保存到: {report_path}")


def install_security_tools():
    """安装必要的安全工具"""
    logger.info("🛠️ 检查并安装安全工具...")
    
    tools = ["safety", "bandit"]
    
    for tool in tools:
        try:
            subprocess.run([sys.executable, "-m", tool, "--version"], 
                          check=True, capture_output=True)
            logger.info(f"✅ {tool} 已安装")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.info(f"📦 安装 {tool}...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", tool], 
                              check=True)
                logger.info(f"✅ {tool} 安装完成")
            except subprocess.CalledProcessError:
                logger.error(f"❌ {tool} 安装失败")


def generate_security_recommendations(report: Dict[str, Any]) -> List[str]:
    """生成安全建议"""
    recommendations = []
    
    summary = report.get("summary", {})
    
    # 基于发现的问题生成建议
    if report.get("dependency_vulnerabilities"):
        recommendations.append("📦 立即更新存在漏洞的依赖包")
    
    if report.get("secrets_found"):
        recommendations.append("🔐 立即移除代码中的硬编码密钥，使用环境变量或密钥管理服务")
    
    if report.get("code_security_issues"):
        recommendations.append("🔍 修复代码中的安全问题，遵循安全编码最佳实践")
    
    if report.get("configuration_issues"):
        recommendations.append("⚙️ 加强配置文件安全，设置适当的文件权限")
    
    # 通用安全建议
    recommendations.extend([
        "🔒 启用依赖自动安全扫描",
        "📋 建立定期安全审计流程",
        "🛡️ 实施代码审查安全检查",
        "🔑 使用密钥管理服务存储敏感信息",
        "📊 监控和记录安全事件",
        "🎯 进行定期渗透测试"
    ])
    
    return recommendations


def main():
    """主函数"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print("🔒 COT-DIR 项目安全审计工具")
    print("=" * 60)
    
    # 安装安全工具
    install_security_tools()
    
    # 创建安全审计器
    auditor = SecurityAuditor(project_root)
    
    # 运行安全审计
    report = auditor.run_full_audit()
    
    # 保存报告
    auditor.save_report()
    
    # 生成建议
    recommendations = generate_security_recommendations(report)
    
    print("\n🎯 安全改进建议:")
    print("-" * 40)
    for i, rec in enumerate(recommendations[:10], 1):  # 显示前10个建议
        print(f"{i:2d}. {rec}")
    
    print(f"\n📊 完整报告已保存到: security_audit_report.json")
    
    # 根据安全评级给出总结
    rating = report["summary"]["security_rating"]
    if rating == "CRITICAL":
        print("🚨 项目存在严重安全风险，需要立即处理！")
    elif rating == "HIGH_RISK":
        print("⚠️ 项目存在高安全风险，建议尽快处理")
    elif rating == "MEDIUM_RISK":
        print("⚡ 项目存在中等安全风险，建议及时处理")
    elif rating == "LOW_RISK":
        print("✅ 项目安全状况良好，有少量问题需要关注")
    else:
        print("🎉 项目安全状况优秀！")


if __name__ == "__main__":
    main()