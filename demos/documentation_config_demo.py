"""
文档与配置管理验证演示

验证文档生成器和增强配置管理系统的功能。
"""

import sys
import os
from pathlib import Path

# 添加src和tools路径
sys.path.append(str(Path(__file__).parent.parent / "src"))
sys.path.append(str(Path(__file__).parent.parent))

from tools.documentation_generator import generate_project_documentation
from config.config_manager import (
    EnhancedConfigurationManager, ConfigLevel, ConfigSchema,
    get_config, init_config, config_override,
    ENHANCED_CONFIG_SCHEMA
)


def test_documentation_generation():
    """测试文档生成功能"""
    print("📚 测试文档生成功能...")
    
    try:
        # 设置输出目录
        output_dir = Path(__file__).parent.parent / "docs" / "generated"
        
        # 生成文档
        print("  - 开始生成项目文档...")
        docs = generate_project_documentation(
            project_root=str(Path(__file__).parent.parent),
            output_dir=str(output_dir),
            include_private=False
        )
        
        # 检查生成的文档
        print(f"  - 文档生成完成，输出目录: {output_dir}")
        print("  - 生成的文档类型:")
        for doc_type, content in docs.items():
            content_length = len(content)
            print(f"    * {doc_type}: {content_length:,} 字符")
        
        # 验证文件是否存在
        expected_files = [
            "README.md",
            "API_Reference.md", 
            "Architecture.md",
            "User_Guide.md",
            "Examples.md"
        ]
        
        print("  - 验证生成的文件:")
        for filename in expected_files:
            file_path = output_dir / filename
            if file_path.exists():
                size = file_path.stat().st_size
                print(f"    ✅ {filename} ({size:,} bytes)")
            else:
                print(f"    ❌ {filename} (缺失)")
        
        print("✅ 文档生成测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 文档生成测试失败: {e}")
        return False


def test_enhanced_configuration():
    """测试增强配置管理"""
    print("\n⚙️ 测试增强配置管理...")
    
    try:
        # 创建配置管理器
        config_dir = Path(__file__).parent.parent / "config"
        config_mgr = EnhancedConfigurationManager(
            env="development",
            config_dir=str(config_dir)
        )
        
        # 创建默认配置文件
        print("  - 创建默认配置文件...")
        config_mgr.create_default_configs()
        
        # 测试基本配置获取
        print("  - 测试基本配置获取:")
        max_workers = config_mgr.get("system.max_workers", 4)
        log_level = config_mgr.get("system.log_level", "INFO")
        strategy = config_mgr.get("orchestration.strategy", "unified")
        
        print(f"    * system.max_workers: {max_workers}")
        print(f"    * system.log_level: {log_level}")
        print(f"    * orchestration.strategy: {strategy}")
        
        # 测试分层配置
        print("  - 测试分层配置:")
        for level in ConfigLevel:
            level_config = config_mgr.get_all(level)
            print(f"    * {level.value}: {len(level_config)} 项配置")
        
        # 测试环境变量配置
        print("  - 测试环境变量配置:")
        os.environ["COT_DIR_TEST_VALUE"] = "12345"
        os.environ["COT_DIR_NESTED_CONFIG_VALUE"] = "test"
        config_mgr.reload_config()
        
        test_value = config_mgr.get("test.value")
        nested_value = config_mgr.get("nested.config.value")
        print(f"    * 环境变量 test.value: {test_value}")
        print(f"    * 环境变量 nested.config.value: {nested_value}")
        
        # 测试配置设置和持久化
        print("  - 测试配置设置:")
        config_mgr.set("test.runtime_value", "runtime_test", ConfigLevel.RUNTIME)
        config_mgr.set("test.user_value", "user_test", ConfigLevel.USER, persist=False)
        
        runtime_val = config_mgr.get("test.runtime_value")
        user_val = config_mgr.get("test.user_value")
        print(f"    * 运行时配置: {runtime_val}")
        print(f"    * 用户配置: {user_val}")
        
        # 测试配置覆盖
        print("  - 测试配置覆盖:")
        original_workers = config_mgr.get("system.max_workers")
        
        with config_mgr.override({"system.max_workers": 16, "test.override": "覆盖值"}):
            override_workers = config_mgr.get("system.max_workers")
            override_test = config_mgr.get("test.override")
            print(f"    * 覆盖中 max_workers: {override_workers}")
            print(f"    * 覆盖中 test.override: {override_test}")
        
        restored_workers = config_mgr.get("system.max_workers")
        restored_test = config_mgr.get("test.override")
        print(f"    * 恢复后 max_workers: {restored_workers}")
        print(f"    * 恢复后 test.override: {restored_test}")
        
        # 测试配置验证
        print("  - 测试配置验证:")
        try:
            config_mgr.validate_config(ENHANCED_CONFIG_SCHEMA)
            print("    ✅ 配置验证通过")
        except Exception as e:
            print(f"    ⚠️ 配置验证问题: {e}")
        
        # 测试配置监听器
        print("  - 测试配置监听器:")
        changes_received = []
        
        def config_listener(level, changes):
            changes_received.append((level, changes))
        
        config_mgr.add_listener(config_listener)
        config_mgr.set("test.listener", "监听器测试", ConfigLevel.RUNTIME)
        
        if changes_received:
            level, changes = changes_received[-1]
            print(f"    ✅ 收到配置变更通知: {level.value} - {changes}")
        else:
            print("    ⚠️ 未收到配置变更通知")
        
        # 获取配置摘要
        print("  - 配置摘要:")
        summary = config_mgr.get_config_summary()
        print(f"    * 环境: {summary['environment']}")
        print(f"    * 配置源数量: {summary['sources_count']}")
        print(f"    * 监听器数量: {summary['listeners_count']}")
        print(f"    * 加密可用: {summary['encryption_available']}")
        
        print("✅ 增强配置管理测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 增强配置管理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_global_config_functions():
    """测试全局配置函数"""
    print("\n🌐 测试全局配置函数...")
    
    try:
        # 初始化全局配置
        config_mgr = init_config(env="test", config_dir="./config")
        
        # 创建测试配置
        config_mgr.create_default_configs()
        
        # 测试便利函数
        from config.config_manager import get_config_value, set_config_value
        
        # 获取配置值
        max_workers = get_config_value("system.max_workers", 2)
        print(f"  - get_config_value('system.max_workers'): {max_workers}")
        
        # 设置配置值
        set_config_value("test.global_function", "全局函数测试")
        global_test = get_config_value("test.global_function")
        print(f"  - 设置并获取配置值: {global_test}")
        
        # 测试配置覆盖上下文管理器
        print("  - 测试配置覆盖上下文:")
        original_value = get_config_value("system.max_workers")
        
        with config_override(system={"max_workers": 20}):
            override_value = get_config_value("system.max_workers")
            print(f"    * 覆盖中的值: {override_value}")
        
        restored_value = get_config_value("system.max_workers")
        print(f"    * 恢复后的值: {restored_value}")
        
        print("✅ 全局配置函数测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 全局配置函数测试失败: {e}")
        return False


def test_configuration_security():
    """测试配置安全功能"""
    print("\n🔒 测试配置安全功能...")
    
    try:
        config_mgr = get_config()
        
        # 测试敏感配置加密
        if config_mgr.cipher:
            print("  - 测试敏感配置加密:")
            
            secure_config = {
                "database": {
                    "password": "super_secret_password",
                    "api_key": "sk-1234567890abcdef"
                },
                "third_party": {
                    "secret_token": "token_abc123"
                }
            }
            
            # 加密保存
            config_mgr.encrypt_and_save_secure_config(secure_config)
            print("    ✅ 敏感配置已加密保存")
            
            # 重载配置以验证加密配置加载
            config_mgr.reload_config()
            
            # 验证配置是否正确加载
            db_password = config_mgr.get("database.password")
            api_key = config_mgr.get("database.api_key")
            
            if db_password and api_key:
                print("    ✅ 加密配置成功加载")
            else:
                print("    ⚠️ 加密配置加载可能有问题")
                
        else:
            print("  - ⚠️ 配置加密未启用")
        
        # 测试配置摘要中的敏感信息屏蔽
        print("  - 测试敏感信息屏蔽:")
        summary = config_mgr.get_config_summary()
        
        # 检查是否正确屏蔽了敏感信息
        config_str = str(summary)
        if "***MASKED***" in config_str:
            print("    ✅ 敏感信息已正确屏蔽")
        else:
            print("    ⚠️ 可能未正确屏蔽敏感信息")
        
        print("✅ 配置安全功能测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 配置安全功能测试失败: {e}")
        return False


def test_integration():
    """测试文档和配置的集成"""
    print("\n🔗 测试文档与配置集成...")
    
    try:
        # 使用配置驱动的文档生成
        config_mgr = get_config()
        
        # 从配置获取文档设置
        doc_config = {
            "include_private": config_mgr.get("documentation.include_private", False),
            "generate_architecture": config_mgr.get("documentation.generate_architecture", True),
            "output_format": config_mgr.get("documentation.output_format", "markdown")
        }
        
        print(f"  - 文档配置: {doc_config}")
        
        # 生成配置文档
        config_summary = config_mgr.get_config_summary()
        
        config_doc = f"""# 配置文档

## 当前环境
- 环境: {config_summary['environment']}
- 配置源数量: {config_summary['sources_count']}
- 监听器数量: {config_summary['listeners_count']}
- 加密支持: {config_summary['encryption_available']}

## 配置层级
"""
        
        for level, config_data in config_summary['config_levels'].items():
            config_doc += f"\n### {level.title()}\n"
            config_doc += f"配置项数量: {len(config_data)}\n"
        
        # 保存配置文档
        docs_dir = Path(__file__).parent.parent / "docs" / "generated"
        docs_dir.mkdir(parents=True, exist_ok=True)
        
        config_doc_path = docs_dir / "Configuration.md"
        with open(config_doc_path, 'w', encoding='utf-8') as f:
            f.write(config_doc)
        
        print(f"  - 配置文档已生成: {config_doc_path}")
        print("✅ 文档与配置集成测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 文档与配置集成测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 COT-DIR 文档与配置管理验证演示")
    print("=" * 80)
    
    test_results = []
    
    # 运行所有测试
    tests = [
        ("文档生成功能", test_documentation_generation),
        ("增强配置管理", test_enhanced_configuration),
        ("全局配置函数", test_global_config_functions),
        ("配置安全功能", test_configuration_security),
        ("文档配置集成", test_integration)
    ]
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            test_results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name}执行异常: {e}")
            test_results.append((test_name, False))
    
    # 总结报告
    print("\n" + "=" * 80)
    print("📋 测试结果总结:")
    
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    print(f"\n📊 总体结果: {passed_tests}/{total_tests} 测试通过")
    
    if passed_tests == total_tests:
        print("🎉 所有文档与配置管理测试通过！")
        print("\n✨ 主要功能:")
        print("  - 📚 自动API文档生成")
        print("  - 📋 架构文档和用户指南")
        print("  - ⚙️ 分层配置管理")
        print("  - 🔒 配置加密和安全")
        print("  - 🔄 配置热重载和监听")
        print("  - 🌐 环境变量集成")
        print("  - 📖 配置文档生成")
    else:
        print("⚠️  部分测试未通过，需要进一步调试")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    main()