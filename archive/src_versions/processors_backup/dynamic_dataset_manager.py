"""
🚀 Dynamic Dataset Manager - 零代码添加新题目
动态从数据集加载，支持自动发现和热加载
"""

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class DatasetMetadata:
    """数据集元数据"""
    name: str
    path: str
    size: int
    format: str  # json, jsonl, yaml, csv
    last_modified: float
    encoding: str = 'utf-8'
    complexity_distribution: Optional[Dict[str, float]] = None
    language: str = 'unknown'
    domain: str = 'unknown'
    checksum: str = ''
    auto_detected: bool = False


@dataclass
class ProblemBatch:
    """问题批次"""
    problems: List[Dict[str, Any]]
    source_dataset: str
    batch_id: str
    complexity_level: str = 'unknown'
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class DynamicDatasetManager:
    """🚀 零代码动态数据集管理器"""
    
    def __init__(self, 
                 data_dirs: List[str] = ["Data", "datasets", "data"],
                 watch_mode: bool = True,
                 auto_reload: bool = True,
                 cache_enabled: bool = True):
        """
        初始化动态数据集管理器
        
        Args:
            data_dirs: 数据目录列表
            watch_mode: 是否启用文件监控模式
            auto_reload: 是否自动重新加载变更的数据集
            cache_enabled: 是否启用缓存
        """
        self.data_dirs = [Path(d) for d in data_dirs if Path(d).exists()]
        self.watch_mode = watch_mode
        self.auto_reload = auto_reload
        self.cache_enabled = cache_enabled
        
        # 数据集注册表
        self.datasets: Dict[str, DatasetMetadata] = {}
        self.dataset_cache: Dict[str, List[Dict]] = {}
        self.file_checksums: Dict[str, str] = {}
        
        # 监控状态
        self._watching = False
        self._watch_thread = None
        self._callbacks: List[Callable] = []
        
        # 统计信息
        self.stats = {
            'datasets_discovered': 0,
            'problems_loaded': 0,
            'auto_reloads': 0,
            'last_scan': None
        }
        
        # 初始化扫描
        self.discover_datasets()
        
        # 启动监控
        if self.watch_mode:
            self.start_watching()
    
    def discover_datasets(self) -> int:
        """🔍 自动发现数据集"""
        print(f"🔍 扫描数据集目录: {[str(d) for d in self.data_dirs]}")
        
        discovered_count = 0
        supported_formats = {'.json', '.jsonl', '.yaml', '.yml'}
        
        for data_dir in self.data_dirs:
            for root, dirs, files in os.walk(data_dir):
                for file in files:
                    file_path = Path(root) / file
                    
                    # 跳过隐藏文件和非支持格式
                    if file.startswith('.') or file_path.suffix.lower() not in supported_formats:
                        continue
                    
                    # 跳过已知文件
                    if str(file_path) in self.datasets:
                        continue
                    
                    try:
                        metadata = self._analyze_dataset_file(file_path)
                        if metadata and metadata.size > 0:
                            self.datasets[metadata.name] = metadata
                            discovered_count += 1
                            print(f"  ✅ 发现数据集: {metadata.name} ({metadata.size} 问题)")
                    
                    except Exception as e:
                        print(f"  ⚠️ 跳过文件 {file_path}: {e}")
        
        self.stats['datasets_discovered'] = len(self.datasets)
        self.stats['last_scan'] = datetime.now()
        
        print(f"🎉 发现 {discovered_count} 个新数据集，总计 {len(self.datasets)} 个数据集")
        return discovered_count
    
    def _analyze_dataset_file(self, file_path: Path) -> Optional[DatasetMetadata]:
        """分析数据集文件"""
        try:
            # 计算文件校验和
            checksum = self._calculate_checksum(file_path)
            
            # 基本信息
            stat = file_path.stat()
            name = self._generate_dataset_name(file_path)
            format_type = file_path.suffix.lower().lstrip('.')
            
            # 读取并分析内容
            content = self._load_file_content(file_path)
            if not content or not isinstance(content, list):
                return None
            
            # 分析复杂度分布
            complexity_dist = self._analyze_complexity_distribution(content)
            
            # 检测语言和领域
            language = self._detect_language(content[:5])  # 使用前5个样本检测
            domain = self._detect_domain(content[:5])
            
            return DatasetMetadata(
                name=name,
                path=str(file_path),
                size=len(content),
                format=format_type,
                last_modified=stat.st_mtime,
                complexity_distribution=complexity_dist,
                language=language,
                domain=domain,
                checksum=checksum,
                auto_detected=True
            )
            
        except Exception as e:
            print(f"分析文件 {file_path} 失败: {e}")
            return None
    
    def _load_file_content(self, file_path: Path) -> List[Dict]:
        """加载文件内容"""
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.suffix.lower() == '.jsonl':
                return [json.loads(line.strip()) for line in f if line.strip()]
            elif file_path.suffix.lower() in ['.yaml', '.yml']:
                if YAML_AVAILABLE:
                    return yaml.safe_load(f) or []
                else:
                    print(f"⚠️ YAML文件需要安装PyYAML: {file_path}")
                    return []
            else:
                data = json.load(f)
                if isinstance(data, dict):
                    # 如果是字典，尝试提取问题列表
                    if 'problems' in data:
                        return data['problems']
                    elif 'data' in data:
                        return data['data']
                    else:
                        return [data]
                return data if isinstance(data, list) else []
    
    def _generate_dataset_name(self, file_path: Path) -> str:
        """生成数据集名称"""
        # 使用父目录名 + 文件名
        parent_name = file_path.parent.name
        file_name = file_path.stem
        
        if parent_name != "Data" and parent_name != "datasets":
            return f"{parent_name}_{file_name}"
        return file_name
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """计算文件校验和"""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def _analyze_complexity_distribution(self, content: List[Dict]) -> Dict[str, float]:
        """分析复杂度分布"""
        if not content:
            return {}
        
        # 简单的复杂度启发式分析
        l0_count = 0  # 简单算术
        l1_count = 0  # 单步推理
        l2_count = 0  # 多步推理
        l3_count = 0  # 复杂推理
        
        for item in content[:100]:  # 分析前100个样本
            text = str(item.get('problem', '') or item.get('question', '') or item.get('text', ''))
            
            # 基于关键词的简单分类
            if any(word in text.lower() for word in ['what is', 'calculate', '+', '-', '*', '/']):
                l0_count += 1
            elif any(word in text.lower() for word in ['find', 'solve', 'determine']):
                l1_count += 1
            elif any(word in text.lower() for word in ['step', 'first', 'then', 'next']):
                l2_count += 1
            else:
                l3_count += 1
        
        total = l0_count + l1_count + l2_count + l3_count
        if total == 0:
            return {}
        
        return {
            'L0': round(l0_count / total * 100, 1),
            'L1': round(l1_count / total * 100, 1),
            'L2': round(l2_count / total * 100, 1),
            'L3': round(l3_count / total * 100, 1)
        }
    
    def _detect_language(self, samples: List[Dict]) -> str:
        """检测语言"""
        if not samples:
            return 'unknown'
        
        chinese_chars = 0
        total_chars = 0
        
        for item in samples:
            text = str(item.get('problem', '') or item.get('question', '') or item.get('text', ''))
            for char in text:
                total_chars += 1
                if '\u4e00' <= char <= '\u9fff':  # 中文字符范围
                    chinese_chars += 1
        
        if total_chars == 0:
            return 'unknown'
        
        chinese_ratio = chinese_chars / total_chars
        if chinese_ratio > 0.3:
            return 'Chinese'
        elif chinese_ratio < 0.05:
            return 'English'
        else:
            return 'Mixed'
    
    def _detect_domain(self, samples: List[Dict]) -> str:
        """检测领域"""
        if not samples:
            return 'unknown'
        
        domain_keywords = {
            'Elementary': ['grade', 'elementary', 'primary', 'basic'],
            'Competition': ['competition', 'contest', 'olympiad', 'AMC'],
            'Grade School': ['school', 'student', 'class'],
            'Multi-domain': ['science', 'physics', 'chemistry', 'biology']
        }
        
        text = ' '.join(str(item.get('problem', '') or item.get('question', '') or item.get('text', '')) 
                       for item in samples).lower()
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in text for keyword in keywords):
                return domain
        
        return 'General'
    
    def load_dataset(self, dataset_name: str, 
                    max_samples: Optional[int] = None,
                    complexity_filter: Optional[str] = None) -> List[Dict]:
        """🔄 加载数据集（支持缓存和过滤）"""
        if dataset_name not in self.datasets:
            raise ValueError(f"数据集 '{dataset_name}' 不存在。可用数据集: {list(self.datasets.keys())}")
        
        metadata = self.datasets[dataset_name]
        
        # 检查缓存
        if self.cache_enabled and dataset_name in self.dataset_cache:
            # 检查文件是否有变更
            current_checksum = self._calculate_checksum(Path(metadata.path))
            if current_checksum == metadata.checksum:
                data = self.dataset_cache[dataset_name]
            else:
                # 文件有变更，重新加载
                data = self._load_file_content(Path(metadata.path))
                self.dataset_cache[dataset_name] = data
                metadata.checksum = current_checksum
                print(f"🔄 检测到文件变更，重新加载 {dataset_name}")
        else:
            # 加载数据
            data = self._load_file_content(Path(metadata.path))
            if self.cache_enabled:
                self.dataset_cache[dataset_name] = data
        
        # 应用过滤器
        if complexity_filter:
            data = self._filter_by_complexity(data, complexity_filter)
        
        # 限制样本数
        if max_samples:
            data = data[:max_samples]
        
        self.stats['problems_loaded'] += len(data)
        return data
    
    def _filter_by_complexity(self, data: List[Dict], complexity_level: str) -> List[Dict]:
        """根据复杂度过滤数据"""
        # 这里可以实现更复杂的过滤逻辑
        # 目前返回原数据
        return data
    
    def get_dynamic_batch(self, 
                         batch_size: int = 10,
                         datasets: Optional[List[str]] = None,
                         complexity_mix: Optional[Dict[str, float]] = None) -> ProblemBatch:
        """📦 获取动态问题批次"""
        if not datasets:
            datasets = list(self.datasets.keys())
        
        problems = []
        selected_datasets = []
        
        for dataset_name in datasets:
            if len(problems) >= batch_size:
                break
            
            try:
                dataset_problems = self.load_dataset(dataset_name, max_samples=batch_size//len(datasets) + 1)
                problems.extend(dataset_problems)
                selected_datasets.append(dataset_name)
            except Exception as e:
                print(f"⚠️ 加载数据集 {dataset_name} 失败: {e}")
        
        # 限制批次大小
        problems = problems[:batch_size]
        
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return ProblemBatch(
            problems=problems,
            source_dataset=','.join(selected_datasets),
            batch_id=batch_id
        )
    
    def start_watching(self):
        """🔍 启动文件监控"""
        if self._watching:
            return
        
        self._watching = True
        self._watch_thread = threading.Thread(target=self._watch_files, daemon=True)
        self._watch_thread.start()
        print("🔍 启动文件监控模式")
    
    def stop_watching(self):
        """⏹️ 停止文件监控"""
        self._watching = False
        if self._watch_thread:
            self._watch_thread.join()
        print("⏹️ 停止文件监控")
    
    def _watch_files(self):
        """文件监控循环"""
        while self._watching:
            try:
                # 检查文件变更
                changes_detected = False
                
                for dataset_name, metadata in self.datasets.items():
                    file_path = Path(metadata.path)
                    if file_path.exists():
                        current_checksum = self._calculate_checksum(file_path)
                        if current_checksum != metadata.checksum:
                            print(f"🔄 检测到数据集变更: {dataset_name}")
                            if self.auto_reload:
                                self._reload_dataset(dataset_name)
                            changes_detected = True
                
                # 扫描新数据集
                new_count = self.discover_datasets()
                if new_count > 0:
                    changes_detected = True
                
                # 通知回调
                if changes_detected and self._callbacks:
                    for callback in self._callbacks:
                        try:
                            callback(self)
                        except Exception as e:
                            print(f"回调执行失败: {e}")
                
                time.sleep(5)  # 每5秒检查一次
                
            except Exception as e:
                print(f"文件监控错误: {e}")
                time.sleep(10)
    
    def _reload_dataset(self, dataset_name: str):
        """重新加载数据集"""
        try:
            metadata = self.datasets[dataset_name]
            
            # 重新分析文件
            new_metadata = self._analyze_dataset_file(Path(metadata.path))
            if new_metadata:
                self.datasets[dataset_name] = new_metadata
                
                # 清除缓存
                if dataset_name in self.dataset_cache:
                    del self.dataset_cache[dataset_name]
                
                self.stats['auto_reloads'] += 1
                print(f"✅ 重新加载数据集: {dataset_name}")
            
        except Exception as e:
            print(f"重新加载数据集失败 {dataset_name}: {e}")
    
    def add_change_callback(self, callback: Callable):
        """添加变更回调"""
        self._callbacks.append(callback)
    
    def get_stats(self) -> Dict[str, Any]:
        """📊 获取统计信息"""
        return {
            **self.stats,
            'total_datasets': len(self.datasets),
            'cached_datasets': len(self.dataset_cache),
            'watching_enabled': self._watching,
            'available_datasets': list(self.datasets.keys())
        }
    
    def list_datasets(self, detailed: bool = False) -> List[Dict[str, Any]]:
        """📋 列出所有数据集"""
        if not detailed:
            return list(self.datasets.keys())
        
        datasets_info = []
        for name, metadata in self.datasets.items():
            info = asdict(metadata)
            info['cached'] = name in self.dataset_cache
            datasets_info.append(info)
        
        return datasets_info
    
    def export_config(self, config_path: str = "dataset_config.yaml"):
        """💾 导出配置"""
        config = {
            'data_dirs': [str(d) for d in self.data_dirs],
            'watch_mode': self.watch_mode,
            'auto_reload': self.auto_reload,
            'cache_enabled': self.cache_enabled,
            'datasets': {name: asdict(meta) for name, meta in self.datasets.items()},
            'stats': self.stats
        }
        
        if YAML_AVAILABLE and config_path.endswith(('.yaml', '.yml')):
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        else:
            # 如果没有yaml或路径不是yaml格式，使用json
            if config_path.endswith(('.yaml', '.yml')):
                config_path = config_path.replace('.yaml', '.json').replace('.yml', '.json')
            elif not config_path.endswith('.json'):
                config_path += '.json'
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"💾 配置已导出到: {config_path}")
    
    def __del__(self):
        """清理资源"""
        if self._watching:
            self.stop_watching()


# 使用示例和测试
def demo_dynamic_dataset_manager():
    """演示动态数据集管理器"""
    print("🚀 Dynamic Dataset Manager Demo")
    print("=" * 50)
    
    # 创建管理器
    manager = DynamicDatasetManager(
        data_dirs=["Data", "datasets"],
        watch_mode=True,
        auto_reload=True
    )
    
    # 显示统计信息
    stats = manager.get_stats()
    print(f"📊 统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 列出数据集
    print(f"\n📋 可用数据集 ({len(manager.datasets)}):")
    for name in manager.list_datasets():
        metadata = manager.datasets[name]
        print(f"  ✅ {name}: {metadata.size} 问题 ({metadata.language}, {metadata.domain})")
    
    # 获取动态批次
    print(f"\n📦 获取动态批次:")
    batch = manager.get_dynamic_batch(batch_size=5)
    print(f"  批次ID: {batch.batch_id}")
    print(f"  源数据集: {batch.source_dataset}")
    print(f"  问题数量: {len(batch.problems)}")
    
    # 显示前2个问题
    for i, problem in enumerate(batch.problems[:2]):
        print(f"  问题 {i+1}: {str(problem)[:100]}...")
    
    return manager


if __name__ == "__main__":
    demo_dynamic_dataset_manager() 