"""
📊 Batch Processor - 批量处理和质量评估
标准化流程，智能质量控制和评估系统
"""

import asyncio
import json
import logging
import statistics
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


class ProcessingStatus(Enum):
    """处理状态"""
    PENDING = "待处理"
    PROCESSING = "处理中"
    COMPLETED = "已完成"
    FAILED = "失败"
    CANCELLED = "已取消"


class QualityLevel(Enum):
    """质量等级"""
    EXCELLENT = "优秀"     # 90-100%
    GOOD = "良好"          # 70-89%
    AVERAGE = "一般"       # 50-69%
    POOR = "较差"          # 30-49%
    UNACCEPTABLE = "不可接受"  # 0-29%


@dataclass
class BatchJob:
    """批处理任务"""
    job_id: str
    name: str
    input_data: List[Any]
    processor_config: Dict[str, Any]
    status: ProcessingStatus = ProcessingStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    results: List[Any] = None
    errors: List[str] = None
    quality_metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.results is None:
            self.results = []
        if self.errors is None:
            self.errors = []
        if self.quality_metrics is None:
            self.quality_metrics = {}


@dataclass
class QualityMetrics:
    """质量指标"""
    accuracy: float = 0.0
    completeness: float = 0.0
    consistency: float = 0.0
    processing_time: float = 0.0
    error_rate: float = 0.0
    overall_score: float = 0.0
    quality_level: QualityLevel = QualityLevel.UNACCEPTABLE
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


@dataclass
class ProcessingReport:
    """处理报告"""
    job_id: str
    total_items: int
    processed_items: int
    successful_items: int
    failed_items: int
    processing_time: float
    average_time_per_item: float
    quality_metrics: QualityMetrics
    error_summary: Dict[str, int]
    performance_stats: Dict[str, float]
    generated_at: datetime = None
    
    def __post_init__(self):
        if self.generated_at is None:
            self.generated_at = datetime.now()


class BatchProcessor:
    """📊 批量处理器"""
    
    def __init__(self, 
                 max_workers: int = 4,
                 use_multiprocessing: bool = False,
                 quality_threshold: float = 0.7,
                 enable_monitoring: bool = True):
        """
        初始化批量处理器
        
        Args:
            max_workers: 最大工作线程/进程数
            use_multiprocessing: 是否使用多进程
            quality_threshold: 质量阈值
            enable_monitoring: 是否启用监控
        """
        self.max_workers = max_workers
        self.use_multiprocessing = use_multiprocessing
        self.quality_threshold = quality_threshold
        self.enable_monitoring = enable_monitoring
        
        # 执行器
        if use_multiprocessing:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 任务管理
        self.jobs: Dict[str, BatchJob] = {}
        self.processing_queue = []
        self.completed_jobs = []
        
        # 质量评估器
        self.quality_evaluators: Dict[str, Callable] = {}
        self._setup_default_evaluators()
        
        # 监控数据
        self.performance_stats = {
            'total_jobs': 0,
            'total_items_processed': 0,
            'average_processing_time': 0.0,
            'error_rate': 0.0,
            'quality_distribution': defaultdict(int)
        }
        
        # 设置日志
        self._setup_logging()
        
        print(f"📊 批量处理器已初始化 (工作者: {max_workers}, 多进程: {use_multiprocessing})")
    
    def _setup_logging(self):
        """设置日志"""
        self.logger = logging.getLogger('BatchProcessor')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _setup_default_evaluators(self):
        """设置默认质量评估器"""
        self.quality_evaluators = {
            'math_problem_solver': self._evaluate_math_solution_quality,
            'classification': self._evaluate_classification_quality,
            'general': self._evaluate_general_quality
        }
    
    def submit_job(self, 
                   name: str,
                   input_data: List[Any],
                   processor_func: Callable,
                   processor_config: Optional[Dict] = None,
                   quality_evaluator: Optional[str] = None) -> str:
        """
        📤 提交批处理任务
        
        Args:
            name: 任务名称
            input_data: 输入数据
            processor_func: 处理函数
            processor_config: 处理器配置
            quality_evaluator: 质量评估器名称
            
        Returns:
            任务ID
        """
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.jobs)}"
        
        if processor_config is None:
            processor_config = {}
        
        processor_config['processor_func'] = processor_func
        processor_config['quality_evaluator'] = quality_evaluator or 'general'
        
        job = BatchJob(
            job_id=job_id,
            name=name,
            input_data=input_data,
            processor_config=processor_config
        )
        
        self.jobs[job_id] = job
        self.processing_queue.append(job_id)
        
        self.logger.info(f"📤 提交任务: {name} (ID: {job_id}, 数据量: {len(input_data)})")
        
        return job_id
    
    async def process_job_async(self, job_id: str) -> ProcessingReport:
        """🔄 异步处理任务"""
        if job_id not in self.jobs:
            raise ValueError(f"任务 {job_id} 不存在")
        
        job = self.jobs[job_id]
        
        try:
            # 更新状态
            job.status = ProcessingStatus.PROCESSING
            job.started_at = datetime.now()
            
            self.logger.info(f"🔄 开始处理任务: {job.name}")
            
            # 获取处理函数和配置
            processor_func = job.processor_config['processor_func']
            quality_evaluator = job.processor_config.get('quality_evaluator', 'general')
            
            # 批量处理
            results = []
            errors = []
            
            total_items = len(job.input_data)
            processed_count = 0
            
            # 使用执行器并行处理
            tasks = []
            batch_size = min(self.max_workers * 2, total_items)
            
            for i in range(0, total_items, batch_size):
                batch = job.input_data[i:i + batch_size]
                if self.use_multiprocessing:
                    # 多进程处理
                    future = self.executor.submit(self._process_batch, batch, processor_func)
                else:
                    # 多线程处理
                    future = self.executor.submit(self._process_batch, batch, processor_func)
                tasks.append((i, future))
            
            # 收集结果
            for batch_start, future in tasks:
                try:
                    batch_results, batch_errors = future.result(timeout=300)  # 5分钟超时
                    results.extend(batch_results)
                    errors.extend(batch_errors)
                    
                    processed_count += len(batch_results) + len(batch_errors)
                    job.progress = processed_count / total_items
                    
                    if self.enable_monitoring:
                        self.logger.info(f"  进度: {job.progress:.1%} ({processed_count}/{total_items})")
                
                except Exception as e:
                    error_msg = f"批次处理失败: {str(e)}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
            
            # 更新任务结果
            job.results = results
            job.errors = errors
            job.completed_at = datetime.now()
            job.status = ProcessingStatus.COMPLETED
            
            # 质量评估
            quality_metrics = self._evaluate_job_quality(job, quality_evaluator)
            job.quality_metrics = asdict(quality_metrics)
            
            # 生成报告
            report = self._generate_report(job)
            
            # 更新统计
            self._update_performance_stats(report)
            
            self.completed_jobs.append(job_id)
            if job_id in self.processing_queue:
                self.processing_queue.remove(job_id)
            
            self.logger.info(f"✅ 任务完成: {job.name} (质量: {quality_metrics.quality_level.value})")
            
            return report
            
        except Exception as e:
            job.status = ProcessingStatus.FAILED
            job.errors.append(f"任务处理失败: {str(e)}")
            job.completed_at = datetime.now()
            
            self.logger.error(f"❌ 任务失败: {job.name} - {str(e)}")
            raise
    
    def process_job(self, job_id: str) -> ProcessingReport:
        """🔄 同步处理任务"""
        return asyncio.run(self.process_job_async(job_id))
    
    def _process_batch(self, batch_data: List[Any], processor_func: Callable) -> Tuple[List[Any], List[str]]:
        """处理数据批次"""
        results = []
        errors = []
        
        for item in batch_data:
            try:
                result = processor_func(item)
                results.append(result)
            except Exception as e:
                error_msg = f"处理项目失败: {str(e)}"
                errors.append(error_msg)
        
        return results, errors
    
    def _evaluate_job_quality(self, job: BatchJob, evaluator_name: str) -> QualityMetrics:
        """评估任务质量"""
        evaluator = self.quality_evaluators.get(evaluator_name, self._evaluate_general_quality)
        
        return evaluator(job)
    
    def _evaluate_math_solution_quality(self, job: BatchJob) -> QualityMetrics:
        """评估数学解题质量"""
        total_items = len(job.input_data)
        successful_items = len(job.results)
        failed_items = len(job.errors)
        
        # 计算基本指标
        completeness = successful_items / total_items if total_items > 0 else 0
        error_rate = failed_items / total_items if total_items > 0 else 1
        
        # 分析解题准确性（假设结果包含正确性信息）
        correct_solutions = 0
        for result in job.results:
            if isinstance(result, dict) and result.get('is_correct', False):
                correct_solutions += 1
        
        accuracy = correct_solutions / successful_items if successful_items > 0 else 0
        
        # 处理时间分析
        processing_time = 0
        if job.started_at and job.completed_at:
            processing_time = (job.completed_at - job.started_at).total_seconds()
        
        # 一致性评估（解题步骤的完整性）
        consistent_solutions = 0
        for result in job.results:
            if isinstance(result, dict) and result.get('solution_steps'):
                consistent_solutions += 1
        
        consistency = consistent_solutions / successful_items if successful_items > 0 else 0
        
        # 计算总体分数
        overall_score = (accuracy * 0.4 + completeness * 0.3 + 
                        consistency * 0.2 + (1 - error_rate) * 0.1)
        
        # 确定质量等级
        quality_level = self._determine_quality_level(overall_score)
        
        # 生成建议
        recommendations = self._generate_quality_recommendations(
            accuracy, completeness, consistency, error_rate
        )
        
        return QualityMetrics(
            accuracy=accuracy,
            completeness=completeness,
            consistency=consistency,
            processing_time=processing_time,
            error_rate=error_rate,
            overall_score=overall_score,
            quality_level=quality_level,
            recommendations=recommendations
        )
    
    def _evaluate_classification_quality(self, job: BatchJob) -> QualityMetrics:
        """评估分类质量"""
        total_items = len(job.input_data)
        successful_items = len(job.results)
        
        # 基本指标
        completeness = successful_items / total_items if total_items > 0 else 0
        error_rate = len(job.errors) / total_items if total_items > 0 else 1
        
        # 分类置信度分析
        confidence_scores = []
        high_confidence_count = 0
        
        for result in job.results:
            if isinstance(result, dict) and 'confidence' in result:
                confidence = result['confidence']
                confidence_scores.append(confidence)
                if confidence > 0.8:
                    high_confidence_count += 1
        
        accuracy = high_confidence_count / successful_items if successful_items > 0 else 0
        consistency = statistics.mean(confidence_scores) if confidence_scores else 0
        
        # 处理时间
        processing_time = 0
        if job.started_at and job.completed_at:
            processing_time = (job.completed_at - job.started_at).total_seconds()
        
        overall_score = (accuracy * 0.4 + completeness * 0.3 + 
                        consistency * 0.2 + (1 - error_rate) * 0.1)
        
        quality_level = self._determine_quality_level(overall_score)
        recommendations = self._generate_quality_recommendations(
            accuracy, completeness, consistency, error_rate
        )
        
        return QualityMetrics(
            accuracy=accuracy,
            completeness=completeness,
            consistency=consistency,
            processing_time=processing_time,
            error_rate=error_rate,
            overall_score=overall_score,
            quality_level=quality_level,
            recommendations=recommendations
        )
    
    def _evaluate_general_quality(self, job: BatchJob) -> QualityMetrics:
        """评估通用质量"""
        total_items = len(job.input_data)
        successful_items = len(job.results)
        
        completeness = successful_items / total_items if total_items > 0 else 0
        error_rate = len(job.errors) / total_items if total_items > 0 else 1
        accuracy = completeness  # 对于通用情况，完成度即准确度
        consistency = 1.0 if successful_items == total_items else 0.8
        
        processing_time = 0
        if job.started_at and job.completed_at:
            processing_time = (job.completed_at - job.started_at).total_seconds()
        
        overall_score = (accuracy * 0.5 + completeness * 0.3 + (1 - error_rate) * 0.2)
        quality_level = self._determine_quality_level(overall_score)
        
        return QualityMetrics(
            accuracy=accuracy,
            completeness=completeness,
            consistency=consistency,
            processing_time=processing_time,
            error_rate=error_rate,
            overall_score=overall_score,
            quality_level=quality_level,
            recommendations=[]
        )
    
    def _determine_quality_level(self, score: float) -> QualityLevel:
        """确定质量等级"""
        if score >= 0.9:
            return QualityLevel.EXCELLENT
        elif score >= 0.7:
            return QualityLevel.GOOD
        elif score >= 0.5:
            return QualityLevel.AVERAGE
        elif score >= 0.3:
            return QualityLevel.POOR
        else:
            return QualityLevel.UNACCEPTABLE
    
    def _generate_quality_recommendations(self, accuracy: float, completeness: float, 
                                        consistency: float, error_rate: float) -> List[str]:
        """生成质量改进建议"""
        recommendations = []
        
        if accuracy < 0.7:
            recommendations.append("准确率偏低，建议优化算法或增加训练数据")
        
        if completeness < 0.9:
            recommendations.append("完成率不足，建议检查输入数据质量和处理逻辑")
        
        if consistency < 0.8:
            recommendations.append("一致性较差，建议标准化处理流程")
        
        if error_rate > 0.1:
            recommendations.append("错误率较高，建议增强异常处理机制")
        
        return recommendations
    
    def _generate_report(self, job: BatchJob) -> ProcessingReport:
        """生成处理报告"""
        total_items = len(job.input_data)
        processed_items = len(job.results) + len(job.errors)
        successful_items = len(job.results)
        failed_items = len(job.errors)
        
        processing_time = 0
        if job.started_at and job.completed_at:
            processing_time = (job.completed_at - job.started_at).total_seconds()
        
        average_time_per_item = processing_time / total_items if total_items > 0 else 0
        
        # 错误类型统计
        error_summary = defaultdict(int)
        for error in job.errors:
            error_type = error.split(':')[0] if ':' in error else 'Unknown'
            error_summary[error_type] += 1
        
        # 性能统计
        performance_stats = {
            'throughput': total_items / processing_time if processing_time > 0 else 0,
            'success_rate': successful_items / total_items if total_items > 0 else 0,
            'error_rate': failed_items / total_items if total_items > 0 else 0
        }
        
        # 构建质量指标对象
        quality_metrics = QualityMetrics(**job.quality_metrics) if job.quality_metrics else QualityMetrics()
        
        return ProcessingReport(
            job_id=job.job_id,
            total_items=total_items,
            processed_items=processed_items,
            successful_items=successful_items,
            failed_items=failed_items,
            processing_time=processing_time,
            average_time_per_item=average_time_per_item,
            quality_metrics=quality_metrics,
            error_summary=dict(error_summary),
            performance_stats=performance_stats
        )
    
    def _update_performance_stats(self, report: ProcessingReport):
        """更新性能统计"""
        self.performance_stats['total_jobs'] += 1
        self.performance_stats['total_items_processed'] += report.total_items
        
        # 更新平均处理时间
        total_time = (self.performance_stats['average_processing_time'] * 
                     (self.performance_stats['total_jobs'] - 1) + report.processing_time)
        self.performance_stats['average_processing_time'] = total_time / self.performance_stats['total_jobs']
        
        # 更新错误率
        total_error_rate = (self.performance_stats['error_rate'] * 
                           (self.performance_stats['total_jobs'] - 1) + 
                           report.performance_stats['error_rate'])
        self.performance_stats['error_rate'] = total_error_rate / self.performance_stats['total_jobs']
        
        # 质量分布
        self.performance_stats['quality_distribution'][report.quality_metrics.quality_level.value] += 1
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """📋 获取任务状态"""
        if job_id not in self.jobs:
            return {"error": f"任务 {job_id} 不存在"}
        
        job = self.jobs[job_id]
        
        return {
            "job_id": job.job_id,
            "name": job.name,
            "status": job.status.value,
            "progress": job.progress,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "total_items": len(job.input_data),
            "processed_items": len(job.results) + len(job.errors),
            "errors_count": len(job.errors)
        }
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """📊 获取性能仪表板"""
        return {
            "总体统计": self.performance_stats,
            "活跃任务": len([j for j in self.jobs.values() if j.status == ProcessingStatus.PROCESSING]),
            "待处理任务": len(self.processing_queue),
            "已完成任务": len(self.completed_jobs),
            "质量分布": dict(self.performance_stats['quality_distribution'])
        }
    
    def export_report(self, job_id: str, output_path: str):
        """💾 导出报告"""
        if job_id not in self.jobs:
            raise ValueError(f"任务 {job_id} 不存在")
        
        job = self.jobs[job_id]
        if job.status != ProcessingStatus.COMPLETED:
            raise ValueError(f"任务 {job_id} 尚未完成")
        
        report = self._generate_report(job)
        
        export_data = {
            "report": asdict(report),
            "job_details": asdict(job),
            "export_time": datetime.now().isoformat()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"💾 报告已导出到: {output_path}")
    
    def cleanup_completed_jobs(self, keep_recent: int = 10):
        """🧹 清理已完成的任务"""
        if len(self.completed_jobs) > keep_recent:
            to_remove = self.completed_jobs[:-keep_recent]
            for job_id in to_remove:
                if job_id in self.jobs:
                    del self.jobs[job_id]
            self.completed_jobs = self.completed_jobs[-keep_recent:]
            
            print(f"🧹 清理了 {len(to_remove)} 个已完成任务")
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


# 使用示例和测试
def demo_batch_processor():
    """演示批量处理器"""
    print("📊 Batch Processor Demo")
    print("=" * 50)
    
    # 创建处理器
    processor = BatchProcessor(max_workers=2, enable_monitoring=True)
    
    # 示例处理函数
    def simple_math_processor(problem):
        """简单数学处理器示例"""
        time.sleep(0.1)  # 模拟处理时间
        
        if isinstance(problem, dict) and 'expression' in problem:
            try:
                result = eval(problem['expression'])
                return {
                    'problem': problem,
                    'result': result,
                    'is_correct': True,
                    'solution_steps': [f"计算 {problem['expression']} = {result}"]
                }
            except:
                return {
                    'problem': problem,
                    'result': None,
                    'is_correct': False,
                    'error': '计算失败'
                }
        
        return {'problem': problem, 'result': None, 'is_correct': False}
    
    # 准备测试数据
    test_data = [
        {'expression': '2 + 3'},
        {'expression': '5 * 4'},
        {'expression': '10 / 2'},
        {'expression': '7 - 3'},
        {'expression': '6 + 8'},
        {'expression': 'invalid'},  # 故意的错误数据
    ]
    
    # 提交任务
    job_id = processor.submit_job(
        name="数学计算测试",
        input_data=test_data,
        processor_func=simple_math_processor,
        quality_evaluator='math_problem_solver'
    )
    
    print(f"📤 提交任务: {job_id}")
    
    # 处理任务
    report = processor.process_job(job_id)
    
    # 显示结果
    print(f"\n📋 处理报告:")
    print(f"  任务ID: {report.job_id}")
    print(f"  总项目数: {report.total_items}")
    print(f"  成功项目: {report.successful_items}")
    print(f"  失败项目: {report.failed_items}")
    print(f"  处理时间: {report.processing_time:.2f}秒")
    print(f"  质量等级: {report.quality_metrics.quality_level.value}")
    print(f"  整体分数: {report.quality_metrics.overall_score:.2f}")
    
    if report.quality_metrics.recommendations:
        print(f"  改进建议:")
        for rec in report.quality_metrics.recommendations:
            print(f"    - {rec}")
    
    # 显示性能仪表板
    dashboard = processor.get_performance_dashboard()
    print(f"\n📊 性能仪表板:")
    for key, value in dashboard.items():
        print(f"  {key}: {value}")
    
    return processor, report


if __name__ == "__main__":
    demo_batch_processor() 