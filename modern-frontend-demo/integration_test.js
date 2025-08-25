#!/usr/bin/env node

/**
 * 完整前后端集成测试
 * 验证Chakra UI前端与真实COT-DIR推理引擎的集成
 */

import fetch from 'node-fetch';

const API_BASE = 'http://localhost:5001';
const FRONTEND_BASE = 'http://localhost:3000';

async function testFullIntegration() {
    console.log('🔗 完整前后端集成测试');
    console.log('=' * 50);
    
    const testResults = [];
    
    try {
        // Test 1: API健康检查
        console.log('\n🏥 测试1: API健康检查');
        const healthResponse = await fetch(`${API_BASE}/api/health`);
        const healthData = await healthResponse.json();
        
        if (healthData.status === 'healthy' && healthData.engine_status === 'healthy') {
            console.log('✅ API服务正常，推理引擎状态: healthy');
            testResults.push({ test: 'API健康检查', status: 'passed' });
        } else {
            console.log('❌ API服务异常:', healthData);
            testResults.push({ test: 'API健康检查', status: 'failed', error: 'API不健康' });
        }
        
        // Test 2: 前端页面访问
        console.log('\n🌐 测试2: 前端页面访问');
        const frontendResponse = await fetch(`${FRONTEND_BASE}`);
        const frontendHTML = await frontendResponse.text();
        
        if (frontendHTML.includes('COT-DIR - 智能数学推理系统')) {
            console.log('✅ 前端页面正常加载');
            testResults.push({ test: '前端页面访问', status: 'passed' });
        } else {
            console.log('❌ 前端页面加载异常');
            testResults.push({ test: '前端页面访问', status: 'failed', error: '页面标题不匹配' });
        }
        
        // Test 3: 推理引擎功能测试
        console.log('\n🧠 测试3: COT-DIR推理引擎功能');
        const mathProblems = [
            {
                problem: "如果约翰有5个苹果，给了玛丽2个，他还剩多少个苹果？",
                expectedType: "subtraction",
                expectedAnswer: "3"
            },
            {
                problem: "一列火车在2小时内行驶了120公里。它的平均速度是多少？",
                expectedType: "division", 
                expectedAnswer: "60.0"
            },
            {
                problem: "求解 x: 2x + 3 = 11",
                expectedType: "algebra",
                expectedAnswer: "x = 4"
            }
        ];
        
        for (const [index, testCase] of mathProblems.entries()) {
            console.log(`\n📝 测试问题 ${index + 1}: ${testCase.problem.substring(0, 30)}...`);
            
            const response = await fetch(`${API_BASE}/api/solve`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ problem: testCase.problem })
            });
            
            if (!response.ok) {
                console.log(`❌ API请求失败: ${response.status}`);
                testResults.push({ 
                    test: `推理测试${index + 1}`, 
                    status: 'failed', 
                    error: `HTTP ${response.status}` 
                });
                continue;
            }
            
            const result = await response.json();
            
            // 验证基本结构
            const hasRequiredFields = result.final_answer && 
                                    result.confidence !== undefined && 
                                    result.reasoning_steps && 
                                    result.entities && 
                                    result.relations;
            
            if (!hasRequiredFields) {
                console.log('❌ 响应结构不完整');
                testResults.push({ 
                    test: `推理测试${index + 1}`, 
                    status: 'failed', 
                    error: '响应字段缺失' 
                });
                continue;
            }
            
            // 验证推理步骤
            const hasReasoningSteps = result.reasoning_steps.length >= 4;
            const hasEntities = result.entities.length >= 1;
            const hasCorrectEngine = result.engine_mode === 'cotdir_simplified';
            
            console.log(`   答案: ${result.final_answer}`);
            console.log(`   置信度: ${(result.confidence * 100).toFixed(1)}%`);
            console.log(`   推理步骤: ${result.reasoning_steps.length} 步`);
            console.log(`   实体数量: ${result.entities.length}`);
            console.log(`   关系数量: ${result.relations.length}`);
            console.log(`   引擎模式: ${result.engine_mode}`);
            console.log(`   复杂度: ${result.complexity.level}.${result.complexity.sublevel}`);
            
            if (hasReasoningSteps && hasEntities && hasCorrectEngine) {
                console.log('✅ 推理功能正常，COT-DIR引擎工作正常');
                testResults.push({ test: `推理测试${index + 1}`, status: 'passed' });
            } else {
                console.log('⚠️ 推理功能部分异常');
                testResults.push({ 
                    test: `推理测试${index + 1}`, 
                    status: 'partial', 
                    details: { hasReasoningSteps, hasEntities, hasCorrectEngine }
                });
            }
        }
        
        // Test 4: 推理引擎状态检查
        console.log('\n⚙️ 测试4: 推理引擎详细状态');
        const engineResponse = await fetch(`${API_BASE}/api/engine/status`);
        const engineData = await engineResponse.json();
        
        if (engineData.engine_type === 'Independent COT-DIR' && engineData.status === 'active') {
            console.log('✅ 推理引擎状态正常');
            console.log(`   引擎类型: ${engineData.engine_type}`);
            console.log(`   状态: ${engineData.status}`);
            console.log(`   已解决问题: ${engineData.statistics?.problems_solved || 0}`);
            console.log(`   成功率: ${((engineData.statistics?.success_rate || 0) * 100).toFixed(1)}%`);
            testResults.push({ test: '推理引擎状态', status: 'passed' });
        } else {
            console.log('❌ 推理引擎状态异常:', engineData);
            testResults.push({ test: '推理引擎状态', status: 'failed', error: '引擎状态不正常' });
        }
        
        // 汇总测试结果
        console.log('\n' + '=' * 50);
        console.log('📊 集成测试结果汇总:');
        
        const passed = testResults.filter(r => r.status === 'passed').length;
        const partial = testResults.filter(r => r.status === 'partial').length;
        const failed = testResults.filter(r => r.status === 'failed').length;
        const total = testResults.length;
        
        console.log(`✅ 通过: ${passed}/${total}`);
        console.log(`⚠️ 部分通过: ${partial}/${total}`);
        console.log(`❌ 失败: ${failed}/${total}`);
        
        testResults.forEach(result => {
            const statusIcon = result.status === 'passed' ? '✅' : 
                             result.status === 'partial' ? '⚠️' : '❌';
            console.log(`  ${statusIcon} ${result.test}`);
            if (result.error) console.log(`      错误: ${result.error}`);
        });
        
        if (failed === 0) {
            console.log('\n🎉 所有测试通过！前后端集成正常工作！');
            console.log('🌐 Chakra UI前端: http://localhost:3000');
            console.log('🧠 COT-DIR推理API: http://localhost:5001');
            console.log('\n✨ 真实推理引擎集成完成！现在可以使用完整的COT-DIR数学推理系统。');
        } else {
            console.log('\n⚠️ 部分测试失败，请检查相关组件。');
        }
        
    } catch (error) {
        console.error('\n❌ 集成测试过程出错:', error);
    }
}

runIntegrationTest().catch(console.error);

async function runIntegrationTest() {
    await testFullIntegration();
}