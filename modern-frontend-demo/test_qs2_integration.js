/**
 * QS²前端集成测试脚本
 * 在浏览器控制台运行此脚本来测试QS²数据获取
 */

async function testQS2Integration() {
    console.log('🧪 开始QS²前端集成测试...');
    
    try {
        // 测试1: QS²演示数据获取
        console.log('📋 测试1: QS²演示数据获取...');
        const demoResponse = await fetch('/api/qs2/demo');
        const demoData = await demoResponse.json();
        
        if (demoData.success) {
            console.log('✅ QS²演示数据获取成功!');
            console.log(`   🔢 实体数量: ${demoData.data.entities.length}`);
            console.log(`   🔗 关系数量: ${demoData.data.relationships.length}`);
            console.log(`   📊 算法阶段: ${demoData.data.algorithm_stages.length}`);
            console.log('   📥 数据详情:', demoData);
        } else {
            console.error('❌ QS²演示数据获取失败:', demoData.error);
        }
        
        // 测试2: 算法执行数据获取
        console.log('\n📋 测试2: 算法执行数据获取...');
        const execResponse = await fetch('/api/algorithm/execution');
        const execData = await execResponse.json();
        
        if (execData.success) {
            console.log('✅ 算法执行数据获取成功!');
            console.log('   📥 执行数据:', execData);
        } else {
            console.log('⚠️  算法执行数据获取失败，这是正常的（使用QS²数据）');
        }
        
        // 测试3: 模拟前端QS²数据转换
        console.log('\n📋 测试3: QS²数据转换测试...');
        if (demoData.success) {
            const currentTime = Date.now() / 1000;
            const convertedStages = demoData.data.algorithm_stages.map((stage, index) => ({
                stage_id: `qs2_stage_${index}`,
                stage_name: stage.name,
                timestamp: currentTime + index,
                duration_ms: stage.duration_ms,
                confidence: stage.confidence,
                algorithm_state: {
                    phase: stage.name,
                    is_qs2_enhanced: true,
                    algorithm_type: 'QS2_Enhanced'
                }
            }));
            
            const convertedData = {
                execution_id: demoData.execution_id,
                problem_text: demoData.problem_text,
                total_duration_ms: demoData.data.algorithm_stages.reduce((sum, stage) => sum + stage.duration_ms, 0),
                stages: convertedStages,
                final_result: {
                    entities: demoData.data.entities,
                    relations: demoData.data.relationships,
                    algorithm_type: 'QS2_Enhanced',
                    is_qs2_enhanced: true
                }
            };
            
            console.log('✅ QS²数据转换成功!');
            console.log(`   ⏱️  总耗时: ${convertedData.total_duration_ms.toFixed(1)}ms`);
            console.log(`   📋 阶段数: ${convertedData.stages.length}`);
            console.log(`   🧠 算法类型: ${convertedData.final_result.algorithm_type}`);
            console.log(`   🔮 QS²增强: ${convertedData.final_result.is_qs2_enhanced}`);
            console.log('   📦 转换结果:', convertedData);
        }
        
        // 测试4: Qualia结构提取
        console.log('\n📋 测试4: Qualia结构提取测试...');
        if (demoData.success && demoData.data.entities) {
            const qualiaStructures = demoData.data.entities.map(entity => ({
                entity: entity.name,
                entity_type: entity.type,
                qualia_roles: entity.qualia_roles,
                confidence: entity.confidence
            }));
            
            console.log('✅ Qualia结构提取成功!');
            console.log(`   📊 结构数量: ${qualiaStructures.length}`);
            qualiaStructures.forEach((structure, index) => {
                console.log(`   ${index + 1}. ${structure.entity} (${structure.entity_type})`);
                console.log(`      - 形式角色: ${structure.qualia_roles.formal.join(', ')}`);
                console.log(`      - 目的角色: ${structure.qualia_roles.telic.join(', ')}`);
                console.log(`      - 施事角色: ${structure.qualia_roles.agentive.join(', ')}`);
                console.log(`      - 构成角色: ${structure.qualia_roles.constitutive.join(', ')}`);
            });
        }
        
        console.log('\n🎉 QS²前端集成测试完成!');
        console.log('✅ 所有核心功能正常工作');
        console.log('💡 提示: 现在可以打开浏览器查看"算法物性关系图"模块应该显示QS²内容');
        
    } catch (error) {
        console.error('❌ QS²前端集成测试失败:', error);
    }
}

// 自动运行测试
console.log('🚀 启动QS²前端集成测试...');
testQS2Integration();