#!/usr/bin/env node

/**
 * Integration Test Script for Chakra UI Mathematical Reasoning Frontend
 * Tests the complete flow from frontend to API
 */

import fetch from 'node-fetch';

const API_BASE = 'http://localhost:5001';
const FRONTEND_BASE = 'http://localhost:3004';

async function testAPIHealth() {
    console.log('🔍 Testing API Health...');
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();
        console.log('✅ API Health:', data.status);
        return true;
    } catch (error) {
        console.error('❌ API Health failed:', error.message);
        return false;
    }
}

async function testMathProblemSolving() {
    console.log('🧮 Testing Mathematical Problem Solving...');
    
    const testProblems = [
        "If John has 5 apples and gives 2 to Mary, how many apples does he have left?",
        "A train travels 120 km in 2 hours. What is its average speed?",
        "Solve for x: 2x + 3 = 11"
    ];

    for (const problem of testProblems) {
        try {
            console.log(`\n📝 Testing: ${problem.substring(0, 50)}...`);
            
            const response = await fetch(`${API_BASE}/api/solve`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ problem })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const result = await response.json();
            
            console.log(`✅ Answer: ${result.final_answer || result.answer}`);
            console.log(`📊 Confidence: ${Math.round(result.confidence * 100)}%`);
            console.log(`🔬 Complexity: ${result.complexity.level}.${result.complexity.sublevel}`);
            console.log(`🎯 Steps: ${result.reasoning_steps.length}`);
            console.log(`🧩 Entities: ${result.entities.length}`);
            console.log(`🔗 Relations: ${result.relations.length}`);
            
        } catch (error) {
            console.error(`❌ Problem failed: ${error.message}`);
            return false;
        }
    }
    
    return true;
}

async function testExamplesEndpoint() {
    console.log('\n📚 Testing Examples Endpoint...');
    try {
        const response = await fetch(`${API_BASE}/api/examples`);
        const data = await response.json();
        console.log(`✅ Examples loaded: ${data.examples.length} problems`);
        return true;
    } catch (error) {
        console.error('❌ Examples endpoint failed:', error.message);
        return false;
    }
}

async function runTests() {
    console.log('🚀 Starting Chakra UI Mathematical Reasoning Integration Tests');
    console.log('='.repeat(65));
    
    const results = [];
    
    results.push(await testAPIHealth());
    results.push(await testMathProblemSolving());
    results.push(await testExamplesEndpoint());
    
    console.log('\n' + '='.repeat(65));
    console.log('📊 Test Results Summary:');
    console.log(`✅ Passed: ${results.filter(r => r).length}/${results.length}`);
    console.log(`❌ Failed: ${results.filter(r => !r).length}/${results.length}`);
    
    if (results.every(r => r)) {
        console.log('\n🎉 All integration tests passed!');
        console.log(`🌐 Frontend available at: ${FRONTEND_BASE}`);
        console.log(`📡 API available at: ${API_BASE}`);
        console.log('\n✨ Chakra UI Mathematical Reasoning System is ready!');
    } else {
        console.log('\n⚠️ Some tests failed. Check the logs above.');
        process.exit(1);
    }
}

runTests().catch(console.error);