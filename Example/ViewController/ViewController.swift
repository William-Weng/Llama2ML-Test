//
//  ViewController.swift
//  Example
//
//  Created by William.Weng on 2024/9/21.
//

import UIKit
import CoreML

// MARK: - ViewController
final class ViewController: UIViewController {

    private var llamaRunner: LlamaManualRunner?

    override func viewDidLoad() {
        super.viewDidLoad()
        
        // 初始化 Llama 執行器
        self.llamaRunner = LlamaManualRunner()
        
        // 示範：點擊按鈕時觸發
    }
    
    @IBAction func test(_ sender: UIButton) {
        // 重要：如此處所示，這個 dummy tokenizer 只能處理簡單的英文/數字字元
        // 無法處理中文或複雜句子。
        let prompt = "Hello"
        
        print("使用者輸入: \(prompt)")
        
        Task(priority: .userInitiated) {
            await llamaRunner?.generate(prompt: prompt)
        }
    }
}

// MARK: - Llama 手動執行器
class LlamaManualRunner {
    
    private let model: Llama_3_2_1B
    private let inputShape: [NSNumber] = [1, 128] // 輸入形狀，1個批次，128個 token 長度
    private let maxTokenLength = 128

    init?() {
        do {
            let config = MLModelConfiguration()
            self.model = try Llama_3_2_1B(configuration: config)
        } catch {
            print("Error loading model: \(error)")
            return nil
        }
    }

    func generate(prompt: String) async {
        
        // 1. 分詞 (Tokenization) - *** 這是虛設的，非真實的分詞器 ***
        var tokens = dummyTokenize(prompt)
        
        // 開始生成循環
        for _ in 0..<maxTokenLength - 1 {
            
            // 2. 準備模型輸入 (MLMultiArray)
            guard let inputArray = try? MLMultiArray(shape: inputShape, dataType: .float32) else {
                print("Error creating MLMultiArray")
                return
            }
            
            // 將目前的 token 序列填入 inputArray
            for (index, token) in tokens.enumerated() {
                if index >= maxTokenLength { break }
                inputArray[index] = NSNumber(value: token)
            }
            
            // 3. 執行模型預測
            let input = Llama_3_2_1BInput(input_ids: inputArray)
            guard let output = try? model.prediction(input: input) else {
                print("Error during model prediction")
                return
            }
            
            // 4. 解碼 (Decoding) - 找出下一個 token
            // output.var_2609 的形狀是 [1, 128, 32000]，代表每個位置下一個 token 的分數
            let outputLogits = output.var_2609
            
            // 我們只關心序列中最後一個有效 token 的下一個預測
            let lastTokenIndex = tokens.count - 1
            let nextTokenLogits = logits(from: outputLogits, at: lastTokenIndex)
            
            // 使用 argmax 找出分數最高的 token ID (最可能的下一個 token)
            guard let nextToken = argmax(array: nextTokenLogits) else {
                print("Could not find next token")
                return
            }
            
            // 5. 檢查是否為結束符號 (End-of-Sequence token)
            // Llama 3 的 EOS token ID 之一是 128009。這裡我們假設一個簡單的數字。
            if nextToken == 2 { // 假設 2 是結束符號
                print("\n[結束]" )
                break
            }
            
            // 將新生成的 token 加入序列
            tokens.append(nextToken)
            
            // 6. 反分詞 (De-tokenize) - *** 這是虛設的 ***
            let nextCharacter = dummyDetokenize([nextToken])
            print(nextCharacter, terminator: "")
        }
        print("\n")
    }
    
    /// 從模型輸出的 MLMultiArray 中提取特定位置的 logits 陣列
    private func logits(from multiArray: MLMultiArray, at index: Int) -> [Float] {
        let vocabularySize = 32000 // Llama 3 詞彙表示量
        var logits: [Float] = []
        let basePointer = multiArray.dataPointer.bindMemory(to: Float32.self, capacity: multiArray.count)
        
        // 計算起始位置
        let startIndex = index * vocabularySize
        
        for i in 0..<vocabularySize {
            logits.append(basePointer[startIndex + i])
        }
        
        return logits
    }
    
    /// 找出陣列中最大值的索引
    private func argmax(array: [Float]) -> Int? {
        return array.indices.max { array[$0] < array[$1] }
    }
    
    // ---
    // 以下是虛設的分詞/反分詞器
    // ---
    
    /// **虛設分詞器**: 將字串轉為假的 token ID (ASCII value)
    private func dummyTokenize(_ text: String) -> [Int] {
        // BOS (Begin-of-Sequence) token, Llama3 常用的 ID 是 128000
        var tokens = [128000]
        tokens.append(contentsOf: text.compactMap { Int($0.asciiValue ?? 0) })
        return tokens
    }
    
    /// **虛設反分詞器**: 將假的 token ID 轉回字元
    private func dummyDetokenize(_ tokens: [Int]) -> String {
        return tokens.map { Character(UnicodeScalar($0) ?? " ") }.map(String.init).joined()
    }
}
