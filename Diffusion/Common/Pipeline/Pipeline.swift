//
//  Pipeline.swift
//  Diffusion
//
//  Created by Pedro Cuenca on December 2022.
//  See LICENSE at https://github.com/huggingface/swift-coreml-diffusers/LICENSE
//

import Foundation
import Combine
import CandleSDXL
import CoreGraphics

//struct StableDiffusionProgress {
//    var progress: StableDiffusionPipeline.Progress
//
//    var step: Int { progress.step }
//    var stepCount: Int { progress.stepCount }
//
//    var currentImages: [CGImage?]
//
//    init(progress: StableDiffusionPipeline.Progress, previewIndices: [Bool]) {
//        self.progress = progress
//        self.currentImages = [nil]
//
//        // Since currentImages is a computed property, only access the preview image if necessary
//        if progress.step < previewIndices.count, previewIndices[progress.step] {
//            self.currentImages = progress.currentImages
//        }
//    }
//}

struct GenerationResult {
    var image: CGImage?
    var lastSeed: UInt32
    var interval: TimeInterval?
    var userCanceled: Bool
    var itsPerSecond: Double?
}

class Pipeline {
    let maxSeed: UInt32
    
    var isXL: Bool {
        return false
    }

//    var progress: StableDiffusionProgress? = nil {
//        didSet {
//            progressPublisher.value = progress
//        }
//    }
//    lazy private(set) var progressPublisher: CurrentValueSubject<StableDiffusionProgress?, Never> = CurrentValueSubject(progress)
    
    private var canceled = false

    init(maxSeed: UInt32 = UInt32.max) {
        self.maxSeed = maxSeed
    }


    func debugPrintImageStats(_ image: candle_sdxlCandleSdxlImage) {
        guard let base = image.data else {
            print("Image buffer pointer was nil.")
            return
        }
        let length = Int(image.len)
        guard length > 0 else {
            print("Image buffer is empty.")
            return
        }

        // Peek at the raw bytes without copying.
        let buffer = UnsafeBufferPointer(start: base, count: length)

        let nonZeroCount = buffer.lazy.filter { $0 != 0 }.count
        let minValue = buffer.min() ?? 0
        let maxValue = buffer.max() ?? 0
        let sample = buffer.prefix(16).map { String(format: "%02X", $0) }.joined(separator: " ")

        print("""
        CandleSdxlImage stats:
          total bytes   : \(length)
          width x height: \(image.width) x \(image.height)
          non-zero bytes: \(nonZeroCount)
          min/max values: \(minValue) / \(maxValue)
          first 16 bytes: \(sample)
        """)
    }

    func makeCGImage(from image: inout candle_sdxlCandleSdxlImage) -> CGImage? {
        // Expect Candle to return 8‑bit RGBA.
        guard image.channels == 4, let baseAddress = image.data else { return nil }

        // Copy data and free candle buffer
        let width = Int(image.width)
        let height = Int(image.height)
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        let byteCount = width * height * bytesPerPixel
        assert(byteCount == image.len)

        let buffer = UnsafeBufferPointer(start: baseAddress, count: byteCount)
        let pixelData = Data(buffer: buffer)
        candle_sdxl_free_image(&image)

        let bitmapInfo = CGBitmapInfo([
            .byteOrder32Big,
            CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue)
        ])
        guard let provider = CGDataProvider(data: pixelData as CFData) else { return nil }
        return CGImage(
            width: width,
            height: height,
            bitsPerComponent: 8,
            bitsPerPixel: 32,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),   // should be sRGB
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        )
    }

    func generate(
        prompt: String,
        numInferenceSteps stepCount: Int = 1,
        seed: UInt32 = 0,
    ) throws -> GenerationResult {
        canceled = false
        let theSeed = seed > 0 ? seed : UInt32.random(in: 1...maxSeed)

        // TODO: improve these symbols / API
        var image = candle_sdxlCandleSdxlImage()
        let status: candle_sdxlCandleSdxlStatusCode = prompt.withCString { promptCString in
            var request = candle_sdxlCandleSdxlRequest(
                prompt: promptCString,
                steps: UInt32(stepCount),
                seed: UInt64(theSeed),
                use_seed: 1
            )
            let s = candle_sdxl_generate(&request, &image)
            return s
        }

        guard status == Ok else {
            let lastError = String(cString: candle_sdxl_last_error())
            print("Candle error: \(lastError))")
            fatalError("Generation error")
        }

        debugPrintImageStats(image)
        let cgImage = makeCGImage(from: &image)
        return GenerationResult(image: cgImage, lastSeed: theSeed, userCanceled: false)

//        // Evenly distribute previews based on inference steps
//        let previewIndices = previewIndices(stepCount, previewCount)
//
//        let images = try pipeline.generateImages(configuration: config) { progress in
//            sampleTimer.stop()
//            handleProgress(StableDiffusionProgress(progress: progress,
//                                                   previewIndices: previewIndices),
//                           sampleTimer: sampleTimer)
//            if progress.stepCount != progress.step {
//                sampleTimer.start()
//            }
//            return !canceled
//        }
//        let interval = Date().timeIntervalSince(beginDate)
//        print("Got images: \(images) in \(interval)")
//        
//        // Unwrap the 1 image we asked for, nil means safety checker triggered
//        let image = images.compactMap({ $0 }).first
//        return GenerationResult(image: image, lastSeed: theSeed, interval: interval, userCanceled: canceled, itsPerSecond: 1.0/sampleTimer.median)
    }

//    func handleProgress(_ progress: StableDiffusionProgress, sampleTimer: SampleTimer) {
//        self.progress = progress
//    }
        
    func setCancelled() {
        canceled = true
    }
}
