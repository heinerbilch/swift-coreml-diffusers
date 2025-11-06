//
//  PipelineLoader.swift
//  Diffusion
//
//  Created by Pedro Cuenca on December 2022.
//  See LICENSE at https://github.com/huggingface/swift-coreml-diffusers/LICENSE
//


import Combine
import Foundation
import CandleSD15Bridge
import Hub

class PipelineLoader {
    let maxSeed: UInt32
    
    private var downloadSubscriber: Cancellable?
    private let downloadProgress = PassthroughSubject<Double, Never>()

    var snapshotURL: URL? = nil

    init(maxSeed: UInt32 = UInt32.max) {
        self.maxSeed = maxSeed
        state = .undetermined
        setInitialState()

        downloadSubscriber = downloadProgress
            .receive(on: DispatchQueue.main)
            .sink { [weak self] progress in
                self?.state = .downloading(progress)
            }
    }

    enum PipelinePreparationPhase {
        case undetermined
        case waitingToDownload
        case downloading(Double)
        case downloaded
        case loaded
        case failed(Error)
    }
    
    var state: PipelinePreparationPhase {
        didSet {
            statePublisher.value = state
        }
    }
    private(set) lazy var statePublisher: CurrentValueSubject<PipelinePreparationPhase, Never> = CurrentValueSubject(state)

    func setInitialState() {
        if ready {
            state = .downloaded
            return
        }
        if downloaded {
            state = .downloaded
            return
        }
        state = .waitingToDownload
    }
}

extension PipelineLoader {
    // TODO: do something
    func cancel() { }
}

extension PipelineLoader {
    var downloaded: Bool {
        guard let snapshotURL = snapshotURL else { return false }
        return FileManager.default.fileExists(atPath: snapshotURL.path)
    }
    
    var ready: Bool { downloaded }

    func prepare() async throws -> Pipeline {
        do {
            try await download()
            guard let snapshotURL = snapshotURL else {
                fatalError("Download failed")
            }
            try await load(url: snapshotURL)
            return Pipeline(maxSeed: maxSeed)
        } catch {
            state = .failed(error)
            throw error
        }
    }

    // Currently unused
    @discardableResult
    func download_sdxl_turbo() async throws -> URL {
        // TODO: download model from stabilityai/sdxl-turbo, and download separately 2 x tokenizer.json from openai/clip-vit-large-patch14 and laion/CLIP-ViT-bigG-14-laion2B-39B-b160k
        // as well as the fp16-fixed vae from https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/tree/main
        let downloadedTo = try await Hub.snapshot(from: "pcuenq/sdxl-turbo", matching: ["*.fp16.safetensors", "*.json"]) { progress in
            print(progress.fractionCompleted)
            self.downloadProgress.send(progress.fractionCompleted)
            if progress.fractionCompleted >= 1.0 {
                self.downloadProgress.send(completion: .finished)
            }
        }
        print("Downloaded to \(downloadedTo)")
        snapshotURL = downloadedTo
        state = .downloaded
        return downloadedTo
    }

    // Currently unused
    @discardableResult
    func download() async throws -> URL {
        // TODO: download model from stable-diffusion-v1-5/stable-diffusion-v1-5, and download separately tokenizer.json from openai/clip-vit-base-patch32
        let downloadedTo = try await Hub.snapshot(from: "pcuenq/stable-diffusion-v1-5", matching: ["*.fp16.safetensors", "*.json"]) { progress in
            print(progress.fractionCompleted)
            self.downloadProgress.send(progress.fractionCompleted)
            if progress.fractionCompleted >= 1.0 {
                self.downloadProgress.send(completion: .finished)
            }
        }
        print("Downloaded to \(downloadedTo)")
        snapshotURL = downloadedTo
        state = .downloaded
        return downloadedTo
    }

    // TODO: don't we have anything to capture state / instantiation?
    // return it when we do
    func load(url: URL) async throws {
        guard let snapshotURL = snapshotURL else {
            throw "Model not downloaded"
        }
        let beginDate = Date()

        // TODO: improve these symbols / API
        // Ensure we pass a stable UnsafePointer<CChar> to the C API
        let initStatus: CandleSdStatusCode = snapshotURL.path.withCString { cPath in
            var options = CandleSdInitOptions(asset_dir: cPath, use_metal: 1)
            return candle_sd_init(&options)
        }

        guard initStatus == Ok else {
            let lastError = String(cString: candle_sd_last_error())
            print("Candle error: \(lastError))")
            fatalError("Generation error")
        }

        print("Pipeline loaded in \(Date().timeIntervalSince(beginDate))")
        state = .loaded
//        return pipeline
    }
}

